"""
DINOv2 ViT-L14 + 纹理增强特征提取
针对陨石识别优化：
- CLAHE 局部对比度增强：放大熔壳裂纹、气印等表面细节
- Sobel 边缘增强：强调不规则外形和表面纹理
- 多视角 TTA 融合原始 + 纹理增强特征
"""
from pathlib import Path
import sys
import json
import hashlib
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps, ImageFile, ImageFilter
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 0. 配置
# ==========================================
DATA_ROOT = Path("/data/数据科学实践project")

# 伪标签配置：原始概率文件，Top-N 陨石 + Bottom-N 非陨石
PROB_FILE = DATA_ROOT / "dinov2_simple_output" / "test_probs_dinov2.csv"
PSEUDO_TOP_N = 30
PSEUDO_BOTTOM_N = 30
PSEUDO_WEIGHT = 2.0

OUTPUT_DIR = DATA_ROOT / "runs" / f"dinov2_texture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4
VAL_RATIO = 0.2

print(f"DEVICE: {DEVICE}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"PROB_FILE: {PROB_FILE}")
print(f"PSEUDO: Top-{PSEUDO_TOP_N} 陨石 + Bottom-{PSEUDO_BOTTOM_N} 非陨石, Weight={PSEUDO_WEIGHT}x")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==========================================
# 1. 纹理增强函数 — 陨石特征专用
# ==========================================
def clahe_enhance(image):
    """
    自适应直方图均衡化 (CLAHE)。
    增强局部对比度，放大表面纹理：
    - 熔壳微裂纹（Fusion Crust cracks）
    - 气印/拇指纹（Regmaglypts）
    - 流线（Flow lines）
    """
    import cv2
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def sobel_texture_map(image):
    """
    Sobel 边缘检测 → 3通道纹理图。
    强调表面形态学特征：
    - 不规则外形边缘（sculpted, irregular shape）
    - 表面凹凸纹理（regmaglypts 边缘轮廓）
    """
    import cv2
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_mag = (edge_mag / (edge_mag.max() + 1e-8) * 255).astype(np.uint8)
    # 3通道
    edge_3ch = np.stack([edge_mag, edge_mag, edge_mag], axis=-1)
    return Image.fromarray(edge_3ch)


# ==========================================
# 2. 数据集类
# ==========================================
class MeteoriteDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transforms = transforms
        self.has_label = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["img_path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        if self.has_label:
            return image, int(row["label"])
        return image, str(row["img_path"])


# ==========================================
# 3. 数据加载与去重切分 (与之前相同)
# ==========================================
def _resolve_image_dir(root, split):
    candidates = [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"找不到 {split} 图片目录")


def load_train_dataframe(data_root):
    data_root = Path(data_root)
    csv_path = data_root / "train_labels.csv"
    train_dir = _resolve_image_dir(data_root, "train")
    df = pd.read_csv(csv_path)
    df["img_path"] = df["id"].apply(lambda x: str(train_dir / x))
    exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
    df = df[exists_mask].copy()
    df["label"] = df["label"].astype(int)
    df["source"] = "real"
    return df


def load_test_dataframe(data_root):
    data_root = Path(data_root)
    csv_path = data_root / "sample_submission.csv"
    test_dir = _resolve_image_dir(data_root, "test")
    df = pd.read_csv(csv_path)
    df = df[["id"]].copy()
    df["img_path"] = df["id"].apply(lambda x: str(test_dir / x))
    exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
    df = df[exists_mask].copy()
    return df


def load_pseudo_labeled_data(data_root, prob_path, top_n, bottom_n):
    """从概率文件中选取最高 top_n 作为陨石(1)，最低 bottom_n 作为非陨石(0)。"""
    data_root = Path(data_root)
    test_dir = _resolve_image_dir(data_root, "test")
    prob_df = pd.read_csv(prob_path)
    prob_df = prob_df.sort_values('prob', ascending=False).reset_index(drop=True)

    top_df = prob_df.head(top_n)[['id']].copy()
    top_df['label'] = 1
    bot_df = prob_df.tail(bottom_n)[['id']].copy()
    bot_df['label'] = 0

    pseudo_df = pd.concat([top_df, bot_df], ignore_index=True)
    pseudo_df['img_path'] = pseudo_df['id'].apply(lambda x: str(test_dir / x))
    exists_mask = pseudo_df['img_path'].apply(lambda p: Path(p).exists())
    pseudo_df = pseudo_df[exists_mask].copy()
    pseudo_df['label'] = pseudo_df['label'].astype(int)
    pseudo_df['source'] = 'pseudo'
    return pseudo_df


def compute_file_md5(path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def attach_md5_group_ids(df, path_col="img_path", group_col="group_id"):
    df = df.copy()
    df[group_col] = df[path_col].apply(compute_file_md5)
    return df


def group_aware_train_val_split(df, label_col="label", group_col="group_id", val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    train_parts, val_parts = [], []
    for cls, cls_df in df.groupby(label_col):
        group_items = [(gid, gdf.index.tolist()) for gid, gdf in cls_df.groupby(group_col)]
        rng.shuffle(group_items)
        target_val_count = max(1, int(round(len(cls_df) * val_ratio)))
        val_indices, train_indices, current_val_count = [], [], 0
        for _, indices in group_items:
            if current_val_count < target_val_count:
                val_indices.extend(indices)
                current_val_count += len(indices)
            else:
                train_indices.extend(indices)
        if len(train_indices) == 0 and len(group_items) > 0:
            _, last_indices = group_items[-1]
            val_indices = [idx for idx in val_indices if idx not in last_indices]
            train_indices.extend(last_indices)
        train_parts.append(df.loc[train_indices])
        val_parts.append(df.loc[val_indices])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def compute_dhash_int(image_path, hash_size=8):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert('L')
    image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_value = 0
    for bit in diff.flatten():
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def hamming_distance_int(x, y):
    return (int(x) ^ int(y)).bit_count()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_near_duplicate_families_from_md5_groups(df, path_col='img_path', md5_group_col='group_id', threshold=6):
    rep_df = (df.sort_values([md5_group_col, path_col])
              .drop_duplicates(subset=md5_group_col, keep='first')
              .reset_index(drop=True).copy())
    dhash_list, failed_paths = [], []
    for image_path in tqdm(rep_df[path_col].tolist(), desc='计算 dHash'):
        try:
            dhash_list.append(compute_dhash_int(image_path))
        except Exception:
            dhash_list.append(None)
            failed_paths.append(str(image_path))
    rep_df['dhash'] = pd.Series(dhash_list, dtype='object')
    valid_mask = rep_df['dhash'].notna()
    valid_rep_df = rep_df[valid_mask].reset_index(drop=True).copy()
    valid_rep_df['dhash'] = valid_rep_df['dhash'].map(int)
    n = len(valid_rep_df)
    uf = UnionFind(n)
    for i in tqdm(range(n), desc='构建 near-duplicate family'):
        hi = valid_rep_df.loc[i, 'dhash']
        for j in range(i + 1, n):
            if hamming_distance_int(hi, valid_rep_df.loc[j, 'dhash']) <= threshold:
                uf.union(i, j)
    root_to_family = {}
    family_ids = []
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_family:
            root_to_family[root] = f'nearfam_{len(root_to_family):05d}'
        family_ids.append(root_to_family[root])
    valid_rep_df['near_dup_family_id'] = family_ids
    invalid_rep_df = rep_df[~valid_mask].copy()
    invalid_rep_df['near_dup_family_id'] = [f'nearfam_failed_{i:05d}' for i in range(len(invalid_rep_df))]
    rep_family_df = pd.concat([
        valid_rep_df[[md5_group_col, 'dhash', 'near_dup_family_id']],
        invalid_rep_df[[md5_group_col, 'dhash', 'near_dup_family_id']],
    ], axis=0, ignore_index=True)
    out_df = df.merge(rep_family_df[[md5_group_col, 'near_dup_family_id']], on=md5_group_col, how='left')
    return out_df, rep_family_df, failed_paths


def drop_exact_duplicates_in_train(df, group_col='group_id'):
    candidate_path_cols = ['img_path', 'image_path', 'filepath', 'file_path', 'path']
    sort_cols = [col for col in candidate_path_cols if col in df.columns]
    if sort_cols:
        dedup_df = (df.sort_values([group_col] + sort_cols)
                    .drop_duplicates(subset=group_col, keep='first').reset_index(drop=True))
    else:
        dedup_df = df.sort_values(group_col).drop_duplicates(subset=group_col, keep='first').reset_index(drop=True)
    removed = len(df) - len(dedup_df)
    return dedup_df, removed


# ==========================================
# 4. 加载数据
# ==========================================
print("\n" + "=" * 60)
print("数据加载：真实训练集 + 伪标签数据")
print("=" * 60)

real_train_df = load_train_dataframe(DATA_ROOT)
print(f"真实训练样本: {len(real_train_df)}")
print(f"真实类别分布:\n{real_train_df['label'].value_counts().sort_index()}")

pseudo_df = load_pseudo_labeled_data(DATA_ROOT, PROB_FILE, PSEUDO_TOP_N, PSEUDO_BOTTOM_N)
print(f"\n伪标签样本: {len(pseudo_df)} (Top-{PSEUDO_TOP_N} 陨石 + Bottom-{PSEUDO_BOTTOM_N} 非陨石)")
print(f"伪标签类别: {pseudo_df['label'].value_counts().to_dict()}")

full_train_df = pd.concat([real_train_df, pseudo_df], ignore_index=True)
print(f"\n合并后训练样本: {len(full_train_df)} ({len(real_train_df)} real + {len(pseudo_df)} pseudo)")
print(f"合并后类别分布:\n{full_train_df['label'].value_counts().sort_index()}")

# ==========================================
# 5. 去重与切分
# ==========================================
print("\n" + "=" * 60)
print("数据预处理：去重 + 划分")
print("=" * 60)

train_pool_df = attach_md5_group_ids(full_train_df)
print(f"唯一 MD5 group 数: {train_pool_df['group_id'].nunique()}")

pseudo_groups = set(train_pool_df[train_pool_df['source'] == 'pseudo']['group_id'])
real_groups = set(train_pool_df[train_pool_df['source'] == 'real']['group_id'])
overlap = pseudo_groups & real_groups
if overlap:
    print(f"! 发现 {len(overlap)} 个伪标签图像与训练集 MD5 重复，将移除")
    train_pool_df = train_pool_df[
        ~((train_pool_df['source'] == 'pseudo') & (train_pool_df['group_id'].isin(overlap)))
    ].copy()
else:
    print("✓ 伪标签图像与训练集无 MD5 重复")

train_pool_df, near_dup_rep_df, near_dup_failed = build_near_duplicate_families_from_md5_groups(
    train_pool_df, md5_group_col='group_id', threshold=6
)
print(f"near-duplicate family 数: {train_pool_df['near_dup_family_id'].nunique()}")

real_pool_df = train_pool_df[train_pool_df['source'] == 'real'].copy()
pseudo_pool_df = train_pool_df[train_pool_df['source'] == 'pseudo'].copy()

real_train_split, val_df = group_aware_train_val_split(
    real_pool_df, label_col='label', group_col='near_dup_family_id',
    val_ratio=VAL_RATIO, seed=SEED
)

train_df = pd.concat([real_train_split, pseudo_pool_df], ignore_index=True)
train_df, removed = drop_exact_duplicates_in_train(train_df)
val_df, val_removed = drop_exact_duplicates_in_train(val_df)

print(f"\n去重后: train={len(train_df)} (real+pseudo), val={len(val_df)} (real only), 移除重复={removed}")
print(f"Train 类别: {train_df['label'].value_counts().sort_index().to_dict()}")
print(f"Train 来源: {train_df['source'].value_counts().to_dict()}")
print(f"Val 类别: {val_df['label'].value_counts().sort_index().to_dict()}")

# ==========================================
# 6. 加载 DINOv2 ViT-L14
# ==========================================
print("\n" + "=" * 60)
print("加载 DINOv2 ViT-L14 (with registers)")
print("=" * 60)

dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_model.eval()
dinov2_model.to(DEVICE)
print("DINOv2 ViT-L14 加载完毕")

# ==========================================
# 7. 特征提取 — 基础 transform
# ==========================================
print("\n" + "=" * 60)
print("特征提取")
print("=" * 60)

base_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class ImagePreprocessor:
    """纹理增强预处理器：在 ToTensor 之前应用图像处理"""
    def __init__(self, enhancer):
        self.enhancer = enhancer

    def __call__(self, image):
        return self.enhancer(image)


class TransformWithPreprocess:
    """先预处理(PIL→PIL)，再应用 torchvision transform"""
    def __init__(self, preprocessor, base_tf):
        self.preprocessor = preprocessor
        self.base_tf = base_tf

    def __call__(self, image):
        image = self.preprocessor(image)
        return self.base_tf(image)


@torch.no_grad()
def extract_embeddings(loader, model, device, is_test=False):
    model.eval()
    all_embeddings, all_labels, all_paths = [], [], []
    for batch in tqdm(loader, desc='Extracting Features'):
        if is_test:
            images, paths = batch
            all_paths.extend(paths)
        else:
            images, labels = batch
            all_labels.extend(labels.cpu().tolist())
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    if is_test:
        return all_embeddings, all_paths
    return all_embeddings, np.array(all_labels)


# 训练集特征（原始图像）
train_dataset = MeteoriteDataset(train_df, transforms=base_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("提取训练集特征 (原始)...")
train_X, train_y = extract_embeddings(train_loader, dinov2_model, DEVICE, is_test=False)
print(f"训练集特征: {train_X.shape}, 标签分布: {np.bincount(train_y.astype(int))}")

is_pseudo = (train_df['source'] == 'pseudo').values.astype(float)
sample_weights = np.where(is_pseudo, PSEUDO_WEIGHT, 1.0)
print(f"伪标签样本数: {int(is_pseudo.sum())}, 权重: {PSEUDO_WEIGHT}x")

# ==========================================
# 8. 纹理增强 TTA 测试集特征提取
# ==========================================
print("\n" + "=" * 60)
print("纹理增强 TTA 测试集特征提取")
print("=" * 60)

test_df = load_test_dataframe(DATA_ROOT)
print(f"测试集样本数: {len(test_df)}")

# TTA 视图：原始 + 翻转 + CLAHE纹理增强 + Sobel边缘图
tta_configs = [
    ("原始", None),
    ("水平翻转", transforms.RandomHorizontalFlip(p=1.0)),
    ("垂直翻转", transforms.RandomVerticalFlip(p=1.0)),
    ("CLAHE纹理增强", TransformWithPreprocess(clahe_enhance, base_transform)),
    ("Sobel边缘图", TransformWithPreprocess(sobel_texture_map, base_transform)),
]

test_X_tta_list = []
for name, t in tta_configs:
    print(f"  [{name}] ...")
    if t is None:
        tf = base_transform
    elif isinstance(t, TransformWithPreprocess):
        tf = t
    else:
        tf = transforms.Compose([t, base_transform])

    test_dataset = MeteoriteDataset(test_df, transforms=tf)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    curr_X, test_ids = extract_embeddings(test_loader, dinov2_model, DEVICE, is_test=True)
    test_X_tta_list.append(curr_X)

test_X = np.mean(test_X_tta_list, axis=0)
test_X = normalize(test_X, norm='l2', axis=1)
print(f"测试集特征: {test_X.shape} (融合 {len(tta_configs)} 视角)")

# ==========================================
# 9. 5折交叉验证 SVM + kNN (加权)
# ==========================================
print("\n" + "=" * 60)
print(f"5折交叉验证 SVM + kNN (伪标签权重: {PSEUDO_WEIGHT}x)")
print("=" * 60)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

svm_test_preds = np.zeros(len(test_X))
knn_test_preds = np.zeros(len(test_X))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
    print(f"Fold {fold + 1}/{n_splits} ...")
    X_tr, y_tr = train_X[train_idx], train_y[train_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]
    w_tr = sample_weights[train_idx]

    # SVM (加权)
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED + fold)
    )
    svm.fit(X_tr, y_tr, svc__sample_weight=w_tr)
    val_acc = svm.score(X_val, y_val)
    print(f"  SVM val acc: {val_acc:.4f}")
    svm_test_preds += svm.predict_proba(test_X)[:, 1] / n_splits

    # kNN (样本复制加权)
    pseudo_mask_tr = w_tr > 1.0
    n_pseudo = int(pseudo_mask_tr.sum())
    if n_pseudo > 0:
        dup_times = int(PSEUDO_WEIGHT)
        X_tr_w = np.concatenate([X_tr] + [X_tr[pseudo_mask_tr]] * dup_times, axis=0)
        y_tr_w = np.concatenate([y_tr] + [y_tr[pseudo_mask_tr]] * dup_times, axis=0)
    else:
        X_tr_w, y_tr_w = X_tr, y_tr

    knn = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine')
    knn.fit(X_tr_w, y_tr_w)
    val_acc = knn.score(X_val, y_val)
    print(f"  kNN val acc: {val_acc:.4f}")
    knn_test_preds += knn.predict_proba(test_X)[:, 1] / n_splits

# ==========================================
# 10. 融合 + 生成提交文件
# ==========================================
print("\n" + "=" * 60)
print("融合 + 生成提交文件")
print("=" * 60)

final_scores = (svm_test_preds + knn_test_preds) / 2.0
clean_test_ids = [Path(p).name for p in test_ids]

sub_df = pd.DataFrame({'id': clean_test_ids, 'prob': final_scores})
sub_df = sub_df.sort_values(by='prob', ascending=False).reset_index(drop=True)

probs_path = OUTPUT_DIR / 'test_pred_probs_dinov2_texture.csv'
sub_df.to_csv(probs_path, index=False, encoding='utf-8-sig')
print(f"概率分布: min={sub_df['prob'].min():.4f}, max={sub_df['prob'].max():.4f}, mean={sub_df['prob'].mean():.4f}")

template_df = pd.read_csv(DATA_ROOT / 'sample_submission.csv')

for top_k in [90]:
    sub_df_k = sub_df.copy()
    sub_df_k['label'] = 0
    sub_df_k.loc[:top_k - 1, 'label'] = 1

    submit_df = template_df.copy()
    submit_df['label'] = submit_df['id'].map(sub_df_k.set_index('id')['label'])
    submit_df['label'] = submit_df['label'].astype(int)

    output_csv = OUTPUT_DIR / f'submission_dinov2_texture_top{top_k}.csv'
    submit_df.to_csv(output_csv, index=False)
    print(f"  Top-{top_k}: 预测正类 {int(sub_df_k['label'].sum())} 个 → {output_csv}")

summary = {
    'method': 'Texture-Enhanced DINOv2 Features',
    'model': 'DINOv2 ViT-L14 (registers)',
    'classifier': 'SVM(RBF) + kNN 5-fold CV ensemble',
    'tta_views': len(tta_configs),
    'tta_configs': [name for name, _ in tta_configs],
    'pseudo_top_n': PSEUDO_TOP_N,
    'pseudo_bottom_n': PSEUDO_BOTTOM_N,
    'pseudo_weight': PSEUDO_WEIGHT,
    'pseudo_count': len(pseudo_df),
    'real_train_count': len(real_train_df),
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df),
    'train_class_dist': train_df['label'].value_counts().sort_index().to_dict(),
}
with open(OUTPUT_DIR / 'run_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("纹理增强训练完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 60)
