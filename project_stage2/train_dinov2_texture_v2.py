"""
DINOv2 ViT-L14 + 表面异质性检测
针对"长尾夹被误判为陨石"问题：
- 局部方差图：度量表面纹理密度 → 陨石(高方差/多裂纹气印) vs 长尾夹(低方差/光滑均匀)
- 局部熵图：度量表面混乱度 → 陨石(混乱) vs 人造物(规整)
- 用纹理特征图作为额外TTA视角，让分类器学会区分"真纹理"和"伪纹理"
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
from PIL import Image, ImageOps, ImageFile
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
PROB_FILE = DATA_ROOT / "dinov2_simple_output" / "test_probs_dinov2.csv"
PSEUDO_TOP_N = 30
PSEUDO_BOTTOM_N = 30
PSEUDO_WEIGHT = 2.0

OUTPUT_DIR = DATA_ROOT / "runs" / f"dinov2_het_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4
VAL_RATIO = 0.2

print(f"DEVICE: {DEVICE}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==========================================
# 1. 表面异质性特征图 — 区分陨石纹理 vs 光滑人造物
# ==========================================
def local_std_map(image, window_size=11):
    """
    局部标准差图 — 度量表面纹理密度。
    陨石：微裂纹、气印产生高局部方差 → 亮区多
    长尾夹/光滑人造物：均匀表面 → 暗区为主
    """
    import cv2
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.float64)
    mean = cv2.blur(gray, (window_size, window_size))
    mean_sq = cv2.blur(gray ** 2, (window_size, window_size))
    var = np.maximum(mean_sq - mean ** 2, 0)
    std = np.sqrt(var)
    std_norm = (std / (std.max() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([std_norm, std_norm, std_norm], axis=-1))


def local_entropy_map(image, window_size=11):
    """
    局部熵图 — 度量表面混乱度。
    陨石：不规则纹理 → 高熵
    人造物(长尾夹)：规则/均匀 → 低熵
    """
    import cv2
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # 用局部直方图近似熵
    pad = window_size // 2
    gray_padded = np.pad(gray, pad, mode='reflect')
    entropy = np.zeros_like(gray, dtype=np.float64)
    n_bins = 32
    for i in range(pad, gray_padded.shape[0] - pad):
        for j in range(pad, gray_padded.shape[1] - pad):
            patch = gray_padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            hist, _ = np.histogram(patch, bins=n_bins, range=(0, 256), density=True)
            hist = hist[hist > 0]
            entropy[i - pad, j - pad] = -np.sum(hist * np.log2(hist))
    entropy_norm = (entropy / (entropy.max() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([entropy_norm, entropy_norm, entropy_norm], axis=-1))


def texture_energy_map(image):
    """
    综合纹理能量图：局部标准差 × 局部熵的几何平均。
    两者同时高 → 真陨石纹理；一高一低或双低 → 非陨石。
    """
    import cv2
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.float64)

    # 快速局部标准差
    window_size = 11
    mean = cv2.blur(gray, (window_size, window_size))
    mean_sq = cv2.blur(gray ** 2, (window_size, window_size))
    var = np.maximum(mean_sq - mean ** 2, 0)
    std = np.sqrt(var)
    std_norm = std / (std.max() + 1e-8)

    # 归一化后几何平均 → 信号更干净
    energy = np.sqrt(std_norm)  # sqrt压缩动态范围，突出中等纹理
    energy_norm = (energy / (energy.max() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([energy_norm, energy_norm, energy_norm], axis=-1))


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
# 3. 数据加载与去重 (与之前相同)
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
        val_indices, train_indices = [], []
        for _, indices in group_items:
            if len(val_indices) < target_val_count:
                val_indices.extend(indices)
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
              .drop_duplicates(subset=md5_group_col, keep='first').reset_index(drop=True).copy())
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
print("数据加载")
print("=" * 60)

real_train_df = load_train_dataframe(DATA_ROOT)
print(f"真实训练样本: {len(real_train_df)}")

pseudo_df = load_pseudo_labeled_data(DATA_ROOT, PROB_FILE, PSEUDO_TOP_N, PSEUDO_BOTTOM_N)
print(f"伪标签: {len(pseudo_df)} (Top-{PSEUDO_TOP_N}陨石 + Bottom-{PSEUDO_BOTTOM_N}非陨石)")

full_train_df = pd.concat([real_train_df, pseudo_df], ignore_index=True)

# ==========================================
# 5. 去重与切分
# ==========================================
print("\n" + "=" * 60)
print("去重 + 划分")
print("=" * 60)

train_pool_df = attach_md5_group_ids(full_train_df)
pseudo_groups = set(train_pool_df[train_pool_df['source'] == 'pseudo']['group_id'])
real_groups = set(train_pool_df[train_pool_df['source'] == 'real']['group_id'])
overlap = pseudo_groups & real_groups
if overlap:
    train_pool_df = train_pool_df[
        ~((train_pool_df['source'] == 'pseudo') & (train_pool_df['group_id'].isin(overlap)))
    ].copy()
    print(f"移除 {len(overlap)} 个重复伪标签")

train_pool_df, _, _ = build_near_duplicate_families_from_md5_groups(
    train_pool_df, md5_group_col='group_id', threshold=6
)

real_pool_df = train_pool_df[train_pool_df['source'] == 'real'].copy()
pseudo_pool_df = train_pool_df[train_pool_df['source'] == 'pseudo'].copy()

real_train_split, val_df = group_aware_train_val_split(
    real_pool_df, label_col='label', group_col='near_dup_family_id',
    val_ratio=VAL_RATIO, seed=SEED
)

train_df = pd.concat([real_train_split, pseudo_pool_df], ignore_index=True)
train_df, removed = drop_exact_duplicates_in_train(train_df)
val_df, _ = drop_exact_duplicates_in_train(val_df)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, 去重移除: {removed}")
print(f"Train 类别: {train_df['label'].value_counts().sort_index().to_dict()}")

# ==========================================
# 6. 加载 DINOv2
# ==========================================
print("\n" + "=" * 60)
print("加载 DINOv2 ViT-L14")
print("=" * 60)

dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_model.eval()
dinov2_model.to(DEVICE)
print("DINOv2 加载完毕")

# ==========================================
# 7. Transform 和 TTA 配置
# ==========================================
base_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class TransformWithPreprocess:
    """先PIL预处理，再torchvision transform"""
    def __init__(self, preprocessor, base_tf):
        self.preprocessor = preprocessor
        self.base_tf = base_tf

    def __call__(self, image):
        image = self.preprocessor(image)
        return self.base_tf(image)


# TTA 视图配置：
# 原始+翻转(通用) + 局部方差(表面异质性) + 纹理能量(综合纹理)
tta_configs = [
    ("原始", None),
    ("水平翻转", transforms.RandomHorizontalFlip(p=1.0)),
    ("垂直翻转", transforms.RandomVerticalFlip(p=1.0)),
    ("局部方差图", TransformWithPreprocess(local_std_map, base_transform)),
    ("纹理能量图", TransformWithPreprocess(texture_energy_map, base_transform)),
]


@torch.no_grad()
def extract_embeddings(loader, model, device, is_test=False):
    model.eval()
    all_embeddings, all_labels, all_paths = [], [], []
    for batch in tqdm(loader, desc='Extracting'):
        if is_test:
            images, paths = batch
            all_paths.extend(paths)
        else:
            images, labels = batch
            all_labels.extend(labels.cpu().tolist())
        images = images.to(device)
        all_embeddings.append(model(images).cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    if is_test:
        return all_embeddings, all_paths
    return all_embeddings, np.array(all_labels)


# 训练集特征（原始图像）
print("\n" + "=" * 60)
print("提取训练集特征")
print("=" * 60)

train_dataset = MeteoriteDataset(train_df, transforms=base_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
train_X, train_y = extract_embeddings(train_loader, dinov2_model, DEVICE, is_test=False)
print(f"训练集特征: {train_X.shape}, 标签: {np.bincount(train_y.astype(int))}")

is_pseudo = (train_df['source'] == 'pseudo').values.astype(float)
sample_weights = np.where(is_pseudo, PSEUDO_WEIGHT, 1.0)
print(f"伪标签样本: {int(is_pseudo.sum())}, 权重: {PSEUDO_WEIGHT}x")

# ==========================================
# 8. 纹理增强 TTA 测试集特征提取
# ==========================================
print("\n" + "=" * 60)
print(f"纹理TTA测试集特征提取 ({len(tta_configs)} 视角)")
print("=" * 60)

test_df = load_test_dataframe(DATA_ROOT)
print(f"测试集: {len(test_df)}")

test_X_tta_list = []
for name, t in tta_configs:
    print(f"  [{name}]")
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
# 9. 5折CV SVM + kNN (加权)
# ==========================================
print("\n" + "=" * 60)
print(f"5折交叉验证 SVM + kNN (伪标签权重 {PSEUDO_WEIGHT}x)")
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

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED + fold)
    )
    svm.fit(X_tr, y_tr, svc__sample_weight=w_tr)
    print(f"  SVM val acc: {svm.score(X_val, y_val):.4f}")
    svm_test_preds += svm.predict_proba(test_X)[:, 1] / n_splits

    pseudo_mask_tr = w_tr > 1.0
    if int(pseudo_mask_tr.sum()) > 0:
        dup = int(PSEUDO_WEIGHT)
        X_tr_w = np.concatenate([X_tr] + [X_tr[pseudo_mask_tr]] * dup, axis=0)
        y_tr_w = np.concatenate([y_tr] + [y_tr[pseudo_mask_tr]] * dup, axis=0)
    else:
        X_tr_w, y_tr_w = X_tr, y_tr
    knn = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine')
    knn.fit(X_tr_w, y_tr_w)
    print(f"  kNN val acc: {knn.score(X_val, y_val):.4f}")
    knn_test_preds += knn.predict_proba(test_X)[:, 1] / n_splits

# ==========================================
# 10. 融合 + 生成提交
# ==========================================
print("\n" + "=" * 60)
print("生成提交文件")
print("=" * 60)

final_scores = (svm_test_preds + knn_test_preds) / 2.0
clean_test_ids = [Path(p).name for p in test_ids]

sub_df = pd.DataFrame({'id': clean_test_ids, 'prob': final_scores})
sub_df = sub_df.sort_values(by='prob', ascending=False).reset_index(drop=True)

probs_path = OUTPUT_DIR / 'test_probs_het.csv'
sub_df.to_csv(probs_path, index=False, encoding='utf-8-sig')
print(f"概率: min={sub_df['prob'].min():.4f} max={sub_df['prob'].max():.4f} mean={sub_df['prob'].mean():.4f}")

sub_df['label'] = 0
sub_df.loc[:89, 'label'] = 1

template_df = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
template_df['label'] = template_df['id'].map(sub_df.set_index('id')['label']).fillna(0).astype(int)

output_csv = OUTPUT_DIR / 'submission_het_top90.csv'
template_df.to_csv(output_csv, index=False)
print(f"Top-90: 陨石={(template_df['label']==1).sum()} → {output_csv}")

with open(OUTPUT_DIR / 'run_summary.json', 'w', encoding='utf-8') as f:
    json.dump({
        'method': 'Surface Heterogeneity Detection',
        'model': 'DINOv2 ViT-L14 (registers)',
        'classifier': 'SVM(RBF) + kNN 5-fold',
        'tta_views': [n for n, _ in tta_configs],
        'pseudo_top_n': PSEUDO_TOP_N, 'pseudo_bottom_n': PSEUDO_BOTTOM_N,
        'pseudo_weight': PSEUDO_WEIGHT, 'pseudo_count': len(pseudo_df),
        'train_size': len(train_df), 'test_size': len(test_df),
    }, f, indent=2, ensure_ascii=False)

print("\n完成！")
