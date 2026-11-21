"""
三模型融合 + 纹理特征混合分类器
DINOv3-L (1024d) + SwinV2-L (1536d) + ConvNeXt-L (1536d) + 纹理特征(18d)
→ 拼接 → SelectKBest → SVM/kNN

保留 train_dinov2_hybrid.py 的所有逻辑，仅替换模型部分
"""
from pathlib import Path
import sys, json, hashlib, random, os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import timm
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 0. 配置
# ==========================================
DATA_ROOT = Path("/data/数据科学实践project")
PROB_FILE = DATA_ROOT / "dinov2_simple_output" / "test_probs_dinov2.csv"
PSEUDO_TOP_N = 30
PSEUDO_BOTTOM_N = 30
PSEUDO_WEIGHT = 2.0

OUTPUT_DIR = DATA_ROOT / "runs" / f"triple_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
# 1. 三模型加载
# ==========================================
MODELS_CONFIG = {
    'dinov3': {
        'name': 'vit_large_patch16_dinov3',
        'path': '/data/models/dinov3_large/model.safetensors',
        'img_size': 224,
        'resize': 256,
        'crop': 224,
    },
    'swinv2': {
        'name': 'swinv2_large_window12to24_192to384',
        'path': '/data/models/swinv2_large/model.safetensors',
        'img_size': 384,
        'resize': 384,
        'crop': 384,
    },
    'convnext': {
        'name': 'convnext_large',
        'path': '/data/models/convnext_large/model.safetensors',
        'img_size': 224,
        'resize': 256,
        'crop': 224,
    },
}

def load_one_model(cfg):
    """Load a single model from local safetensors"""
    print(f"  Loading {cfg['name']}...", end='', flush=True)
    model = timm.create_model(cfg['name'], pretrained=False, num_classes=0)
    state = load_file(cfg['path'])
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f" {n_params:.0f}M params")
    return model

def build_transform(cfg):
    """Build transform for a model config"""
    return transforms.Compose([
        transforms.Resize(cfg['resize'], interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg['crop']),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

print("\n" + "=" * 60)
print("加载三模型")
print("=" * 60)

models = {}
transforms_dict = {}
for key, cfg in MODELS_CONFIG.items():
    models[key] = load_one_model(cfg)
    transforms_dict[key] = build_transform(cfg)

# ==========================================
# 2. 显式纹理特征提取 (原封不动)
# ==========================================
def compute_texture_features(image):
    import cv2
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    features = []

    for ws in [7, 15, 31]:
        mean = cv2.blur(gray, (ws, ws))
        mean_sq = cv2.blur(gray ** 2, (ws, ws))
        var = np.maximum(mean_sq - mean ** 2, 0)
        std = np.sqrt(var)
        features.extend([
            float(np.mean(std)),
            float(np.std(std)),
            float(np.percentile(std, 90)),
            float(np.percentile(std, 50)),
            float(np.max(std)),
        ])

    features.append(float(np.std(gray)))
    features.append(float(np.max(gray) - np.min(gray)))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    features.append(float(np.mean(grad_mag)))
    features.append(float(np.std(grad_mag)))

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask_high = ((y - center_h) ** 2 + (x - center_w) ** 2) > radius ** 2
    total_energy = (magnitude ** 2).sum()
    high_freq_energy = (magnitude[mask_high] ** 2).sum()
    features.append(float(high_freq_energy / (total_energy + 1e-8)))

    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))
    features.append(entropy)

    return np.array(features, dtype=np.float64)


def extract_texture_features_batch(image_paths):
    features_list = []
    for path in tqdm(image_paths, desc='纹理特征'):
        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            feat = compute_texture_features(image)
            features_list.append(feat)
        except Exception:
            features_list.append(np.zeros(18, dtype=np.float64))
    features_matrix = np.stack(features_list, axis=0)
    features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return features_matrix


# ==========================================
# 3. 数据集类 (支持多模型多transform)
# ==========================================
class MeteoriteDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform_dict=None, default_transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform_dict = transform_dict or {}
        self.default_transform = default_transform
        self.has_label = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["img_path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Return PIL image for multi-transform processing
        if self.default_transform is not None:
            image = self.default_transform(image)
        if self.has_label:
            return image, int(row["label"])
        return image, str(row["img_path"])


# ==========================================
# 4. 多模型特征提取
# ==========================================
@torch.no_grad()
def extract_model_features(model, df, transform_fn, desc=""):
    """Extract features for a single model using its transform"""
    features_list = []
    labels_list = []
    paths_list = []
    has_label = "label" in df.columns

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        img_path = row["img_path"]
        try:
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = transform_fn(image).unsqueeze(0).to(DEVICE)
            feat = model(tensor).cpu().numpy().flatten()
            features_list.append(feat)
            paths_list.append(img_path)
            if has_label:
                labels_list.append(int(row["label"]))
        except Exception:
            features_list.append(np.zeros(1024))
            paths_list.append(img_path)
            if has_label:
                labels_list.append(0)

    X = np.array(features_list)
    if has_label:
        return X, np.array(labels_list)
    return X


@torch.no_grad()
def extract_model_features_tta(model, df, transform_fn, desc=""):
    """Extract features with 3-view TTA (orig + hflip + vflip)"""
    from torchvision.transforms.functional import hflip, vflip

    features_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            image = Image.open(row["img_path"])
            image = ImageOps.exif_transpose(image).convert("RGB")

            t_orig = transform_fn(image).unsqueeze(0).to(DEVICE)
            t_h = transform_fn(hflip(image)).unsqueeze(0).to(DEVICE)
            t_v = transform_fn(vflip(image)).unsqueeze(0).to(DEVICE)

            f_orig = model(t_orig).cpu().numpy().flatten()
            f_h = model(t_h).cpu().numpy().flatten()
            f_v = model(t_v).cpu().numpy().flatten()

            features_list.append((f_orig + f_h + f_v) / 3.0)
        except Exception:
            features_list.append(np.zeros(1024))

    return np.array(features_list)


# ==========================================
# 5. 数据加载与去重 (原封不动)
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
    top_df = prob_df.head(top_n)[['id']].copy(); top_df['label'] = 1
    bot_df = prob_df.tail(bottom_n)[['id']].copy(); bot_df['label'] = 0
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
            if not chunk: break
            md5.update(chunk)
    return md5.hexdigest()


def attach_md5_group_ids(df, path_col="img_path", group_col="group_id"):
    df = df.copy(); df[group_col] = df[path_col].apply(compute_file_md5); return df


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
            val_indices = [i for i in val_indices if i not in last_indices]
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
        self.parent = list(range(n)); self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]: self.parent[rb] = ra
        else: self.parent[rb] = ra; self.rank[ra] += 1


def build_near_duplicate_families_from_md5_groups(df, path_col='img_path', md5_group_col='group_id', threshold=6):
    rep_df = (df.sort_values([md5_group_col, path_col])
              .drop_duplicates(subset=md5_group_col, keep='first').reset_index(drop=True).copy())
    dhash_list, failed_paths = [], []
    for image_path in tqdm(rep_df[path_col].tolist(), desc='dHash'):
        try: dhash_list.append(compute_dhash_int(image_path))
        except Exception:
            dhash_list.append(None); failed_paths.append(str(image_path))
    rep_df['dhash'] = pd.Series(dhash_list, dtype='object')
    valid_mask = rep_df['dhash'].notna()
    valid_rep_df = rep_df[valid_mask].reset_index(drop=True).copy()
    valid_rep_df['dhash'] = valid_rep_df['dhash'].map(int)
    n = len(valid_rep_df); uf = UnionFind(n)
    for i in tqdm(range(n), desc='near-duplicate'):
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
# 6. 数据加载
# ==========================================
print("\n" + "=" * 60)
print("数据加载")
print("=" * 60)

real_train_df = load_train_dataframe(DATA_ROOT)
print(f"真实训练样本: {len(real_train_df)}")

pseudo_df = load_pseudo_labeled_data(DATA_ROOT, PROB_FILE, PSEUDO_TOP_N, PSEUDO_BOTTOM_N)
print(f"伪标签: {len(pseudo_df)} (Top-{PSEUDO_TOP_N}陨石 + Bottom-{PSEUDO_BOTTOM_N}非陨石)")

full_train_df = pd.concat([real_train_df, pseudo_df], ignore_index=True)

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
train_pool_df, _, _ = build_near_duplicate_families_from_md5_groups(
    train_pool_df, md5_group_col='group_id', threshold=6)
real_pool_df = train_pool_df[train_pool_df['source'] == 'real'].copy()
pseudo_pool_df = train_pool_df[train_pool_df['source'] == 'pseudo'].copy()
real_train_split, val_df = group_aware_train_val_split(
    real_pool_df, label_col='label', group_col='near_dup_family_id', val_ratio=VAL_RATIO, seed=SEED)
train_df = pd.concat([real_train_split, pseudo_pool_df], ignore_index=True)
train_df, _ = drop_exact_duplicates_in_train(train_df)
val_df, _ = drop_exact_duplicates_in_train(val_df)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# ==========================================
# 7. 三模型特征提取
# ==========================================
print("\n" + "=" * 60)
print("三模型特征提取")
print("=" * 60)

all_train_features = {}
all_test_features = {}

# Load test dataframe for feature extraction
test_df = load_test_dataframe(DATA_ROOT)
print(f"测试样本: {len(test_df)}")

for key in ['dinov3', 'swinv2', 'convnext']:
    model = models[key]
    tfm = transforms_dict[key]
    cfg = MODELS_CONFIG[key]

    print(f"\n--- {key.upper()} ({cfg['name']}) ---")

    # Train features (extract_model_features returns tuple (X, y) for labeled data)
    result = extract_model_features(model, train_df, tfm, desc=f"{key} train")
    X_train = result[0] if isinstance(result, tuple) else result
    all_train_features[key] = X_train
    print(f"  训练特征: {X_train.shape}")

    # Test features with TTA
    X_test = extract_model_features_tta(model, test_df, tfm, desc=f"{key} test TTA")
    all_test_features[key] = X_test
    print(f"  测试特征: {X_test.shape}")

# Train labels (same for all models)
train_y = train_df['label'].values

# ==========================================
# 8. 纹理特征提取
# ==========================================
print("\n" + "=" * 60)
print("提取显式纹理特征")
print("=" * 60)

train_texture_feat = extract_texture_features_batch(train_df['img_path'].tolist())
print(f"训练集纹理特征: {train_texture_feat.shape}")

test_texture_feat = extract_texture_features_batch(test_df['img_path'].tolist())
print(f"测试集纹理特征: {test_texture_feat.shape}")

# ==========================================
# 9. 特征融合
# ==========================================
print("\n" + "=" * 60)
print("特征融合: DINOv3(1024d) + SwinV2(1536d) + ConvNeXt(1536d) + 纹理(18d)")
print("=" * 60)

# L2 normalize each model's features
train_features_norm = []
test_features_norm = []
total_dim = 0

for key in ['dinov3', 'swinv2', 'convnext']:
    tr_norm = normalize(all_train_features[key], norm='l2', axis=1)
    te_norm = normalize(all_test_features[key], norm='l2', axis=1)
    train_features_norm.append(tr_norm)
    test_features_norm.append(te_norm)
    print(f"  {key}: {tr_norm.shape[1]}d")
    total_dim += tr_norm.shape[1]

# Standardize texture
texture_scaler = RobustScaler().fit(train_texture_feat)
train_texture_scaled = texture_scaler.transform(train_texture_feat)
test_texture_scaled = texture_scaler.transform(test_texture_feat)
train_features_norm.append(train_texture_scaled)
test_features_norm.append(test_texture_scaled)
print(f"  texture: {train_texture_feat.shape[1]}d")
total_dim += train_texture_feat.shape[1]

# Concatenate all
train_X = np.concatenate(train_features_norm, axis=1)
test_X = np.concatenate(test_features_norm, axis=1)
print(f"  融合前总维度: {total_dim}")

# Feature selection
selector = SelectKBest(f_classif, k=min(1024, train_X.shape[1]))
train_X = selector.fit_transform(train_X, train_y)
test_X = selector.transform(test_X)
print(f"  SelectKBest后: {train_X.shape[1]}d")

# Final normalize
train_X = normalize(train_X, norm='l2', axis=1)
test_X = normalize(test_X, norm='l2', axis=1)

# Sample weights
is_pseudo = (train_df['source'] == 'pseudo').values.astype(float)
sample_weights = np.where(is_pseudo, PSEUDO_WEIGHT, 1.0)
print(f"  伪标签样本: {int(is_pseudo.sum())}, 权重: {PSEUDO_WEIGHT}x")
print(f"  最终: train_X={train_X.shape}, test_X={test_X.shape}")

# ==========================================
# 10. 5折CV SVM + kNN (加权，原封不动)
# ==========================================
print("\n" + "=" * 60)
print("5折交叉验证 SVM + kNN (三模型+纹理)")
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
# 11. 生成提交 (原封不动)
# ==========================================
print("\n" + "=" * 60)
print("生成提交文件")
print("=" * 60)

final_scores = (svm_test_preds + knn_test_preds) / 2.0
clean_test_ids = [Path(p).name for p in test_df['img_path'].tolist()]

sub_df = pd.DataFrame({'id': clean_test_ids, 'prob': final_scores})
sub_df = sub_df.sort_values(by='prob', ascending=False).reset_index(drop=True)

probs_path = OUTPUT_DIR / 'test_probs_triple_hybrid.csv'
sub_df.to_csv(probs_path, index=False, encoding='utf-8-sig')
print(f"概率: min={sub_df['prob'].min():.4f} max={sub_df['prob'].max():.4f} mean={sub_df['prob'].mean():.4f}")

sub_df['label'] = 0
sub_df.loc[:89, 'label'] = 1

template_df = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
template_df['label'] = template_df['id'].map(sub_df.set_index('id')['label']).fillna(0).astype(int)

output_csv = OUTPUT_DIR / 'submission_triple_hybrid_top90.csv'
template_df.to_csv(output_csv, index=False)
print(f"Top-90: 陨石={(template_df['label']==1).sum()} → {output_csv}")

with open(OUTPUT_DIR / 'run_summary.json', 'w', encoding='utf-8') as f:
    json.dump({
        'method': 'DINOv3-L + SwinV2-L + ConvNeXt-L + Texture Hybrid',
        'models': {
            'dinov3': {'dim': int(all_train_features['dinov3'].shape[1]), 'params': '303M'},
            'swinv2': {'dim': int(all_train_features['swinv2'].shape[1]), 'params': '195M'},
            'convnext': {'dim': int(all_train_features['convnext'].shape[1]), 'params': '196M'},
        },
        'texture_dim': int(train_texture_feat.shape[1]),
        'fused_dim': int(train_X.shape[1]),
        'classifier': 'SVM(RBF) + kNN 5-fold, SelectKBest',
        'pseudo_top_n': PSEUDO_TOP_N, 'pseudo_bottom_n': PSEUDO_BOTTOM_N,
        'pseudo_weight': PSEUDO_WEIGHT,
    }, f, indent=2, ensure_ascii=False)

print("\n完成！")
