"""
DINOv2 ViT-L14 + Texture-enhanced feature extraction
adapted for meteorite images:
- CLAHE Enhance local contrast to reveal cracks and depressions
- Sobel Enhance edges to emphasize shape and texture
- Combine original and texture-enhanced TTA views
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
# 0. Configuration
# ==========================================
DATA_ROOT = Path("/data/数据科学实践project")

# Pseudo-label settings: select top N meteorites and bottom N terrestrial rocks
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
print(f"PSEUDO: Top-{PSEUDO_TOP_N} meteorites + Bottom-{PSEUDO_BOTTOM_N} non-meteorites, Weight={PSEUDO_WEIGHT}x")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==========================================
# 1. Texture transforms for meteorite images
# ==========================================
def clahe_enhance(image):
    """
    Adaptive histogram equalization (CLAHE).
    Increase local contrast to reveal surface detail:
    - fine fusion-crust cracks (Fusion Crust cracks)
    - regmaglypts or thumbprint-like depressions (Regmaglypts)
    - flow lines (Flow lines)
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
    Sobel edge detection to produce a three-channel map.
    Emphasize surface morphology:
    - Irregular specimen contours (sculpted, irregular shape)
    - Surface depressions (regmaglypt outlines)
    """
    import cv2
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_mag = (edge_mag / (edge_mag.max() + 1e-8) * 255).astype(np.uint8)
    # 3 channels
    edge_3ch = np.stack([edge_mag, edge_mag, edge_mag], axis=-1)
    return Image.fromarray(edge_3ch)


# ==========================================
# 2. Dataset class
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
# 3. Load, deduplicate and split data (original procedure)
# ==========================================
def _resolve_image_dir(root, split):
    candidates = [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Missing {split} image directory")


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
    """Choose the highest top_n scores as meteorites (1) and the lowest bottom_n as non-meteorites(0)."""
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
    for image_path in tqdm(rep_df[path_col].tolist(), desc='Compute dHash'):
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
    for i in tqdm(range(n), desc='Build near-duplicate family'):
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
# 4. Load data
# ==========================================
print("\n" + "=" * 60)
print("Load labeled and pseudo-labeled training data")
print("=" * 60)

real_train_df = load_train_dataframe(DATA_ROOT)
print(f"Labeled training samples: {len(real_train_df)}")
print(f"Observed label distribution:\n{real_train_df['label'].value_counts().sort_index()}")

pseudo_df = load_pseudo_labeled_data(DATA_ROOT, PROB_FILE, PSEUDO_TOP_N, PSEUDO_BOTTOM_N)
print(f"\n Pseudo-labeled samples: {len(pseudo_df)} (Top-{PSEUDO_TOP_N} meteorites + Bottom-{PSEUDO_BOTTOM_N} non-meteorites)")
print(f"Pseudo-label classes: {pseudo_df['label'].value_counts().to_dict()}")

full_train_df = pd.concat([real_train_df, pseudo_df], ignore_index=True)
print(f"\n Combined training samples: {len(full_train_df)} ({len(real_train_df)} real + {len(pseudo_df)} pseudo)")
print(f"Combined class distribution:\n{full_train_df['label'].value_counts().sort_index()}")

# ==========================================
# 5. Deduplicate and split
# ==========================================
print("\n" + "=" * 60)
print("Preprocess: deduplicate and split")
print("=" * 60)

train_pool_df = attach_md5_group_ids(full_train_df)
print(f"Unique MD5 groups: {train_pool_df['group_id'].nunique()}")

pseudo_groups = set(train_pool_df[train_pool_df['source'] == 'pseudo']['group_id'])
real_groups = set(train_pool_df[train_pool_df['source'] == 'real']['group_id'])
overlap = pseudo_groups & real_groups
if overlap:
    print(f"! Found {len(overlap)} pseudo-labeled images with matching training MD5 hashes; removing")
    train_pool_df = train_pool_df[
        ~((train_pool_df['source'] == 'pseudo') & (train_pool_df['group_id'].isin(overlap)))
    ].copy()
else:
    print("✓ No MD5 overlap between pseudo-labeled and training images")

train_pool_df, near_dup_rep_df, near_dup_failed = build_near_duplicate_families_from_md5_groups(
    train_pool_df, md5_group_col='group_id', threshold=6
)
print(f"near-duplicate family count: {train_pool_df['near_dup_family_id'].nunique()}")

real_pool_df = train_pool_df[train_pool_df['source'] == 'real'].copy()
pseudo_pool_df = train_pool_df[train_pool_df['source'] == 'pseudo'].copy()

real_train_split, val_df = group_aware_train_val_split(
    real_pool_df, label_col='label', group_col='near_dup_family_id',
    val_ratio=VAL_RATIO, seed=SEED
)

train_df = pd.concat([real_train_split, pseudo_pool_df], ignore_index=True)
train_df, removed = drop_exact_duplicates_in_train(train_df)
val_df, val_removed = drop_exact_duplicates_in_train(val_df)

print(f"\n After deduplication: train={len(train_df)} (real+pseudo), val={len(val_df)} (real only), duplicates removed={removed}")
print(f"Train classes: {train_df['label'].value_counts().sort_index().to_dict()}")
print(f"Train source: {train_df['source'].value_counts().to_dict()}")
print(f"Val classes: {val_df['label'].value_counts().sort_index().to_dict()}")

# ==========================================
# 6. Load DINOv2 ViT-L14
# ==========================================
print("\n" + "=" * 60)
print("Load DINOv2 ViT-L14 (with registers)")
print("=" * 60)

dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_model.eval()
dinov2_model.to(DEVICE)
print("DINOv2 ViT-L14 loaded")

# ==========================================
# 7. feature extraction: base transform
# ==========================================
print("\n" + "=" * 60)
print("Feature extraction")
print("=" * 60)

base_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class ImagePreprocessor:
    """Apply texture processing before ToTensor"""
    def __init__(self, enhancer):
        self.enhancer = enhancer

    def __call__(self, image):
        return self.enhancer(image)


class TransformWithPreprocess:
    """Preprocess PIL to PIL, then apply torchvision transform"""
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


# Training features (original images)
train_dataset = MeteoriteDataset(train_df, transforms=base_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("Extract training features (original images)...")
train_X, train_y = extract_embeddings(train_loader, dinov2_model, DEVICE, is_test=False)
print(f"Training features: {train_X.shape}, Class distribution: {np.bincount(train_y.astype(int))}")

is_pseudo = (train_df['source'] == 'pseudo').values.astype(float)
sample_weights = np.where(is_pseudo, PSEUDO_WEIGHT, 1.0)
print(f"Pseudo-labeled sample count: {int(is_pseudo.sum())}, weight: {PSEUDO_WEIGHT}x")

# ==========================================
# 8. Extract texture-enhanced TTA test features
# ==========================================
print("\n" + "=" * 60)
print("Extract texture-enhanced TTA test features")
print("=" * 60)

test_df = load_test_dataframe(DATA_ROOT)
print(f"Test images: {len(test_df)}")

# TTA views: original, flips, CLAHE and Sobel edges
tta_configs = [
    ("original", None),
    ("horizontal flip", transforms.RandomHorizontalFlip(p=1.0)),
    ("vertical flip", transforms.RandomVerticalFlip(p=1.0)),
    ("CLAHE texture enhancement", TransformWithPreprocess(clahe_enhance, base_transform)),
    ("Sobel edge map", TransformWithPreprocess(sobel_texture_map, base_transform)),
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
print(f"Test features: {test_X.shape} (blend {len(tta_configs)} view)")

# ==========================================
# 9. 5 fold SVM + kNN cross-validation (weighted)
# ==========================================
print("\n" + "=" * 60)
print(f"5 fold SVM + kNN cross-validation (pseudo-label weight: {PSEUDO_WEIGHT}x)")
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

    # SVM (weighted)
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED + fold)
    )
    svm.fit(X_tr, y_tr, svc__sample_weight=w_tr)
    val_acc = svm.score(X_val, y_val)
    print(f"  SVM val acc: {val_acc:.4f}")
    svm_test_preds += svm.predict_proba(test_X)[:, 1] / n_splits

    # kNN (weight by sample replication)
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
# 10. Blend predictions and write submission
# ==========================================
print("\n" + "=" * 60)
print("Blend predictions and write submission")
print("=" * 60)

final_scores = (svm_test_preds + knn_test_preds) / 2.0
clean_test_ids = [Path(p).name for p in test_ids]

sub_df = pd.DataFrame({'id': clean_test_ids, 'prob': final_scores})
sub_df = sub_df.sort_values(by='prob', ascending=False).reset_index(drop=True)

probs_path = OUTPUT_DIR / 'test_pred_probs_dinov2_texture.csv'
sub_df.to_csv(probs_path, index=False, encoding='utf-8-sig')
print(f"Probability distribution: min={sub_df['prob'].min():.4f}, max={sub_df['prob'].max():.4f}, mean={sub_df['prob'].mean():.4f}")

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
    print(f"  Top-{top_k}: Predicted positives {int(sub_df_k['label'].sum())} samples → {output_csv}")

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
print("Texture-enhanced training complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)
