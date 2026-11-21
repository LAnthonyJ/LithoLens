"""
DINOv2 ViT-L/14 + Classifier with proper near-duplicate handling
Runs on GPU server. Uses natural classification threshold (not top-k).
"""
import os, sys, json, time, hashlib, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# Config
# ==========================================
DATA_ROOT = Path('/data/数据科学实践project')
OUTPUT_DIR = DATA_ROOT / 'dinov2_natural_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda')
SEED = 42
BATCH_SIZE = 32

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Output: {OUTPUT_DIR}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ==========================================
# Data loading (same logic as train_dinov2.py)
# ==========================================
def _resolve_image_dir(root, split):
    candidates = [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Cannot find {split} image dir")

def load_train_dataframe(data_root):
    data_root = Path(data_root)
    train_dir = _resolve_image_dir(data_root, "train")
    df = pd.read_csv(data_root / "train_labels.csv")
    df["img_path"] = df["id"].apply(lambda x: str(train_dir / x))
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()
    df["label"] = df["label"].astype(int)
    return df

def load_test_dataframe(data_root):
    data_root = Path(data_root)
    test_dir = _resolve_image_dir(data_root, "test")
    df = pd.read_csv(data_root / "sample_submission.csv")
    df = df[["id"]].copy()
    df["img_path"] = df["id"].apply(lambda x: str(test_dir / x))
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()
    return df

# ==========================================
# Near-duplicate detection (same as train_dinov2.py)
# ==========================================
def compute_file_md5(path, chunk_size=1024*1024):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            md5.update(chunk)
    return md5.hexdigest()

def attach_md5_group_ids(df, path_col="img_path", group_col="group_id"):
    df = df.copy()
    df[group_col] = df[path_col].apply(compute_file_md5)
    return df

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
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]: self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def build_near_duplicate_families_from_md5_groups(df, path_col='img_path', md5_group_col='group_id', threshold=6):
    rep_df = (df.sort_values([md5_group_col, path_col])
              .drop_duplicates(subset=md5_group_col, keep='first')
              .reset_index(drop=True).copy())
    dhash_list, failed_paths = [], []
    for image_path in tqdm(rep_df[path_col].tolist(), desc='Computing dHash'):
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
    for i in tqdm(range(n), desc='Building near-duplicate families'):
        hi = valid_rep_df.loc[i, 'dhash']
        for j in range(i+1, n):
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

def drop_exact_duplicates_in_train(df, group_col='group_id'):
    sort_cols = ['img_path'] if 'img_path' in df.columns else []
    if sort_cols:
        dedup_df = (df.sort_values([group_col] + sort_cols)
                    .drop_duplicates(subset=group_col, keep='first').reset_index(drop=True))
    else:
        dedup_df = df.sort_values(group_col).drop_duplicates(subset=group_col, keep='first').reset_index(drop=True)
    removed = len(df) - len(dedup_df)
    return dedup_df, removed

# ==========================================
# Dataset class
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
# Load data with dedup
# ==========================================
print("\n" + "="*60)
print("DATA LOADING & DEDUPLICATION")
print("="*60)

full_train_df = load_train_dataframe(DATA_ROOT)
print(f"Training images: {len(full_train_df)}")
print(f"Label distribution:\n{full_train_df['label'].value_counts().sort_index()}")

train_pool_df = attach_md5_group_ids(full_train_df)
print(f"Unique MD5 groups: {train_pool_df['group_id'].nunique()}")

train_pool_df, _, _ = build_near_duplicate_families_from_md5_groups(
    train_pool_df, md5_group_col='group_id', threshold=6
)
print(f"Near-duplicate families: {train_pool_df['near_dup_family_id'].nunique()}")

train_df, val_df = group_aware_train_val_split(
    train_pool_df, label_col='label', group_col='near_dup_family_id',
    val_ratio=0.2, seed=SEED
)

train_df, removed = drop_exact_duplicates_in_train(train_df)
print(f"After dedup - Train: {len(train_df)}, Val: {len(val_df)}, Removed dup: {removed}")
print(f"Train labels: {train_df['label'].value_counts().sort_index().to_dict()}")
print(f"Val labels: {val_df['label'].value_counts().sort_index().to_dict()}")

test_df = load_test_dataframe(DATA_ROOT)
print(f"Test images: {len(test_df)}")

# ==========================================
# Load DINOv2 ViT-L/14
# ==========================================
print("\n" + "="*60)
print("LOADING DINOv2 ViT-L/14")
print("="*60)

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', trust_repo=True)
dinov2.eval()
dinov2.to(DEVICE)
print("DINOv2 ViT-L/14 loaded!")

# ==========================================
# Feature extraction
# ==========================================
print("\n" + "="*60)
print("FEATURE EXTRACTION")
print("="*60)

base_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

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

# Train features
train_dataset = MeteoriteDataset(train_df, transforms=base_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
train_X, train_y = extract_embeddings(train_loader, dinov2, DEVICE, is_test=False)
print(f"Train features: {train_X.shape}, Labels: {np.bincount(train_y.astype(int))}")

# Val features
val_dataset = MeteoriteDataset(val_df, transforms=base_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
val_X, val_y = extract_embeddings(val_loader, dinov2, DEVICE, is_test=False)
print(f"Val features: {val_X.shape}, Labels: {np.bincount(val_y.astype(int))}")

# Test features with TTA
print("\nTTA Feature Extraction for Test...")
TTA_TRANSFORMS = [
    base_transform,
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), base_transform]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), base_transform])
]

test_X_tta = []
for i, t in enumerate(TTA_TRANSFORMS):
    test_dataset = MeteoriteDataset(test_df, transforms=t)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    curr_X, test_ids = extract_embeddings(test_loader, dinov2, DEVICE, is_test=True)
    test_X_tta.append(curr_X)

test_X = np.mean(test_X_tta, axis=0)
test_X = normalize(test_X, norm='l2', axis=1)
print(f"Test features: {test_X.shape}")

# Get clean test IDs
clean_test_ids = [Path(p).name for p in test_ids]

# ==========================================
# Classifier comparison
# ==========================================
print("\n" + "="*60)
print("CLASSIFIER COMPARISON (with dedup)")
print("="*60)

classifiers = {
    "SVM_RBF": make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED)
    ),
    "SVM_Linear": make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=SEED)
    ),
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
    ),
    "kNN_k25_cosine": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine')
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=(train_y==0).sum()/max((train_y==1).sum(),1),
        random_state=SEED, verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        class_weight='balanced', random_state=SEED, verbose=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=15, class_weight='balanced',
        random_state=SEED, n_jobs=-1
    ),
}

results = []
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

for name, clf in classifiers.items():
    cv_acc = cross_val_score(clf, train_X, train_y, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(clf, train_X, train_y, cv=skf, scoring='f1')

    print(f"  {name:<25s} Acc: {cv_acc.mean():.4f}+/-{cv_acc.std():.4f}  F1: {cv_f1.mean():.4f}+/-{cv_f1.std():.4f}")
    results.append({'name': name, 'acc_mean': cv_acc.mean(), 'acc_std': cv_acc.std(),
                    'f1_mean': cv_f1.mean(), 'f1_std': cv_f1.std()})

results_df = pd.DataFrame(results).sort_values('f1_mean', ascending=False)
best_model = results_df.iloc[0]['name']
print(f"\nBest model: {best_model}")

# ==========================================
# Train best model + predict with natural threshold
# ==========================================
print("\n" + "="*60)
print(f"FINAL PREDICTION: {best_model}")
print("="*60)

best_clf = classifiers[best_model]

# 5-fold probability averaging for test
test_probs = np.zeros(len(test_X))
fold_accs, fold_f1s = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
    X_tr, y_tr = train_X[train_idx], train_y[train_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]

    best_clf.fit(X_tr, y_tr)
    val_pred = best_clf.predict(X_val)
    fold_acc = accuracy_score(y_val, val_pred)
    fold_f1 = f1_score(y_val, val_pred)
    fold_accs.append(fold_acc)
    fold_f1s.append(fold_f1)
    print(f"  Fold {fold+1}: Acc={fold_acc:.4f}, F1={fold_f1:.4f}")

    test_probs += best_clf.predict_proba(test_X)[:, 1] / n_splits

print(f"\nCV Acc: {np.mean(fold_accs):.4f}+/-{np.std(fold_accs):.4f}")
print(f"CV F1:  {np.mean(fold_f1s):.4f}+/-{np.std(fold_f1s):.4f}")

# Also evaluate on held-out validation set
best_clf.fit(train_X, train_y)
val_pred = best_clf.predict(val_X)
val_acc = accuracy_score(val_y, val_pred)
val_f1 = f1_score(val_y, val_pred)
print(f"\nHeld-out Val: Acc={val_acc:.4f}, F1={val_f1:.4f}")

# Natural threshold classification
test_labels = (test_probs > 0.5).astype(int)
n_meteor = int(test_labels.sum())
print(f"\nTest predictions (threshold=0.5):")
print(f"  Meteorites (1): {n_meteor}")
print(f"  Non-meteorites (0): {len(test_labels) - n_meteor}")

# ==========================================
# Generate submission
# ==========================================
print("\n=== Generating Submissions ===")

template = pd.read_csv(DATA_ROOT / 'sample_submission.csv')

# Natural threshold
sub_natural = template[['id']].copy()
sub_natural['label'] = sub_natural['id'].map(
    dict(zip(clean_test_ids, test_labels))
).fillna(0).astype(int)
natural_file = OUTPUT_DIR / 'submission_natural.csv'
sub_natural.to_csv(natural_file, index=False)
print(f"Natural threshold: {sub_natural['label'].sum()} positives -> {natural_file.name}")

# Also generate top-k variants for comparison
prob_df = pd.DataFrame({'id': clean_test_ids, 'prob': test_probs})
prob_df = prob_df.sort_values('prob', ascending=False).reset_index(drop=True)

for top_k in [75, 80, 85]:
    sub_topk = template[['id']].copy()
    topk_ids = set(prob_df.head(top_k)['id'])
    sub_topk['label'] = sub_topk['id'].apply(lambda x: 1 if x in topk_ids else 0)
    topk_file = OUTPUT_DIR / f'submission_top{top_k}.csv'
    sub_topk.to_csv(topk_file, index=False)
    print(f"Top-{top_k}: {topk_file.name}")

# Save probabilities
prob_df.to_csv(OUTPUT_DIR / 'test_probs.csv', index=False)

# Save summary
summary = {
    'model': 'DINOv2 ViT-L/14',
    'best_classifier': best_model,
    'cv_acc_mean': float(np.mean(fold_accs)),
    'cv_f1_mean': float(np.mean(fold_f1s)),
    'heldout_val_acc': float(val_acc),
    'heldout_val_f1': float(val_f1),
    'train_dedup_size': len(train_df),
    'test_size': len(test_df),
    'natural_meteorites': int(n_meteor),
}
with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs: {OUTPUT_DIR}")
print("Done!")
