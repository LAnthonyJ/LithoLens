"""
DINOv2 ViT-L14 features with multiple classifiers
Compare classifiers on a shared set of DINOv2 embeddings
"""
from pathlib import Path
import sys
import json
import hashlib
import random
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# Configuration
# ==========================================
DATA_ROOT = Path("/data/数据科学实践project")
FEATURES_DIR = DATA_ROOT / "runs" / "dinov2_features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = DATA_ROOT / "runs" / f"classifier_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4

print(f"DEVICE: {DEVICE}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==========================================
# Dataset and loading (simplified implementation)
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


def _resolve_image_dir(root, split):
    candidates = [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Missing {split} image directory")


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
# 1. Load or extract features
# ==========================================
print("\n" + "=" * 60)
print("Preprocessing and feature extraction")
print("=" * 60)

train_feat_path = FEATURES_DIR / "train_features.npz"
test_feat_path = FEATURES_DIR / "test_features.npz"

if train_feat_path.exists() and test_feat_path.exists():
    print("Loading cached features...")
    train_data = np.load(train_feat_path)
    train_X = train_data['X']
    train_y = train_data['y']
    test_data = np.load(test_feat_path)
    test_X = test_data['X']
    test_ids = test_data['ids'].tolist()
    print(f"Training features: {train_X.shape}, Test features: {test_X.shape}")
else:
    print("Extract features again...")

    # Load model
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dinov2_model.eval()
    dinov2_model.to(DEVICE)
    print("DINOv2 ViT-L14 loaded")

    # Load data
    full_train_df = load_train_dataframe(DATA_ROOT)
    train_pool_df = attach_md5_group_ids(full_train_df)
    train_pool_df, _, _ = build_near_duplicate_families_from_md5_groups(train_pool_df, md5_group_col='group_id', threshold=6)
    train_df, val_df = group_aware_train_val_split(train_pool_df, label_col='label', group_col='near_dup_family_id', val_ratio=0.2, seed=SEED)
    train_df, _ = drop_exact_duplicates_in_train(train_df)
    test_df = load_test_dataframe(DATA_ROOT)

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
        for batch in tqdm(loader, desc='Extracting Features'):
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

    # Extract training features
    train_dataset = MeteoriteDataset(train_df, transforms=base_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    train_X, train_y = extract_embeddings(train_loader, dinov2_model, DEVICE, is_test=False)

    # TTA Extract test features
    tta_transforms = [
        base_transform,
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), base_transform]),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0), base_transform])
    ]
    test_X_tta = []
    for t in tqdm(tta_transforms, desc='TTA'):
        test_dataset = MeteoriteDataset(test_df, transforms=t)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        curr_X, ids = extract_embeddings(test_loader, dinov2_model, DEVICE, is_test=True)
        test_X_tta.append(curr_X)
    test_X = np.mean(test_X_tta, axis=0)
    test_X = normalize(test_X, norm='l2', axis=1)
    test_ids = ids

    # Cache features
    np.savez_compressed(train_feat_path, X=train_X, y=train_y)
    np.savez_compressed(test_feat_path, X=test_X, ids=np.array(test_ids))
    print(f"Features cached at {FEATURES_DIR}")

print(f"\n Training set: {train_X.shape}, Class distribution: {np.bincount(train_y.astype(int))}")
print(f"Test set: {test_X.shape}")

# ==========================================
# 2. Define candidate classifiers
# ==========================================
print("\n" + "=" * 60)
print("5 fold cross-validation and classifier comparison")
print("=" * 60)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

classifiers = {
    # Linear models
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
    ),
    # SVM variants
    "SVM_RBF": make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED)
    ),
    "SVM_Linear": make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=SEED)
    ),
    # kNN
    "kNN_k25_cosine": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine')
    ),
    "kNN_k15_euclidean": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean')
    ),
    # Tree models
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=15, class_weight='balanced',
        random_state=SEED, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=SEED
    ),
    # XGBoost
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=(train_y == 0).sum() / max((train_y == 1).sum(), 1),
        random_state=SEED, verbosity=0
    ),
    # LightGBM
    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        class_weight='balanced', random_state=SEED,
        verbose=-1
    ),
}

results = []

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")

    # 5 fold cross-validation
    val_accs, val_f1s = [], []
    test_pred = np.zeros(len(test_X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        X_tr, y_tr = train_X[train_idx], train_y[train_idx]
        X_val, y_val = train_X[val_idx], train_y[val_idx]

        clf.fit(X_tr, y_tr)

        # Validation evaluation
        val_pred = clf.predict(X_val)
        val_acc = (val_pred == y_val).mean()
        tp = ((val_pred == 1) & (y_val == 1)).sum()
        fp = ((val_pred == 1) & (y_val == 0)).sum()
        fn = ((val_pred == 0) & (y_val == 1)).sum()
        val_f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # Test predictions (average probabilities)
        test_pred += clf.predict_proba(test_X)[:, 1] / n_splits

    mean_acc = np.mean(val_accs)
    std_acc = np.std(val_accs)
    mean_f1 = np.mean(val_f1s)
    std_f1 = np.std(val_f1s)

    results.append({
        'name': name,
        'val_acc_mean': mean_acc,
        'val_acc_std': std_acc,
        'val_f1_mean': mean_f1,
        'val_f1_std': std_f1,
        'test_prob_max': test_pred.max(),
        'test_prob_min': test_pred.min(),
        'test_prob_mean': test_pred.mean(),
    })

    print(f"  Val Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Val F1:  {mean_f1:.4f} ± {std_f1:.4f}")

# ==========================================
# 3. Compare results
# ==========================================
print("\n" + "=" * 60)
print("Classifier comparison (sorted by validation F1)")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values('val_f1_mean', ascending=False)
print(f"\n{'Name':<25s} {'Val F1':>10s} {'Val Acc':>10s}")
print("-" * 50)
for _, row in results_df.iterrows():
    print(f"{row['name']:<25s} {row['val_f1_mean']:.4f}±{row['val_f1_std']:.4f}  {row['val_acc_mean']:.4f}±{row['val_acc_std']:.4f}")

# Save full results
results_df.to_csv(OUTPUT_DIR / 'classifier_comparison.csv', index=False)
print(f"\n Comparison results saved: {OUTPUT_DIR / 'classifier_comparison.csv'}")

# ==========================================
# 4. Write submissions for the top three classifiers
# ==========================================
print("\n" + "=" * 60)
print("Write submissions for the top three classifiers")
print("=" * 60)

# Refit the three highest-ranked classifiers from cross-validation
top3_names = results_df.head(3)['name'].tolist()

for rank, name in enumerate(top3_names):
    print(f"\n--- {rank+1}. {name} ---")

    clf = classifiers[name]
    clf.fit(train_X, train_y)

    # 5 fold probability average
    test_pred = np.zeros(len(test_X))
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        X_tr, y_tr = train_X[train_idx], train_y[train_idx]
        clf.fit(X_tr, y_tr)
        test_pred += clf.predict_proba(test_X)[:, 1] / n_splits

    clean_ids = [Path(p).name for p in test_ids]
    sub_df = pd.DataFrame({'id': clean_ids, 'prob': test_pred})
    sub_df = sub_df.sort_values('prob', ascending=False).reset_index(drop=True)

    for top_k in [75, 80, 85]:
        sub_df[f'label_{top_k}'] = 0
        sub_df.loc[:top_k - 1, f'label_{top_k}'] = 1

        template_df = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
        template_df['label'] = template_df['id'].map(
            sub_df.set_index('id')[f'label_{top_k}'].to_dict()
        )
        template_df['label'] = template_df['label'].astype(int)

        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        output_csv = OUTPUT_DIR / f'submission_{safe_name}_top{top_k}.csv'
        template_df.to_csv(output_csv, index=False)
        print(f"  top_{top_k}: {int(sub_df[f'label_{top_k}'].sum())} positives -> {output_csv.name}")

# ==========================================
# 5. ensemble Top-3
# ==========================================
print(f"\n--- Ensemble the three best classifiers ---")
ensemble_pred = np.zeros(len(test_X))
for name in top3_names:
    clf = classifiers[name]
    test_pred = np.zeros(len(test_X))
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        X_tr, y_tr = train_X[train_idx], train_y[train_idx]
        clf.fit(X_tr, y_tr)
        test_pred += clf.predict_proba(test_X)[:, 1] / n_splits
    ensemble_pred += test_pred / len(top3_names)

clean_ids = [Path(p).name for p in test_ids]
sub_df = pd.DataFrame({'id': clean_ids, 'prob': ensemble_pred})
sub_df = sub_df.sort_values('prob', ascending=False).reset_index(drop=True)

# Save ensemble probabilities
sub_df.to_csv(OUTPUT_DIR / 'test_pred_probs_ensemble_top3.csv', index=False, encoding='utf-8-sig')

for top_k in [75, 80, 85]:
    sub_df['label'] = 0
    sub_df.loc[:top_k - 1, 'label'] = 1

    template_df = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    template_df['label'] = template_df['id'].map(sub_df.set_index('id')['label'])
    template_df['label'] = template_df['label'].astype(int)

    output_csv = OUTPUT_DIR / f'submission_ensemble_top3_top{top_k}.csv'
    template_df.to_csv(output_csv, index=False)
    print(f"  ensemble top_{top_k}: {int(sub_df['label'].sum())} positives -> {output_csv.name}")

print("\n🎉 Finished!")
print(f"Outputs: {OUTPUT_DIR}")
