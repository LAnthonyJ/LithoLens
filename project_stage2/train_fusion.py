"""
ResNet18 + DINOv2 ViT-L14 Feature fusion
ResNet18 captures local texture and DINOv2 captures global features; concatenate before classification
include supplementary data from Dataset/
"""
from pathlib import Path
import sys, json, random, hashlib, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = Path("/data/数据科学实践project")
RESNET_CKPT = DATA_ROOT / "runs/20260529_221800_img224_resnet18_adamw_lr0.0003_wd0.0001_bs16_val0.2_seed42_schcosine_losscross_entropy_pre1_subsetfull_e20/best_model.pt"
OUTPUT_DIR = DATA_ROOT / "runs" / f"fusion_resnet_dinov2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4
VAL_RATIO = 0.2

print(f"DEVICE: {DEVICE}")
print(f"OUTPUT: {OUTPUT_DIR}")
print(f"RENet CKPT: {RESNET_CKPT}")

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ============================================================
# Dataset & Data Utils (reuse)
# ============================================================
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}

class MeteoriteDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transforms = transforms
        self.has_label = "label" in self.df.columns
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["img_path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        if self.has_label: return image, int(row["label"])
        return image, str(row["img_path"])

def _resolve_image_dir(root, split):
    for c in [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]:
        if c.is_dir(): return c
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
                val_indices.extend(indices); current_val_count += len(indices)
            else: train_indices.extend(indices)
        if len(train_indices) == 0 and len(group_items) > 0:
            _, last_indices = group_items[-1]
            val_indices = [idx for idx in val_indices if idx not in last_indices]
            train_indices.extend(last_indices)
        train_parts.append(df.loc[train_indices]); val_parts.append(df.loc[val_indices])
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
    for bit in diff.flatten(): hash_value = (hash_value << 1) | int(bit)
    return hash_value

def hamming_distance_int(x, y): return (int(x) ^ int(y)).bit_count()

class UnionFind:
    def __init__(self, n): self.parent = list(range(n)); self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x: self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
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
        except: dhash_list.append(None); failed_paths.append(str(image_path))
    rep_df['dhash'] = pd.Series(dhash_list, dtype='object')
    valid_mask = rep_df['dhash'].notna()
    valid_rep_df = rep_df[valid_mask].reset_index(drop=True).copy()
    valid_rep_df['dhash'] = valid_rep_df['dhash'].map(int)
    n = len(valid_rep_df)
    uf = UnionFind(n)
    for i in tqdm(range(n), desc='near-dup family'):
        hi = valid_rep_df.loc[i, 'dhash']
        for j in range(i+1, n):
            if hamming_distance_int(hi, valid_rep_df.loc[j, 'dhash']) <= threshold: uf.union(i, j)
    root_to_family = {}
    family_ids = [];
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_family: root_to_family[root] = f'nearfam_{len(root_to_family):05d}'
        family_ids.append(root_to_family[root])
    valid_rep_df['near_dup_family_id'] = family_ids
    invalid_rep_df = rep_df[~valid_mask].copy()
    invalid_rep_df['near_dup_family_id'] = [f'nearfam_failed_{i:05d}' for i in range(len(invalid_rep_df))]
    rep_family_df = pd.concat([valid_rep_df[[md5_group_col,'dhash','near_dup_family_id']],
                                invalid_rep_df[[md5_group_col,'dhash','near_dup_family_id']]], axis=0, ignore_index=True)
    out_df = df.merge(rep_family_df[[md5_group_col,'near_dup_family_id']], on=md5_group_col, how='left')
    return out_df, rep_family_df, failed_paths

def drop_exact_duplicates_in_train(df, group_col='group_id'):
    candidate_cols = ['img_path','image_path','filepath','file_path','path']
    sort_cols = [c for c in candidate_cols if c in df.columns]
    if sort_cols: dedup = df.sort_values([group_col]+sort_cols).drop_duplicates(subset=group_col, keep='first').reset_index(drop=True)
    else: dedup = df.sort_values(group_col).drop_duplicates(subset=group_col, keep='first').reset_index(drop=True)
    return dedup, len(df)-len(dedup)

def load_supplementary_data(data_root):
    """Load supplementary Dataset/ images and original training data"""
    print("Load original training data...")
    full_df = load_train_dataframe(data_root)

    dataset_dir = Path(data_root) / "Dataset"
    rock_rows, meteorite_rows = [], []

    if dataset_dir.is_dir():
        for rock_type_dir in sorted(dataset_dir.iterdir()):
            if not rock_type_dir.is_dir() or rock_type_dir.name.startswith('.'):
                continue
            if rock_type_dir.name == 'Meteorite':
                for img_path in rock_type_dir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                        meteorite_rows.append({'id': f"Meteorite_{img_path.name}", 'label': 1, 'img_path': str(img_path)})
                continue
            for img_path in rock_type_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                    rock_rows.append({'id': f"{rock_type_dir.name}_{img_path.name}", 'label': 0, 'img_path': str(img_path)})

    rock_df = pd.DataFrame(rock_rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    meteorite_df = pd.DataFrame(meteorite_rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Limit the negative sample ratio
    target_ratio = 1.5
    n_pos = int(full_df['label'].sum())
    if target_ratio is not None:
        max_neg = int(n_pos * target_ratio)
        n_orig_neg = int((full_df['label'] == 0).sum())
        max_supp = max(0, max_neg - n_orig_neg)
        if len(rock_df) > max_supp:
            rock_df = rock_df.iloc[:max_supp]

    full_df = pd.concat([full_df, rock_df, meteorite_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Combined samples: {len(full_df)}, classes: {full_df['label'].value_counts().to_dict()}")
    return full_df

# ============================================================
# 1. Load data
# ============================================================
print("\n=== Load data ===")
full_train_df = load_supplementary_data(DATA_ROOT)
train_pool_df = attach_md5_group_ids(full_train_df)
print(f"MD5 groups: {train_pool_df['group_id'].nunique()}")

train_pool_df, _, _ = build_near_duplicate_families_from_md5_groups(train_pool_df, md5_group_col='group_id', threshold=6)
train_df, val_df = group_aware_train_val_split(train_pool_df, label_col='label', group_col='near_dup_family_id', val_ratio=VAL_RATIO, seed=SEED)
train_df, removed = drop_exact_duplicates_in_train(train_df)
test_df = load_test_dataframe(DATA_ROOT)

print(f"Train: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
print(f"Val: {len(val_df)} ({val_df['label'].value_counts().to_dict()})")
print(f"Test: {len(test_df)}")

# ============================================================
# 2. Load model
# ============================================================
print("\n=== Load model ===")

# ResNet18
from models import build_model
resnet = build_model('resnet18', num_classes=2, pretrained=False)
ckpt = torch.load(RESNET_CKPT, map_location='cpu')
resnet.load_state_dict(ckpt['model_state_dict'])
resnet.eval().to(DEVICE)
# Remove FC to get 512-dim features
resnet_features = nn.Sequential(*list(resnet.children())[:-1])  # → [B,512,1,1]
print(f"ResNet18 loaded (best val_f1={ckpt['val_f1']:.4f})")

# DINOv2
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2.eval().to(DEVICE)
print("DINOv2 ViT-L14 loaded")

# ============================================================
# 3. Feature extraction
# ============================================================
print("\n=== Feature extraction ===")

# ResNet18 preprocessing (as used for training)
resnet_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# DINOv2 Preprocessing
dinov2_tfms = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@torch.no_grad()
def extract_model_features(df, transform_fn, model, device, desc="Features"):
    """Extract one model's features in batches."""
    from torch.utils.data import DataLoader, Dataset

    class TempDataset(Dataset):
        def __init__(self, df, tfm):
            self.df = df.reset_index(drop=True)
            self.tfm = tfm
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            img = Image.open(row['img_path'])
            img = ImageOps.exif_transpose(img).convert('RGB')
            img = self.tfm(img)
            has_label = 'label' in row.index
            if has_label:
                return img, int(row['label'])
            return img, str(row['img_path'])

    ds = TempDataset(df, transform_fn)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    all_feats, all_labels, all_paths = [], [], []

    for batch in tqdm(dl, desc=desc):
        imgs = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            feats = model(imgs)
        all_feats.append(feats.detach().cpu().numpy())

        if isinstance(batch[1][0], str):  # test mode
            all_paths.extend(batch[1])
        else:
            all_labels.extend(batch[1].tolist())

    all_feats = np.concatenate(all_feats, axis=0)
    if all_labels:
        return all_feats, np.array(all_labels).astype(int)
    return all_feats, all_paths

# Extract training features
print("Extract ResNet18 training features...")
r_train_X, train_y = extract_model_features(train_df, resnet_tfms, resnet_features, DEVICE, "ResNet18 Train")
r_train_X = r_train_X.reshape(r_train_X.shape[0], -1)  # flatten to [N, 512]
print(f"ResNet18: {r_train_X.shape}")

print("Extract DINOv2 training embeddings...")
d_train_X, _ = extract_model_features(train_df, dinov2_tfms, dinov2, DEVICE, "DINOv2 Train")
print(f"DINOv2: {d_train_X.shape}")

train_X = np.concatenate([r_train_X, d_train_X], axis=1)
print(f"Fused Train: {train_X.shape}, labels: {np.bincount(train_y)}")

# Extract test features (single view for speed)
print("Extract ResNet18 test features...")
r_test_X, test_ids = extract_model_features(test_df, resnet_tfms, resnet_features, DEVICE, "ResNet18 Test")
r_test_X = r_test_X.reshape(r_test_X.shape[0], -1)
print(f"ResNet18 Test: {r_test_X.shape}")

print("Extract DINOv2 test embeddings...")
d_test_X, _ = extract_model_features(test_df, dinov2_tfms, dinov2, DEVICE, "DINOv2 Test")
print(f"DINOv2 Test: {d_test_X.shape}")

test_X = np.concatenate([r_test_X, d_test_X], axis=1)
test_X = normalize(test_X, norm='l2', axis=1)
print(f"Fused Test: {test_X.shape}")

# ============================================================
# 4. Train classifiers and write submissions
# ============================================================
print("\n=== classifier ===")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

classifiers = {
    'SVM_RBF': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED)),
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
}

results = {}
for name, clf in classifiers.items():
    print(f"\n{name}...")
    t0 = time.time()
    test_pred = np.zeros(len(test_X))
    val_f1s = []

    for tr_idx, val_idx in skf.split(train_X, train_y):
        X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
        X_val, y_val = train_X[val_idx], train_y[val_idx]
        clf.fit(X_tr, y_tr)
        val_pred = clf.predict(X_val)
        tp = ((val_pred==1)&(y_val==1)).sum()
        fp = ((val_pred==1)&(y_val==0)).sum()
        fn = ((val_pred==0)&(y_val==1)).sum()
        val_f1s.append(2*tp/max(2*tp+fp+fn,1))
        test_pred += clf.predict_proba(test_X)[:,1] / n_splits

    results[name] = test_pred
    print(f"  Val F1: {np.mean(val_f1s):.4f}, Time: {time.time()-t0:.0f}s")

# Ensemble
ep = (results['SVM_RBF'] + results['LogisticRegression']) / 2
results['Ensemble_SVM_LR'] = ep

# Generate submissions
print(f"\n=== SUBMISSIONS -> {OUTPUT_DIR} ===")
clean_ids = [Path(p).name for p in test_ids]

for name, pred in results.items():
    name_clean = name.replace(' ', '_')
    sub = pd.DataFrame({'id': clean_ids, 'prob': pred}).sort_values('prob', ascending=False).reset_index(drop=True)
    sub.to_csv(OUTPUT_DIR / f'test_probs_{name_clean}.csv', index=False, encoding='utf-8-sig')
    for tk in [75, 80, 85]:
        sub['label'] = 0; sub.loc[:tk-1,'label'] = 1
        tmpl = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
        tmpl['label'] = tmpl['id'].map(sub.set_index('id')['label']).astype(int)
        tmpl.to_csv(OUTPUT_DIR / f'submission_{name_clean}_top{tk}.csv', index=False)
    print(f"  {name:<25s} generated (top_75/80/85)")

# Save summary
summary = {
    'model': 'ResNet18 + DINOv2 ViT-L14 Fusion',
    'feature_dim': 1536,
    'train_size': len(train_df),
    'test_size': len(test_df),
    'resnet_val_f1': float(ckpt['val_f1']),
}
with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nDone! {OUTPUT_DIR}")
