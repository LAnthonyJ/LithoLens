"""
Simplified DINOv2 ViT-L/14 + SVM with natural classification
Runs on GPU server. Robust error handling & intermediate saves.
"""
import os, sys, time, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = Path('/data/数据科学实践project')
OUTPUT_DIR = DATA_ROOT / 'dinov2_simple_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda')
BATCH_SIZE = 32
SEED = 42

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Output: {OUTPUT_DIR}")

# Simple dataset class
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img = ImageOps.exif_transpose(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img, self.paths[idx]

# Load data
def _resolve_image_dir(root, split):
    candidates = [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]
    for c in candidates:
        if c.is_dir():
            print(f"  Found {split} dir: {c}")
            return c
    raise FileNotFoundError(f"Cannot find {split} images dir")

train_dir = _resolve_image_dir(DATA_ROOT, "train")
test_dir = _resolve_image_dir(DATA_ROOT, "test")

df_train = pd.read_csv(DATA_ROOT / "train_labels.csv")
df_test = pd.read_csv(DATA_ROOT / "sample_submission.csv")[["id"]]

# Debug: check data
print(f"df_train columns: {df_train.columns.tolist()}")
print(f"df_train head: {df_train.head(2).to_dict()}")

train_paths = [str(train_dir / rid) for rid in df_train['id']]
train_labels = df_train['label'].values.astype(int)
test_paths = [str(test_dir / rid) for rid in df_test['id']]
test_ids = df_test['id'].tolist()

print(f"First train path: {train_paths[0]}")
print(f"Exists: {Path(train_paths[0]).exists()}")

# Filter existing
train_mask = [Path(p).exists() for p in train_paths]
train_paths = [p for p, m in zip(train_paths, train_mask) if m]
train_labels = train_labels[train_mask]
test_mask = [Path(p).exists() for p in test_paths]
test_paths = [p for p, m in zip(test_paths, test_mask) if m]
test_ids = [rid for rid, m in zip(test_ids, test_mask) if m]

print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")
print(f"Train dist: {np.bincount(train_labels)}")

# Transform
transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# TTA transforms for test
tta_transforms = [
    transform,
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transform]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transform]),
]

# Load DINOv2
print("\nLoading DINOv2 ViT-L/14...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', trust_repo=True)
dinov2.eval()
dinov2.to(DEVICE)
print("Loaded!")

# Extract features
@torch.no_grad()
def extract(paths, labels, model, device, transform, batch_size, desc="Extracting"):
    dataset = SimpleDataset(paths, labels, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    feats, labs, all_paths = [], [], []
    for batch in tqdm(loader, desc=desc):
        if labels is not None:
            imgs, lbs = batch
            labs.extend(lbs.cpu().tolist())
        else:
            imgs, paths_batch = batch
            all_paths.extend(paths_batch)
        feats.append(model(imgs.to(device)).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    if labels is not None:
        return feats, np.array(labs)
    return feats, all_paths

# Extract train features
print("\nExtracting train features...")
train_X, train_y = extract(train_paths, train_labels, dinov2, DEVICE, transform, BATCH_SIZE)
print(f"Train features: {train_X.shape}")

# Extract test features with TTA
print("\nTTA test feature extraction...")
test_X_tta = []
for i, t in enumerate(tta_transforms):
    curr_X, ids = extract(test_paths, None, dinov2, DEVICE, t, BATCH_SIZE, desc=f"TTA {i+1}/3")
    test_X_tta.append(curr_X)
test_X = np.mean(test_X_tta, axis=0)

# L2 normalize
from sklearn.preprocessing import normalize
test_X = normalize(test_X, norm='l2', axis=1)
print(f"Test features: {test_X.shape}")

# 5-fold CV
print("\n5-fold CV...")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

for name, clf in [
    ("SVM_RBF", make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED))),
    ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED))),
]:
    cv_acc = cross_val_score(clf, train_X, train_y, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(clf, train_X, train_y, cv=skf, scoring='f1')
    print(f"  {name}: Acc={cv_acc.mean():.4f}+/-{cv_acc.std():.4f}, F1={cv_f1.mean():.4f}+/-{cv_f1.std():.4f}")

# Train SVM with 5-fold probability averaging
print("\nTraining SVM with 5-fold averaging...")
best_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED))

test_probs = np.zeros(len(test_X))
fold_accs, fold_f1s = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
    X_tr, y_tr = train_X[train_idx], train_y[train_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]
    best_clf.fit(X_tr, y_tr)
    val_pred = best_clf.predict(X_val)
    fold_accs.append(accuracy_score(y_val, val_pred))
    fold_f1s.append(f1_score(y_val, val_pred))
    print(f"  Fold {fold+1}: Acc={fold_accs[-1]:.4f}, F1={fold_f1s[-1]:.4f}")
    test_probs += best_clf.predict_proba(test_X)[:, 1] / n_splits

print(f"\nCV Acc: {np.mean(fold_accs):.4f}+/-{np.std(fold_accs):.4f}")
print(f"CV F1:  {np.mean(fold_f1s):.4f}+/-{np.std(fold_f1s):.4f}")

# Natural threshold
test_labels = (test_probs > 0.5).astype(int)
n_meteor = int(test_labels.sum())
print(f"\n=== NATURAL CLASSIFICATION (threshold=0.5) ===")
print(f"Meteorites: {n_meteor}")
print(f"Non-meteorites: {len(test_labels) - n_meteor}")

# Probability distribution
print(f"\nProb distribution:")
print(f"  Min: {test_probs.min():.4f}")
print(f"  Max: {test_probs.max():.4f}")
print(f"  Mean: {test_probs.mean():.4f}")
print(f"  Median: {np.median(test_probs):.4f}")
print(f"  >0.5: {(test_probs > 0.5).sum()}")
print(f"  >0.7: {(test_probs > 0.7).sum()}")
print(f"  >0.9: {(test_probs > 0.9).sum()}")
print(f"  <0.3: {(test_probs < 0.3).sum()}")

# Top predictions
order = np.argsort(-test_probs)
print(f"\nTop-10 meteorites:")
for i in order[:10]:
    print(f"  {test_ids[i]}: {test_probs[i]:.4f}")

# Save
clean_ids = [Path(p).name for p in ids]
prob_df = pd.DataFrame({'id': clean_ids, 'prob': test_probs})
prob_df.to_csv(OUTPUT_DIR / 'test_probs_dinov2.csv', index=False)

# Natural threshold submission
template = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
sub = template[['id']].copy()
sub['label'] = sub['id'].map(dict(zip(clean_ids, test_labels))).fillna(0).astype(int)
sub.to_csv(OUTPUT_DIR / 'submission_dinov2_natural.csv', index=False)
print(f"\nSubmission saved: {OUTPUT_DIR / 'submission_dinov2_natural.csv'}")

# Save summary
summary = {
    'model': 'DINOv2 ViT-L/14',
    'classifier': 'SVM_RBF',
    'cv_acc_mean': float(np.mean(fold_accs)),
    'cv_acc_std': float(np.std(fold_accs)),
    'cv_f1_mean': float(np.mean(fold_f1s)),
    'cv_f1_std': float(np.std(fold_f1s)),
    'train_size': len(train_paths),
    'test_size': len(test_paths),
    'natural_meteorites': int(n_meteor),
    'prob_max': float(test_probs.max()),
    'prob_min': float(test_probs.min()),
}
with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nDone!")
