"""
CLIP ViT-L/14 feature extraction + classifier for meteorite identification
Approach: Use CLIP image encoder to extract features, then train a classifier
on training data. This combines a mature pre-trained model with task-specific learning.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ==========================================
# Configuration
# ==========================================
ROOT = Path(r"H:\数据科学与大数据技术\数据科学实践\project1\project_stage2")
TRAIN_CSV = ROOT / "train_labels.csv"
TEMPLATE_CSV = ROOT / "sample_submission.csv"
OUTPUT_CSV = ROOT / "submission_clip_svm.csv"

# Auto-detect image directories
def find_image_dir(base, name):
    base = base / f"{name}_images"
    if base.is_dir():
        inner = list(base.iterdir())
        if len(inner) == 1 and inner[0].is_dir():
            return inner[0]
        return base
    return base

TRAIN_DIR = find_image_dir(ROOT, "train")
TEST_DIR = find_image_dir(ROOT, "test")
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"TEST_DIR: {TEST_DIR}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8  # Small batch for CPU
IMAGE_SIZE = 224
SEED = 42

# ==========================================
# Load CLIP (ViT-L/14 for best quality)
# ==========================================
from transformers import CLIPProcessor, CLIPModel

# Use ViT-L/14 for higher accuracy (24M+ downloads, very mature)
MODEL_ID = "openai/clip-vit-base-patch32"  # ViT-B/32 is faster on CPU, already cached
print(f"\nLoading {MODEL_ID}...")
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded!")

# ==========================================
# Extract CLIP image embeddings
# ==========================================
@torch.no_grad()
def extract_clip_embeddings(image_paths, model, processor, device, batch_size=8):
    """Extract CLIP image embeddings for a list of image paths"""
    all_embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_batch_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
                valid_batch_paths.append(p)
            except Exception as e:
                print(f"  Error loading {p}: {e}")
                continue

        if not batch_images:
            continue

        inputs = processor(
            images=batch_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        # Get image embeddings from CLIP vision encoder
        outputs = model.get_image_features(**inputs)
        # outputs is BaseModelOutputWithPooling, get the pooler_output
        embeddings = outputs.pooler_output.cpu().numpy()

        # L2 normalize (standard for CLIP embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        all_embeddings.append(embeddings)
        valid_paths.extend(valid_batch_paths)

    return np.concatenate(all_embeddings, axis=0), valid_paths

# ==========================================
# Load training data
# ==========================================
print("\n=== Loading Training Data ===")
df_train = pd.read_csv(TRAIN_CSV)
train_paths = [str(TRAIN_DIR / rid) for rid in df_train['id']]
train_labels = df_train['label'].values.astype(int)

# Filter to existing files
exist_mask = [Path(p).exists() for p in train_paths]
train_paths = [p for p, m in zip(train_paths, exist_mask) if m]
train_labels = train_labels[exist_mask]
print(f"Training images: {len(train_paths)}")
print(f"Label distribution: {np.bincount(train_labels)}")

# ==========================================
# Load test data
# ==========================================
print("\n=== Loading Test Data ===")
df_template = pd.read_csv(TEMPLATE_CSV)
test_ids = df_template['id'].tolist()
test_paths = [str(TEST_DIR / rid) for rid in test_ids]

exist_test = [Path(p).exists() for p in test_paths]
test_paths = [p for p, m in zip(test_paths, exist_test) if m]
test_ids = [rid for rid, m in zip(test_ids, exist_test) if m]
print(f"Test images: {len(test_paths)}")

# ==========================================
# Extract CLIP features
# ==========================================
print("\n=== Extracting CLIP Features ===")

train_emb, train_valid_paths = extract_clip_embeddings(
    train_paths, model, processor, DEVICE, batch_size=BATCH_SIZE
)
# Re-align labels with valid paths
train_path_set = set(train_valid_paths)
train_label_aligned = []
for p, l in zip(train_paths, train_labels):
    if p in train_path_set:
        train_label_aligned.append(l)
train_labels_aligned = np.array(train_label_aligned)

print(f"Train embeddings: {train_emb.shape}")
print(f"Train labels: {np.bincount(train_labels_aligned)}")

test_emb, test_valid_paths = extract_clip_embeddings(
    test_paths, model, processor, DEVICE, batch_size=BATCH_SIZE
)
# Map valid paths back to ids
test_valid_set = set(test_valid_paths)
test_ids_valid = []
for p, rid in zip(test_paths, test_ids):
    if p in test_valid_set:
        test_ids_valid.append(rid)
print(f"Test embeddings: {test_emb.shape}")

# ==========================================
# 5-fold CV to find best classifier
# ==========================================
print("\n=== 5-Fold Cross Validation ===")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

classifiers = {
    "SVM_RBF": make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED)
    ),
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
    ),
}

for name, clf in classifiers.items():
    scores = cross_val_score(clf, train_emb, train_labels_aligned, cv=skf, scoring='accuracy')
    print(f"  {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

# ==========================================
# Train best classifier and predict
# ==========================================
print("\n=== Training SVM + Predicting Test Set ===")
clf = classifiers["SVM_RBF"]

# 5-fold probability averaging for test predictions
test_probs = np.zeros(len(test_emb))
for fold, (train_idx, val_idx) in enumerate(skf.split(train_emb, train_labels_aligned)):
    X_tr, y_tr = train_emb[train_idx], train_labels_aligned[train_idx]
    X_val, y_val = train_emb[val_idx], train_labels_aligned[val_idx]

    clf.fit(X_tr, y_tr)
    val_acc = clf.score(X_val, y_val)
    print(f"  Fold {fold+1}: val_acc={val_acc:.4f}")
    test_probs += clf.predict_proba(test_emb)[:, 1] / n_splits

# ==========================================
# Generate submission (Top-85 for consistency with existing approach)
# ==========================================
print("\n=== Generating Submissions ===")

# Sort by probability descending
sub_df = pd.DataFrame({'id': test_ids_valid, 'prob': test_probs})
sub_df = sub_df.sort_values('prob', ascending=False).reset_index(drop=True)

# Multiple thresholds
for top_k in [75, 80, 85]:
    sub_df[f'label_{top_k}'] = 0
    sub_df.loc[:top_k - 1, f'label_{top_k}'] = 1

    # Ensure all test ids are in submission (missing = 0)
    out_df = df_template[['id']].copy()
    out_df['label'] = out_df['id'].map(
        sub_df.set_index('id')[f'label_{top_k}'].to_dict()
    ).fillna(0).astype(int)

    output_file = ROOT / f'submission_clip_vitl14_top{top_k}.csv'
    out_df.to_csv(output_file, index=False)
    n_pos = int(out_df['label'].sum())
    print(f"  top_{top_k}: {n_pos} positive -> {output_file.name}")

# Also save probabilities for reference
sub_df.to_csv(ROOT / 'test_pred_probs_clip_vitl14.csv', index=False)
print(f"\nFull probabilities saved to test_pred_probs_clip_vitl14.csv")
print("\nDone!")
