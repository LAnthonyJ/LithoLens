"""
DINOv3-L + CLIP ViT-L/336 Blend (offline-compatible)
=====================================================
Uses models already available on the server — no internet required.
- DINOv3-L: local safetensors, Flip TTA, SVM+kNN 5-fold CV
- CLIP ViT-L/336: local HF model, Flip TTA, SVM+kNN 5-fold CV
- Blend: 0.85 * DINOv3 + 0.15 * CLIP, threshold 0.5

Based on v13 experiment log pattern: one strong backbone + correction branch.
"""
import os, sys, time, json, hashlib, random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageOps, ImageFile
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# 0. Configuration
# ============================================================
DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_DIR = DATA_ROOT / "runs" / f"blend_available_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 16

# Blend weights
W_DINOV3 = 0.85
W_CLIP = 0.15
THRESHOLD = 0.5
W_SVM = 0.7
W_KNN = 0.3

print(f"DEVICE: {DEVICE}")
print(f"OUTPUT: {OUTPUT_DIR}")
print(f"Blend: {W_DINOV3}*DINOv3 + {W_CLIP}*CLIP, threshold={THRESHOLD}")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed(SEED)


# ============================================================
# 1. Data utilities
# ============================================================
def _resolve_image_dir(root, split):
    for c in [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Cannot find {split} image directory")


def load_train_df(data_root):
    data_root = Path(data_root)
    train_dir = _resolve_image_dir(data_root, "train")
    df = pd.read_csv(data_root / "train_labels.csv")
    df["img_path"] = df["id"].apply(lambda x: str(train_dir / x))
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()
    df["label"] = df["label"].astype(int)
    return df


def load_test_df(data_root):
    data_root = Path(data_root)
    test_dir = _resolve_image_dir(data_root, "test")
    df = pd.read_csv(data_root / "sample_submission.csv")
    df = df[["id"]].copy()
    df["img_path"] = df["id"].apply(lambda x: str(test_dir / x))
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()
    return df


# ============================================================
# 2. Model loading
# ============================================================
def load_dinov3():
    """Load DINOv3-L from local safetensors."""
    print("\nLoading DINOv3-L from local safetensors...")
    model = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m",
        pretrained=False, num_classes=0,
    )
    from safetensors.torch import load_file
    state = load_file("/data/models/dinov3_large/model.safetensors")
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  DINOv3-L loaded. Params: {n_p:.1f}M")
    return model


def load_clip_vitl():
    """Load CLIP ViT-L/336 vision encoder from local HF model."""
    print("\nLoading CLIP ViT-L/336 vision encoder from local HF model...")
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained("/data/models/clip-vit-large-patch14-336")
    vision = model.vision_model
    # Wrap to return pooled features (like timm's forward_features + forward_head)
    class CLIPVisionWrapper(nn.Module):
        def __init__(self, vision_model):
            super().__init__()
            self.model = vision_model
        def forward(self, x):
            out = self.model(x, output_hidden_states=True)
            # Use the [CLS] token embedding as the image feature
            pooled = out.pooler_output  # [B, 1024]
            return pooled
    wrapper = CLIPVisionWrapper(vision)
    wrapper.eval().to(DEVICE)
    n_p = sum(p.numel() for p in wrapper.parameters()) / 1e6
    print(f"  CLIP ViT-L/336 loaded. Params: {n_p:.1f}M")
    return wrapper


# ============================================================
# 3. Transforms
# ============================================================
def build_dinov3_transform():
    return transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_clip_transform():
    return transforms.Compose([
        transforms.Resize(336),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                           std=(0.26862954, 0.26130258, 0.27577711)),
    ])


# ============================================================
# 4. Feature extraction
# ============================================================
@torch.no_grad()
def extract_features(model, df, img_dir, base_tf, desc, batch_size, img_size):
    features, labels, ids = [], [], []
    has_label = "label" in df.columns
    indices = list(range(0, len(df), batch_size))
    for start in tqdm(indices, desc=desc):
        batch_df = df.iloc[start:start + batch_size]
        tensors, batch_ids, batch_labels = [], [], []
        for _, row in batch_df.iterrows():
            img_path = f"{img_dir}/{row['id']}"
            try:
                img = Image.open(img_path).convert("RGB")
                tensors.append(base_tf(img))
                batch_ids.append(row["id"])
                if has_label:
                    batch_labels.append(int(row["label"]))
            except Exception:
                tensors.append(torch.zeros(3, img_size, img_size))
                batch_ids.append(row["id"])
                if has_label:
                    batch_labels.append(0)
        if tensors:
            batch = torch.stack(tensors).to(DEVICE)
            feats = model(batch).cpu().numpy()
            features.extend(feats)
            ids.extend(batch_ids)
            if has_label:
                labels.extend(batch_labels)
    X = np.array(features)
    if has_label:
        return X, np.array(labels), ids
    return X, ids


@torch.no_grad()
def extract_features_tta(model, df, img_dir, base_tf, desc, batch_size, img_size):
    """Flip TTA: orig + hflip + vflip, average features."""
    features, ids = [], []
    indices = list(range(0, len(df), batch_size))
    for start in tqdm(indices, desc=desc):
        batch_df = df.iloc[start:start + batch_size]
        all_tensors, batch_ids = [], []
        for _, row in batch_df.iterrows():
            img_path = f"{img_dir}/{row['id']}"
            try:
                img = Image.open(img_path).convert("RGB")
                all_tensors.append(base_tf(img))
                all_tensors.append(base_tf(hflip(img)))
                all_tensors.append(base_tf(vflip(img)))
                batch_ids.append(row["id"])
            except Exception:
                all_tensors.extend([torch.zeros(3, img_size, img_size)] * 3)
                batch_ids.append(row["id"])
        if all_tensors:
            batch = torch.stack(all_tensors).to(DEVICE)
            feats = model(batch).cpu().numpy()
            for i in range(len(batch_ids)):
                f_orig, f_h, f_v = feats[i * 3], feats[i * 3 + 1], feats[i * 3 + 2]
                features.append((f_orig + f_h + f_v) / 3.0)
            ids.extend(batch_ids)
    return np.array(features), ids


# ============================================================
# 5. Classifier training
# ============================================================
def train_classifiers(train_X, train_y):
    print(f"\n  Training SVM+kNN on {train_X.shape[0]} samples, {train_X.shape[1]} dims")
    print(f"  Labels: {np.bincount(train_y)}")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    svm_accs, knn_accs = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
        X_val, y_val = train_X[val_idx], train_y[val_idx]
        svm = make_pipeline(StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=SEED + fold))
        svm.fit(X_tr, y_tr)
        svm_accs.append(svm.score(X_val, y_val))
        knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
        knn.fit(X_tr, y_tr)
        knn_accs.append(knn.score(X_val, y_val))
        print(f"    Fold {fold + 1}: SVM={svm_accs[-1]:.4f}, kNN={knn_accs[-1]:.4f}")
    print(f"    SVM avg: {np.mean(svm_accs):.4f}, kNN avg: {np.mean(knn_accs):.4f}")
    final_svm = make_pipeline(StandardScaler(),
        SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=SEED))
    final_svm.fit(train_X, train_y)
    final_knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
    final_knn.fit(train_X, train_y)
    return final_svm, final_knn


# ============================================================
# 6. Main pipeline
# ============================================================
def main():
    print("=" * 70)
    print("DINOv3-L + CLIP ViT-L/336 Blend Pipeline")
    print("=" * 70)

    train_df = load_train_df(DATA_ROOT)
    test_df = load_test_df(DATA_ROOT)
    train_dir = _resolve_image_dir(DATA_ROOT, "train")
    test_dir = _resolve_image_dir(DATA_ROOT, "test")
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train labels: {train_df['label'].value_counts().sort_index().to_dict()}")

    model_probs = {}

    # ---- Process DINOv3-L ----
    print(f"\n{'=' * 70}")
    print("Model 1/2: DINOv3-L (weight=0.85)")
    print(f"{'=' * 70}")

    model = load_dinov3()
    base_tf = build_dinov3_transform()
    img_sz = 384

    t0 = time.time()
    train_X, train_y, _ = extract_features(
        model, train_df, train_dir, base_tf, "Train DINOv3-L", BATCH_SIZE, img_sz)
    print(f"  Train features: {train_X.shape} ({time.time() - t0:.0f}s)")

    svm, knn = train_classifiers(train_X, train_y)

    t0 = time.time()
    test_X, test_ids = extract_features_tta(
        model, test_df, test_dir, base_tf, "Test DINOv3-L (TTA)", BATCH_SIZE, img_sz)
    test_X = normalize(test_X, norm="l2", axis=1)
    print(f"  Test features: {test_X.shape} ({time.time() - t0:.0f}s)")

    svm_p = svm.predict_proba(test_X)[:, 1]
    knn_p = knn.predict_proba(test_X)[:, 1]
    model_probs["dinov3_l"] = W_SVM * svm_p + W_KNN * knn_p

    clean_ids = [Path(p).name for p in test_ids]
    pd.DataFrame({"id": clean_ids, "prob": model_probs["dinov3_l"]})\
        .sort_values("prob", ascending=False)\
        .to_csv(OUTPUT_DIR / "test_probs_dinov3_l.csv", index=False, encoding="utf-8-sig")

    del model; torch.cuda.empty_cache()

    # ---- Process CLIP ViT-L/336 ----
    print(f"\n{'=' * 70}")
    print("Model 2/2: CLIP ViT-L/336 (weight=0.15)")
    print(f"{'=' * 70}")

    model = load_clip_vitl()
    base_tf = build_clip_transform()
    img_sz = 336

    t0 = time.time()
    train_X, train_y, _ = extract_features(
        model, train_df, train_dir, base_tf, "Train CLIP ViT-L", BATCH_SIZE, img_sz)
    print(f"  Train features: {train_X.shape} ({time.time() - t0:.0f}s)")

    svm, knn = train_classifiers(train_X, train_y)

    t0 = time.time()
    test_X, _ = extract_features_tta(
        model, test_df, test_dir, base_tf, "Test CLIP ViT-L (TTA)", BATCH_SIZE, img_sz)
    test_X = normalize(test_X, norm="l2", axis=1)
    print(f"  Test features: {test_X.shape} ({time.time() - t0:.0f}s)")

    svm_p = svm.predict_proba(test_X)[:, 1]
    knn_p = knn.predict_proba(test_X)[:, 1]
    model_probs["clip_vitl"] = W_SVM * svm_p + W_KNN * knn_p

    pd.DataFrame({"id": clean_ids, "prob": model_probs["clip_vitl"]})\
        .sort_values("prob", ascending=False)\
        .to_csv(OUTPUT_DIR / "test_probs_clip_vitl.csv", index=False, encoding="utf-8-sig")

    del model; torch.cuda.empty_cache()

    # ---- Blend ----
    print(f"\n{'=' * 70}")
    print(f"Blending: {W_DINOV3}*DINOv3 + {W_CLIP}*CLIP")
    print(f"{'=' * 70}")

    blend = W_DINOV3 * model_probs["dinov3_l"] + W_CLIP * model_probs["clip_vitl"]

    blend_df = pd.DataFrame({"id": clean_ids, "prob": blend})\
        .sort_values("prob", ascending=False).reset_index(drop=True)
    blend_df.to_csv(OUTPUT_DIR / "test_probs_blend.csv", index=False, encoding="utf-8-sig")

    # Submission (threshold 0.5)
    template = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    id_to_prob = dict(zip(clean_ids, blend))

    sub = template.copy()
    sub["label"] = sub["id"].map(lambda x: 1 if id_to_prob.get(x, 0) >= THRESHOLD else 0).astype(int)
    n_pos = int(sub["label"].sum())
    print(f"Threshold {THRESHOLD}: {n_pos} positives, {len(sub) - n_pos} negatives")

    sub.to_csv(OUTPUT_DIR / "submission_blend_threshold05.csv", index=False)

    # Top-k variants
    for k in [85, 88, 90, 92, 95]:
        sub_k = template.copy()
        top_ids = set(blend_df.head(k)["id"].tolist())
        sub_k["label"] = sub_k["id"].map(lambda x: 1 if x in top_ids else 0).astype(int)
        sub_k.to_csv(OUTPUT_DIR / f"submission_blend_top{k}.csv", index=False)
        print(f"  Top-{k}: {int(sub_k['label'].sum())} positives")

    # Summary
    summary = {
        "blend_weights": {"dinov3_l": W_DINOV3, "clip_vitl": W_CLIP},
        "threshold": THRESHOLD,
        "svm_knn_weights": {"svm": W_SVM, "knn": W_KNN},
        "tta": "flip (orig + hflip + vflip avg)",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "n_positives": n_pos,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
