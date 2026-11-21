"""
v13 Blend: DINOv3-L + ConvNeXtV2-Base + ConvNeXtV2-Large
==========================================================
Three models, each with flip TTA (orig + hflip + vflip), SVM+kNN 5-fold CV.
Final blend: 0.80 * DINOv3 + 0.15 * ConvNeXtV2-Base + 0.05 * ConvNeXtV2-Large
Threshold: 0.5

Based on experiment log — this blend achieved 0.83236 public F1 score.
"""
import os, sys, time, json, hashlib, random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
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
# Server paths (Linux)
DATA_ROOT = Path("/data/数据科学实践project")
DINOV3_LOCAL = Path("/data/models/dinov3_large")

# Local paths (Windows) — uncomment for local testing:
# DATA_ROOT = Path(__file__).resolve().parent
# DINOV3_LOCAL = DATA_ROOT / "meteor_models" / "dinov3_large"
OUTPUT_DIR = DATA_ROOT / "runs" / f"v13_blend_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4

# Blend weights (from experiment log v13)
W_DINOV3 = 0.80
W_CONVNEXT_BASE = 0.15
W_CONVNEXT_LARGE = 0.05
THRESHOLD = 0.5

# SVM+kNN ensemble weight
W_SVM = 0.7
W_KNN = 0.3

print(f"DEVICE: {DEVICE}")
print(f"OUTPUT: {OUTPUT_DIR}")
print(f"Blend: {W_DINOV3}*DINOv3 + {W_CONVNEXT_BASE}*ConvNeXtV2-B + {W_CONVNEXT_LARGE}*ConvNeXtV2-L")
print(f"Threshold: {THRESHOLD}")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed(SEED)


# ============================================================
# 1. Data loading utilities (reused from existing codebase)
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
# 2. Model-specific configurations
# ============================================================
MODEL_CONFIGS = {
    "dinov3_l": {
        "name": "DINOv3-L",
        "timm_name": "vit_large_patch16_dinov3.lvd1689m",
        "local_weights": DINOV3_LOCAL / "model.safetensors",
        "img_size": 384,
        "crop_pct": 1.0,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "weight": W_DINOV3,
    },
    "convnextv2_base": {
        "name": "ConvNeXtV2-Base",
        "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "local_weights": None,
        "img_size": 384,
        "crop_pct": 1.0,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "weight": W_CONVNEXT_BASE,
    },
    "convnextv2_large": {
        "name": "ConvNeXtV2-Large",
        "timm_name": "convnextv2_large.fcmae_ft_in22k_in1k_384",
        "local_weights": None,
        "img_size": 384,
        "crop_pct": 1.0,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "weight": W_CONVNEXT_LARGE,
    },
}


# ============================================================
# 3. Model loading
# ============================================================
def load_model(cfg):
    """Load a model from timm, optionally with local weights."""
    print(f"\nLoading {cfg['name']} ({cfg['timm_name']})...")

    local_weights = cfg.get("local_weights")
    use_local = local_weights is not None and local_weights.exists()

    if use_local:
        print(f"  Loading from local weights: {local_weights}")
        model = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=0)
        from safetensors.torch import load_file
        state = load_file(str(local_weights))
        model.load_state_dict(state, strict=True)
    else:
        model = timm.create_model(cfg["timm_name"], pretrained=True, num_classes=0)

    model.eval()
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {cfg['name']} loaded. Params: {n_params:.1f}M")
    return model


def build_transform(cfg):
    """Build the inference transform matching model config."""
    img_size = cfg["img_size"]
    mean = cfg["mean"]
    std = cfg["std"]
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ============================================================
# 4. Feature extraction with Flip TTA
# ============================================================
@torch.no_grad()
def extract_features(model, df, img_dir, base_tf, desc, batch_size, img_size):
    """Extract features WITHOUT TTA (for training set)."""
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
    """Extract features WITH Flip TTA (orig + hflip + vflip, average)."""
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
# 5. Classifier training (SVM + kNN 5-fold CV)
# ============================================================
def train_classifiers(train_X, train_y, model_name):
    """Train SVM(RBF) + kNN with 5-fold CV. Returns per-fold test predictions."""
    print(f"\n  Training SVM+kNN on {train_X.shape[0]} samples, {train_X.shape[1]} dims")
    print(f"  Labels: {np.bincount(train_y)}")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    svm_accs, knn_accs = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
        X_val, y_val = train_X[val_idx], train_y[val_idx]

        svm = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced",
                probability=True, random_state=SEED + fold),
        )
        svm.fit(X_tr, y_tr)
        svm_accs.append(svm.score(X_val, y_val))

        knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
        knn.fit(X_tr, y_tr)
        knn_accs.append(knn.score(X_val, y_val))

        print(f"    Fold {fold + 1}: SVM={svm_accs[-1]:.4f}, kNN={knn_accs[-1]:.4f}")

    print(f"    SVM avg: {np.mean(svm_accs):.4f} ± {np.std(svm_accs):.4f}")
    print(f"    kNN avg: {np.mean(knn_accs):.4f} ± {np.std(knn_accs):.4f}")

    # Train final classifiers on all data
    final_svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, class_weight="balanced",
            probability=True, random_state=SEED),
    )
    final_svm.fit(train_X, train_y)

    final_knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
    final_knn.fit(train_X, train_y)

    return final_svm, final_knn, {"svm_acc": float(np.mean(svm_accs)), "knn_acc": float(np.mean(knn_accs))}


# ============================================================
# 6. Main pipeline
# ============================================================
def main():
    print("=" * 70)
    print("v13 Blending Pipeline")
    print("=" * 70)

    # ---- Load data ----
    print("\n" + "=" * 70)
    print("Loading data")
    print("=" * 70)

    train_df = load_train_df(DATA_ROOT)
    test_df = load_test_df(DATA_ROOT)
    train_dir = _resolve_image_dir(DATA_ROOT, "train")
    test_dir = _resolve_image_dir(DATA_ROOT, "test")

    print(f"Train samples: {len(train_df)}")
    print(f"Train labels: {train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"Test samples: {len(test_df)}")

    # ---- Process each model ----
    model_probs = {}
    model_info = {}

    for model_key in ["dinov3_l", "convnextv2_base", "convnextv2_large"]:
        cfg = MODEL_CONFIGS[model_key]

        print(f"\n{'=' * 70}")
        print(f"Model: {cfg['name']} (weight={cfg['weight']})")
        print(f"{'=' * 70}")

        # Load model
        model = load_model(cfg)
        base_tf = build_transform(cfg)

        # Extract training features (no TTA)
        t0 = time.time()
        print(f"\n  Extracting training features (no TTA)...")
        train_X, train_y, _ = extract_features(
            model, train_df, train_dir, base_tf,
            f"Train {cfg['name']}", BATCH_SIZE, cfg["img_size"],
        )
        print(f"  Train features: {train_X.shape} ({time.time() - t0:.0f}s)")

        # Train classifiers
        svm, knn, cv_info = train_classifiers(train_X, train_y, cfg["name"])

        # Extract test features (WITH flip TTA)
        t0 = time.time()
        print(f"\n  Extracting test features (Flip TTA)...")
        test_X, test_ids = extract_features_tta(
            model, test_df, test_dir, base_tf,
            f"Test {cfg['name']}", BATCH_SIZE, cfg["img_size"],
        )
        test_X = normalize(test_X, norm="l2", axis=1)
        print(f"  Test features: {test_X.shape} ({time.time() - t0:.0f}s)")

        # Get probability predictions
        svm_probs = svm.predict_proba(test_X)[:, 1]
        knn_probs = knn.predict_proba(test_X)[:, 1]
        final_probs = W_SVM * svm_probs + W_KNN * knn_probs

        model_probs[model_key] = final_probs
        model_info[model_key] = {
            "name": cfg["name"],
            "weight": cfg["weight"],
            "feature_dim": int(train_X.shape[1]),
            "svm_probs": svm_probs.tolist(),
            "knn_probs": knn_probs.tolist(),
            **cv_info,
        }

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

        # Save per-model probabilities
        prob_df = pd.DataFrame({
            "id": [Path(p).name for p in test_ids],
            "prob": final_probs,
        }).sort_values("prob", ascending=False).reset_index(drop=True)
        prob_df.to_csv(
            OUTPUT_DIR / f"test_probs_{model_key}.csv",
            index=False, encoding="utf-8-sig",
        )

    # ---- Blend probabilities ----
    print(f"\n{'=' * 70}")
    print(f"Blending: {W_DINOV3}*DINOv3 + {W_CONVNEXT_BASE}*ConvNeXtV2-B + {W_CONVNEXT_LARGE}*ConvNeXtV2-L")
    print(f"{'=' * 70}")

    clean_ids = [Path(p).name for p in test_ids]

    blend_probs = (
        W_DINOV3 * model_probs["dinov3_l"]
        + W_CONVNEXT_BASE * model_probs["convnextv2_base"]
        + W_CONVNEXT_LARGE * model_probs["convnextv2_large"]
    )

    blend_df = pd.DataFrame({
        "id": clean_ids,
        "prob": blend_probs,
    }).sort_values("prob", ascending=False).reset_index(drop=True)

    blend_df.to_csv(
        OUTPUT_DIR / "test_probs_v13_blend.csv",
        index=False, encoding="utf-8-sig",
    )

    # ---- Generate submission (threshold 0.5) ----
    print(f"\n{'=' * 70}")
    print("Generating submission (threshold=0.5)")
    print(f"{'=' * 70}")

    template = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    id_to_prob = dict(zip(clean_ids, blend_probs))

    submission = template.copy()
    submission["label"] = submission["id"].map(
        lambda x: 1 if id_to_prob.get(x, 0) >= THRESHOLD else 0
    ).astype(int)

    n_pos = int(submission["label"].sum())
    n_neg = int((submission["label"] == 0).sum())
    print(f"Positives: {n_pos}, Negatives: {n_neg}")

    output_csv = OUTPUT_DIR / "submission_v13_blend_threshold05.csv"
    submission.to_csv(output_csv, index=False)
    print(f"Submission saved: {output_csv}")

    # Also save top-k variants for reference
    for k in [85, 88, 90, 92, 95]:
        sub_k = template.copy()
        top_ids = set(blend_df.head(k)["id"].tolist())
        sub_k["label"] = sub_k["id"].map(lambda x: 1 if x in top_ids else 0).astype(int)
        sub_k.to_csv(OUTPUT_DIR / f"submission_v13_blend_top{k}.csv", index=False)
        print(f"  Top-{k}: {int(sub_k['label'].sum())} positives")

    # ---- Save summary ----
    summary = {
        "version": "v13",
        "blend_weights": {
            "dinov3_l": W_DINOV3,
            "convnextv2_base": W_CONVNEXT_BASE,
            "convnextv2_large": W_CONVNEXT_LARGE,
        },
        "svm_knn_weights": {"svm": W_SVM, "knn": W_KNN},
        "threshold": THRESHOLD,
        "tta": "flip (orig + hflip + vflip avg)",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "model_details": model_info,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}")
    print(f"Done! All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
