"""
Texture-Feature Fusion Pipeline — per texture_feature_reference.py

Pipeline:
1. DINOv3-L frozen feature extraction (1024-dim) + TTA → L2 normalize
2. 21-dim texture features (cv2) → RobustScaler
3. Concatenate → SelectKBest(k=512) → L2 normalize
4. SVM(RBF) + kNN(cosine) 5-fold CV ensemble → (svm + knn) / 2
5. Blend with ConvNeXtV2-Base: 0.80 * DINO+texture + 0.15 * ConvB
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, time, pickle, numpy as np, pandas as pd
import torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path
from PIL import Image, ImageOps
import cv2

DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "v13_proper" / "texture_fusion"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH = 16
IMG_SIZE = 512

np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)
print(f"Device: {DEVICE}")


# ================================================================
# 1. Texture Feature Extractor (EXACT from texture_feature_reference.py)
# ================================================================
def compute_texture_features(image):
    """21-dim texture features from PIL RGB image."""
    image = image.resize((256, 256), Image.LANCZOS)
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    features = []

    # Multi-scale local variance (15 dims)
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

    # Global contrast + Gradient energy (4 dims)
    features.append(float(np.std(gray)))
    features.append(float(np.max(gray) - np.min(gray)))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    features.append(float(np.mean(grad_mag)))
    features.append(float(np.std(grad_mag)))

    # FFT high-frequency energy ratio (1 dim)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    ch, cw = h // 2, w // 2
    radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask_high = ((y - ch) ** 2 + (x - cw) ** 2) > radius ** 2
    features.append(float((magnitude[mask_high] ** 2).sum() / ((magnitude ** 2).sum() + 1e-8)))

    # Histogram entropy (1 dim)
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    features.append(float(-np.sum(hist * np.log2(hist))))

    return np.array(features, dtype=np.float64)


def extract_texture_batch(df, img_dir, desc="Texture"):
    """Extract 21-dim texture features for a dataframe, with RobustScaler."""
    feats = []
    img_dir = str(img_dir)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            img = Image.open(f"{img_dir}/{row['id']}")
            img = ImageOps.exif_transpose(img).convert("RGB")
            feats.append(compute_texture_features(img))
        except Exception:
            feats.append(np.zeros(21, dtype=np.float64))
    mat = np.stack(feats, axis=0)  # (N, 21)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = RobustScaler().fit(mat)
    return scaler.transform(mat).astype(np.float32), scaler


# ================================================================
# 2. DINOv3-L Feature Extraction with TTA
# ================================================================
class FeatDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["id"]).convert("RGB")
        if self.transform: img = self.transform(img)
        if self.is_test: return img, row["id"]
        return img, int(row["label"])


def build_dino_transform():
    return transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dinov3():
    model = timm.create_model("vit_large_patch16_dinov3.lvd1689m",
        pretrained=True, num_classes=0, img_size=IMG_SIZE,
        pretrained_cfg_overlay={"file": "/data/models/dinov3_large/model.safetensors"})
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def extract_dino_features(model, df, img_dir, base_tf, desc, is_test=False):
    """Extract DINO features. For test, 4-view TTA averaging at feature level."""
    model.eval()
    if is_test:
        # TTA: orig + hflip + vflip, average features
        feats_sum = None
        ids_ref = None
        tta_ops = [
            lambda img: img,
            lambda img: hflip(img),
            lambda img: vflip(img),
        ]
        for view_idx, op in enumerate(tta_ops):
            features, ids, labels = [], [], []
            indices = list(range(0, len(df), BATCH))
            for start in tqdm(indices, desc=f"{desc} view{view_idx+1}"):
                batch_df = df.iloc[start:start + BATCH]
                tensors, batch_ids = [], []
                for _, row in batch_df.iterrows():
                    img = Image.open(f"{img_dir}/{row['id']}").convert("RGB")
                    img = op(img)
                    tensors.append(base_tf(img))
                    batch_ids.append(row["id"])
                if tensors:
                    batch = torch.stack(tensors).to(DEVICE)
                    feats = model(batch).cpu().numpy()
                    features.extend(feats)
                    ids.extend(batch_ids)
            farr = np.array(features)
            if feats_sum is None:
                feats_sum = farr.copy()
                ids_ref = ids
            else:
                feats_sum += farr
        dino_feats = feats_sum / len(tta_ops)
        return dino_feats, ids_ref
    else:
        features, labels, ids = [], [], []
        indices = list(range(0, len(df), BATCH))
        for start in tqdm(indices, desc=desc):
            batch_df = df.iloc[start:start + BATCH]
            tensors, batch_ids, batch_labels = [], [], []
            for _, row in batch_df.iterrows():
                img = Image.open(f"{img_dir}/{row['id']}").convert("RGB")
                tensors.append(base_tf(img))
                batch_ids.append(row["id"])
                batch_labels.append(int(row["label"]))
            if tensors:
                batch = torch.stack(tensors).to(DEVICE)
                feats = model(batch).cpu().numpy()
                features.extend(feats)
                ids.extend(batch_ids)
                labels.extend(batch_labels)
        return np.array(features), np.array(labels), ids


# ================================================================
# 3. Main Pipeline
# ================================================================
def main():
    # Data paths
    train_dir = DATA_ROOT / "train_images" / "train_images"
    test_dir = DATA_ROOT / "test_images" / "test_images"
    if not train_dir.is_dir(): train_dir = DATA_ROOT / "train_images"
    if not test_dir.is_dir(): test_dir = DATA_ROOT / "test_images"

    train_df = pd.read_csv(DATA_ROOT / "train_labels.csv")
    test_df = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # ---- Step 1: Extract DINOv3-L features (cached) ----
    dino_cache = OUTPUT_BASE / "dino_features.npz"
    if dino_cache.exists():
        print("\nLoading cached DINO features...")
        data = np.load(dino_cache)
        train_dino = data["train_dino"]
        train_y = data["train_y"]
        test_dino = data["test_dino"]
        test_ids_arr = data["test_ids"]
        test_ids = test_ids_arr.tolist()
    else:
        print("\nLoading DINOv3-L...")
        model = load_dinov3()
        base_tf = build_dino_transform()

        print("Extracting train DINO features (no TTA)...")
        t0 = time.time()
        train_dino, train_y, _ = extract_dino_features(model, train_df, train_dir, base_tf, "Train DINO")
        print(f"  Train DINO: {train_dino.shape} ({time.time()-t0:.0f}s)")

        print("Extracting test DINO features (3-view TTA)...")
        t0 = time.time()
        test_dino, test_ids = extract_dino_features(model, test_df, test_dir, base_tf, "Test DINO", is_test=True)
        print(f"  Test DINO: {test_dino.shape} ({time.time()-t0:.0f}s)")

        np.savez_compressed(dino_cache,
            train_dino=train_dino, train_y=train_y,
            test_dino=test_dino, test_ids=np.array(test_ids))
        del model; torch.cuda.empty_cache()

    # ---- Step 2: Extract 21-dim texture features (cached) ----
    tex_cache = OUTPUT_BASE / "texture_features.npz"
    if tex_cache.exists():
        print("\nLoading cached texture features...")
        data = np.load(tex_cache)
        train_tex = data["train_tex"]
        test_tex = data["test_tex"]
    else:
        print("\nExtracting 21-dim texture features...")
        t0 = time.time()
        train_tex, _ = extract_texture_batch(train_df, train_dir, "Train Tex")
        test_tex, _ = extract_texture_batch(test_df, test_dir, "Test Tex")
        print(f"  Train tex: {train_tex.shape}, Test tex: {test_tex.shape} ({time.time()-t0:.0f}s)")
        np.savez_compressed(tex_cache, train_tex=train_tex, test_tex=test_tex)

    # ---- Step 3: Feature Fusion per reference pipeline ----
    print("\n=== Feature Fusion ===")
    # 3a. L2 normalize DINO features
    train_dino_l2 = normalize(train_dino, norm='l2', axis=1)
    test_dino_l2 = normalize(test_dino, norm='l2', axis=1)

    # 3b. Concatenate
    train_fused = np.concatenate([train_dino_l2, train_tex], axis=1)  # (N, 1024+21)
    test_fused = np.concatenate([test_dino_l2, test_tex], axis=1)
    print(f"Fused dim: {train_fused.shape[1]}")

    # 3c. SelectKBest to 512
    selector = SelectKBest(f_classif, k=512).fit(train_fused, train_y)
    train_fused = selector.transform(train_fused)
    test_fused = selector.transform(test_fused)
    print(f"After SelectKBest: {train_fused.shape[1]}")
    n_texture_selected = (selector.get_support()[-21:]).sum()
    print(f"  Texture features selected: {n_texture_selected}/21")

    # 3d. Final L2 normalization
    train_fused = normalize(train_fused, norm='l2', axis=1)
    test_fused = normalize(test_fused, norm='l2', axis=1)

    # ---- Step 4: SVM + kNN 5-fold CV ensemble ----
    print(f"\n=== SVM(RBF) + kNN(cosine) 5-fold CV ===")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    svm_test = np.zeros(len(test_fused))
    knn_test = np.zeros(len(test_fused))
    svm_accs, knn_accs = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_fused, train_y)):
        X_tr, y_tr = train_fused[tr_idx], train_y[tr_idx]
        X_val, y_val = train_fused[val_idx], train_y[val_idx]

        svm = make_pipeline(StandardScaler(),
            SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED+fold))
        svm.fit(X_tr, y_tr)
        svm_accs.append(svm.score(X_val, y_val))
        svm_test += svm.predict_proba(test_fused)[:, 1] / n_splits

        knn = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine')
        knn.fit(X_tr, y_tr)
        knn_accs.append(knn.score(X_val, y_val))
        knn_test += knn.predict_proba(test_fused)[:, 1] / n_splits

        print(f"  Fold {fold+1}: SVM={svm_accs[-1]:.4f}, kNN={knn_accs[-1]:.4f}")

    print(f"  SVM avg: {np.mean(svm_accs):.4f} ± {np.std(svm_accs):.4f}")
    print(f"  kNN avg: {np.mean(knn_accs):.4f} ± {np.std(knn_accs):.4f}")

    # ---- Step 5: Ensemble ----
    test_prob = (svm_test + knn_test) / 2.0
    clean_ids = [Path(p).name for p in test_ids]

    out = pd.DataFrame({"id": clean_ids, "prob": test_prob,
                         "prob_svm": svm_test, "prob_knn": knn_test})
    out = out.sort_values("prob", ascending=False).reset_index(drop=True)
    out.to_csv(OUTPUT_BASE / "test_prob.csv", index=False, encoding="utf-8-sig")
    print(f"\nTexture Fusion prob: mean={test_prob.mean():.4f}, "
          f"range=[{test_prob.min():.4f}, {test_prob.max():.4f}]")

    # Submissions
    tmpl = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    for th in [0.5, 0.55, 0.6]:
        id2label = dict(zip(clean_ids, test_prob >= th))
        sub = tmpl.copy(); sub["label"] = sub["id"].map(id2label).fillna(0).astype(int)
        sub.to_csv(OUTPUT_BASE / f"submission_th{int(th*100):02d}.csv", index=False)
        print(f"  th={th:.2f}: {int(sub['label'].sum())} positives")

    for k in [80, 82, 84, 85, 88, 90, 92, 95]:
        top = set(out.head(k)["id"].tolist())
        s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
        s.to_csv(OUTPUT_BASE / f"submission_top{k}.csv", index=False)
        print(f"  Top-{k}: {int(s['label'].sum())} positives")

    # ---- Step 6: Blend with ConvNeXtV2 ----
    conv_path = DATA_ROOT / "runs" / "v13_proper" / "convnextv2_base_frozen_head_v2" / "test_prob.csv"
    if conv_path.exists():
        print("\n=== Blend with ConvNeXtV2 ===")
        conv = pd.read_csv(conv_path)
        df = out[["id","prob"]].rename(columns={"prob":"prob_dino_tex"})
        df = df.merge(conv[["id","prob"]].rename(columns={"prob":"prob_conv"}), on="id")
        df["prob_v13"] = 0.80*df["prob_dino_tex"] + 0.15*df["prob_conv"]
        df["label"] = (df["prob_v13"] >= 0.5).astype(int)

        bd = OUTPUT_BASE / "blend"; bd.mkdir(exist_ok=True)
        df.to_csv(bd / "test_prob_blend.csv", index=False)
        for k in [80, 82, 84, 85, 88, 90, 92, 95]:
            top = set(df.sort_values("prob_v13", ascending=False).head(k)["id"].tolist())
            s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
            s.to_csv(bd / f"submission_blend_top{k}.csv", index=False)
            print(f"  Blend Top-{k}: {int(s['label'].sum())} positives")

    summary = {
        "method": "DINOv3-L + 21-dim texture + SelectKBest(512) + SVM/kNN ensemble",
        "svm_acc": float(np.mean(svm_accs)), "knn_acc": float(np.mean(knn_accs)),
        "texture_features_selected": int(n_texture_selected),
    }
    with open(OUTPUT_BASE / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(f"\nDone! Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
