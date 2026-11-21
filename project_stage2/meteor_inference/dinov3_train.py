"""
DINOv3-L feature extraction + Flip TTA + SVM/kNN
Batch processing for speed, TTA only on test set
"""
import os, sys, pickle, time, json
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from tqdm import tqdm, trange
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

DATA_ROOT = "/data/数据科学实践project"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images", "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images", "test_images")
TEMPLATE_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "meteor_output")
MODEL_PATH = "/data/models/dinov3_large"
IMG_SIZE = 384
TRAIN_BATCH = 32
TEST_BATCH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model():
    print("Loading DINOv3-L from local...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    model = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0)
    from safetensors.torch import load_file
    state = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


@torch.no_grad()
def extract_batch(model, df, img_dir, base_tf, desc, batch_size):
    """Batch feature extraction without TTA"""
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
                tensors.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
                batch_ids.append(row["id"])
                if has_label:
                    batch_labels.append(0)

        if tensors:
            batch = torch.stack(tensors).to(device)
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
def extract_batch_tta(model, df, img_dir, base_tf, desc, batch_size):
    """Batch feature extraction with Flip TTA (orig + hflip + vflip)"""
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
                all_tensors.extend([torch.zeros(3, IMG_SIZE, IMG_SIZE)] * 3)
                batch_ids.append(row["id"])

        if all_tensors:
            batch = torch.stack(all_tensors).to(device)
            feats = model(batch).cpu().numpy()
            for i in range(len(batch_ids)):
                f_orig, f_h, f_v = feats[i*3], feats[i*3+1], feats[i*3+2]
                features.append((f_orig + f_h + f_v) / 3.0)
            ids.extend(batch_ids)

    return np.array(features), ids


def train_classifiers(train_X, train_y):
    print(f"\nTraining on {train_X.shape[0]} samples, {train_X.shape[1]} dims")
    print(f"Labels: {np.bincount(train_y)}")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    svm_accs, knn_accs = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
        X_val, y_val = train_X[val_idx], train_y[val_idx]

        svm = make_pipeline(StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=42))
        svm.fit(X_tr, y_tr)
        svm_accs.append(svm.score(X_val, y_val))

        knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
        knn.fit(X_tr, y_tr)
        knn_accs.append(knn.score(X_val, y_val))

        print(f"Fold {fold+1}: SVM={svm_accs[-1]:.4f}, kNN={knn_accs[-1]:.4f}")

    print(f"\nSVM avg: {np.mean(svm_accs):.4f} +/- {np.std(svm_accs):.4f}")
    print(f"kNN avg: {np.mean(knn_accs):.4f} +/- {np.std(knn_accs):.4f}")

    final_svm = make_pipeline(StandardScaler(),
        SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=42))
    final_svm.fit(train_X, train_y)

    final_knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
    final_knn.fit(train_X, train_y)

    return final_svm, final_knn


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_tf = build_transforms()

    model = load_model()

    # Train features (no TTA for speed)
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"\nExtracting train features (batch, no TTA)...")
    t0 = time.time()
    train_X, train_y, _ = extract_batch(model, train_df, TRAIN_DIR, base_tf, "Train", TRAIN_BATCH)
    print(f"Done in {time.time()-t0:.0f}s. Shape: {train_X.shape}")

    svm, knn = train_classifiers(train_X, train_y)

    with open(os.path.join(OUTPUT_DIR, "dinov3_classifiers.pkl"), "wb") as f:
        pickle.dump({"svm": svm, "knn": knn}, f)

    # Test features (with TTA)
    test_df = pd.read_csv(TEMPLATE_CSV)
    print(f"\nExtracting test features (batch, with TTA)...")
    t0 = time.time()
    test_X, test_ids = extract_batch_tta(model, test_df, TEST_DIR, base_tf, "Test", TEST_BATCH)
    test_X = normalize(test_X, norm="l2", axis=1)
    print(f"Done in {time.time()-t0:.0f}s. Shape: {test_X.shape}")

    # Predict
    svm_probs = svm.predict_proba(test_X)[:, 1]
    knn_probs = knn.predict_proba(test_X)[:, 1]
    final_probs = 0.7 * svm_probs + 0.3 * knn_probs

    prob_df = pd.DataFrame({"id": test_ids, "prob": final_probs})
    prob_df = prob_df.sort_values("prob", ascending=False)
    prob_df.to_csv(os.path.join(OUTPUT_DIR, "dinov3_test_probs.csv"), index=False)

    template = pd.read_csv(TEMPLATE_CSV)
    id_to_prob = dict(zip(test_ids, final_probs))

    # 0.5 threshold
    sub_05 = template.copy()
    sub_05["label"] = sub_05["id"].map(lambda x: 1 if id_to_prob.get(x, 0) >= 0.5 else 0).astype(int)
    sub_05.to_csv(os.path.join(OUTPUT_DIR, "submission_dinov3_threshold05.csv"), index=False)
    n05 = int(sub_05["label"].sum())

    # Top-k variants
    for k in [85, 88, 90, 91, 92, 93, 95, 100, 105]:
        sub = template.copy()
        top_ids = set(prob_df.head(k)["id"].tolist())
        sub["label"] = sub["id"].map(lambda x: 1 if x in top_ids else 0).astype(int)
        sub.to_csv(os.path.join(OUTPUT_DIR, f"submission_dinov3_top{k}.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Features: {train_X.shape[1]} dims")
    print(f"Threshold 0.5: {n05} positives")
    for k in [85, 88, 90, 91, 92, 93, 95, 100, 105]:
        n = int(pd.read_csv(os.path.join(OUTPUT_DIR, f"submission_dinov3_top{k}.csv"))["label"].sum())
        print(f"Top-{k}: {n} positives")
    print(f"All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
