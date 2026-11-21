"""
三模型概率加权融合
DINOv3-L (80%) + SwinV2-L (15%) + ConvNeXt-L (10%)
每个模型独立提取特征 → 独立SVM/kNN → 概率加权融合 → top-90
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
import timm
from safetensors.torch import load_file
from PIL import Image, ImageOps
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import hflip, vflip
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

DATA_ROOT = "/data/数据科学实践project"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images", "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images", "test_images")
TEMPLATE_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "runs", f"triple_blend_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

print(f"DEVICE: {DEVICE}")
print(f"OUTPUT: {OUTPUT_DIR}")

# ==========================================
# Model configs with weights
# ==========================================
MODELS = {
    'dinov3': {
        'name': 'vit_large_patch16_dinov3',
        'path': '/data/models/dinov3_large/model.safetensors',
        'img_size': 224, 'resize': 256, 'crop': 224,
        'weight': 0.80,
    },
    'swinv2': {
        'name': 'swinv2_large_window12to24_192to384',
        'path': '/data/models/swinv2_large/model.safetensors',
        'img_size': 384, 'resize': 384, 'crop': 384,
        'weight': 0.10,
    },
    'convnext': {
        'name': 'convnext_large',
        'path': '/data/models/convnext_large/model.safetensors',
        'img_size': 224, 'resize': 256, 'crop': 224,
        'weight': 0.10,
    },
}


def load_data():
    """Load train/test dataframes with paths"""
    def resolve_dir(split):
        candidates = [
            os.path.join(DATA_ROOT, f"{split}_images", f"{split}_images"),
            os.path.join(DATA_ROOT, f"{split}_images"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        raise FileNotFoundError(f"找不到 {split}_images")

    train_df = pd.read_csv(TRAIN_CSV)
    train_dir = resolve_dir("train")
    train_df["img_path"] = train_df["id"].apply(lambda x: os.path.join(train_dir, x))
    train_df["label"] = train_df["label"].astype(int)

    test_df = pd.read_csv(TEMPLATE_CSV)
    test_dir = resolve_dir("test")
    test_df["img_path"] = test_df["id"].apply(lambda x: os.path.join(test_dir, x))

    return train_df, test_df


def load_model(cfg):
    print(f"  Loading {cfg['name']}...", end="", flush=True)
    model = timm.create_model(cfg['name'], pretrained=False, num_classes=0)
    state = load_file(cfg['path'])
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f" {n:.0f}M params")
    return model


def build_transform(cfg, flip=False):
    t_list = [
        transforms.Resize(cfg['resize'], interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg['crop']),
    ]
    if flip:
        t_list.insert(0, transforms.RandomHorizontalFlip(p=1.0))
    t_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(t_list)


@torch.no_grad()
def extract_features(model, df, transform_fn, desc=""):
    """Single-pass feature extraction"""
    feats, labels = [], []
    has_label = "label" in df.columns
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            img = Image.open(row["img_path"])
            img = ImageOps.exif_transpose(img).convert("RGB")
            tensor = transform_fn(img).unsqueeze(0).to(DEVICE)
            feat = model(tensor).cpu().numpy().flatten()
            feats.append(feat)
            if has_label:
                labels.append(int(row["label"]))
        except Exception:
            feats.append(np.zeros(1024))
            if has_label:
                labels.append(0)
    X = np.array(feats)
    if has_label:
        return X, np.array(labels)
    return X


@torch.no_grad()
def extract_features_tta(model, df, transform_fn, desc=""):
    """3-view TTA feature extraction"""
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            img = Image.open(row["img_path"])
            img = ImageOps.exif_transpose(img).convert("RGB")
            t_orig = transform_fn(img).unsqueeze(0).to(DEVICE)
            t_h = transform_fn(hflip(img)).unsqueeze(0).to(DEVICE)
            t_v = transform_fn(vflip(img)).unsqueeze(0).to(DEVICE)
            f = (model(t_orig) + model(t_h) + model(t_v)).cpu().numpy().flatten() / 3.0
            feats.append(f)
        except Exception:
            feats.append(np.zeros(1024))
    return np.array(feats)


def train_single_model(X_train, y_train, X_test, name):
    """Train SVM+kNN for one model, return test probabilities"""
    print(f"\n  [{name}] Training SVM+kNN on {X_train.shape[1]}d...")

    # Feature selection
    selector = SelectKBest(f_classif, k=min(512, X_train.shape[1]))
    X_tr_s = selector.fit_transform(X_train, y_train)
    X_te_s = selector.transform(X_test)
    X_tr_s = normalize(X_tr_s, norm='l2', axis=1)
    X_te_s = normalize(X_te_s, norm='l2', axis=1)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    svm_probs = np.zeros(len(X_test))
    knn_probs = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_s, y_train)):
        X_tr, y_tr = X_tr_s[tr_idx], y_train[tr_idx]
        X_val, y_val = X_tr_s[val_idx], y_train[val_idx]

        svm = make_pipeline(StandardScaler(),
            SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=SEED+fold))
        svm.fit(X_tr, y_tr)
        svm_probs += svm.predict_proba(X_te_s)[:, 1] / n_splits
        print(f"    Fold {fold+1}: SVM={svm.score(X_val, y_val):.4f}", end="")

        knn = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine')
        knn.fit(X_tr, y_tr)
        knn_probs += knn.predict_proba(X_te_s)[:, 1] / n_splits
        print(f"  kNN={knn.score(X_val, y_val):.4f}")

    # Ensemble within model
    return 0.7 * svm_probs + 0.3 * knn_probs


def main():
    train_df, test_df = load_data()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    all_probs = {}

    for key, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {key.upper()} (weight={cfg['weight']})")
        print(f"{'='*60}")

        model = load_model(cfg)
        train_tf = build_transform(cfg)
        test_tf = build_transform(cfg)

        # Extract features
        t0 = time.time()
        X_train, y_train = extract_features(model, train_df, train_tf, desc=f"{key} train")
        X_test = extract_features_tta(model, test_df, test_tf, desc=f"{key} test TTA")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape} ({time.time()-t0:.0f}s)")

        # Train and predict
        probs = train_single_model(X_train, y_train, X_test, key)
        all_probs[key] = probs

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Weighted blend
    print(f"\n{'='*60}")
    print("Weighted Probability Blend")
    print(f"{'='*60}")

    blend_probs = np.zeros(len(test_df))
    for key, cfg in MODELS.items():
        blend_probs += cfg['weight'] * all_probs[key]
        print(f"  {key}: weight={cfg['weight']}, prob range=[{all_probs[key].min():.4f}, {all_probs[key].max():.4f}]")

    print(f"  Blend: prob range=[{blend_probs.min():.4f}, {blend_probs.max():.4f}]")

    # Generate submission
    sub_df = pd.DataFrame({
        'id': test_df['id'].tolist(),
        'prob': blend_probs,
    }).sort_values('prob', ascending=False).reset_index(drop=True)

    sub_df.to_csv(os.path.join(OUTPUT_DIR, 'blend_probs.csv'), index=False)

    sub_df['label'] = 0
    sub_df.loc[:89, 'label'] = 1  # top-90

    template = pd.read_csv(TEMPLATE_CSV)
    template['label'] = template['id'].map(sub_df.set_index('id')['label']).fillna(0).astype(int)
    template.to_csv(os.path.join(OUTPUT_DIR, 'submission_blend_top90.csv'), index=False)

    # Also save individual model submissions for comparison
    for key in MODELS:
        single_df = pd.DataFrame({
            'id': test_df['id'].tolist(),
            'prob': all_probs[key],
        }).sort_values('prob', ascending=False).reset_index(drop=True)
        single_df['label'] = 0
        single_df.loc[:89, 'label'] = 1
        single_sub = template.copy()
        single_sub['label'] = single_sub['id'].map(single_df.set_index('id')['label']).fillna(0).astype(int)
        single_sub.to_csv(os.path.join(OUTPUT_DIR, f'submission_{key}_top90.csv'), index=False)

    print(f"\nTop-90 submissions saved to {OUTPUT_DIR}")
    for key in MODELS:
        f = os.path.join(OUTPUT_DIR, f'submission_{key}_top90.csv')
        s = pd.read_csv(f)
        print(f"  {key}: {s['label'].sum()} positives")

    blend_file = os.path.join(OUTPUT_DIR, 'submission_blend_top90.csv')
    blend_sub = pd.read_csv(blend_file)
    print(f"  blend: {blend_sub['label'].sum()} positives")

    with open(os.path.join(OUTPUT_DIR, 'blend_config.json'), 'w') as f:
        json.dump({k: {'weight': v['weight']} for k, v in MODELS.items()}, f, indent=2)

    print("\n完成！")


if __name__ == "__main__":
    main()
