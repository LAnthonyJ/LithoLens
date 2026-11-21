"""提取纹理特征概率，用于三方融合"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = Path("/data/数据科学实践project")

def _resolve_image_dir(root, split):
    candidates = [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]
    for c in candidates:
        if c.is_dir(): return c
    raise FileNotFoundError(f"找不到 {split}")


def compute_texture_features(image):
    import cv2
    image = image.resize((256, 256), Image.LANCZOS)
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    features = []
    for ws in [7, 15, 31]:
        mean = cv2.blur(gray, (ws, ws))
        mean_sq = cv2.blur(gray ** 2, (ws, ws))
        var = np.maximum(mean_sq - mean ** 2, 0)
        std = np.sqrt(var)
        features.extend([float(np.mean(std)), float(np.std(std)),
                         float(np.percentile(std, 90)), float(np.percentile(std, 50)), float(np.max(std))])
    features.append(float(np.std(gray)))
    features.append(float(np.max(gray) - np.min(gray)))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    features.append(float(np.mean(grad_mag)))
    features.append(float(np.std(grad_mag)))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    ch, cw = h // 2, w // 2
    radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask_high = ((y - ch) ** 2 + (x - cw) ** 2) > radius ** 2
    features.append(float((magnitude[mask_high] ** 2).sum() / ((magnitude ** 2).sum() + 1e-8)))
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    features.append(float(-np.sum(hist * np.log2(hist))))
    return np.array(features, dtype=np.float64)


def extract_all(image_paths, desc="纹理"):
    feats = []
    for path in tqdm(image_paths, desc=desc):
        try:
            img = Image.open(path); ImageOps.exif_transpose(img).convert("RGB")
            feats.append(compute_texture_features(img))
        except Exception:
            feats.append(np.zeros(21, dtype=np.float64))
    return np.nan_to_num(np.stack(feats), nan=0.0, posinf=0.0, neginf=0.0)


# Load data
train_dir = _resolve_image_dir(DATA_ROOT, "train")
test_dir = _resolve_image_dir(DATA_ROOT, "test")
df_train = pd.read_csv(DATA_ROOT / "train_labels.csv")
df_train["img_path"] = df_train["id"].apply(lambda x: str(train_dir / x))
df_train = df_train[df_train["img_path"].apply(lambda p: Path(p).exists())].copy()
df_train["label"] = df_train["label"].astype(int)
df_test = pd.read_csv(DATA_ROOT / "sample_submission.csv")[["id"]]
df_test["img_path"] = df_test["id"].apply(lambda x: str(test_dir / x))
df_test = df_test[df_test["img_path"].apply(lambda p: Path(p).exists())].copy()

# Extract
print("提取训练集纹理特征...")
train_X = extract_all(df_train["img_path"].tolist(), "训练纹理")
train_y = df_train["label"].values
print("提取测试集纹理特征...")
test_X = extract_all(df_test["img_path"].tolist(), "测试纹理")

# Train
idx_tr, _ = train_test_split(np.arange(len(train_X)), test_size=0.2, stratify=train_y, random_state=42)
X_tr, y_tr = train_X[idx_tr], train_y[idx_tr]

scaler = RobustScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
test_X_s = scaler.transform(test_X)

clf = make_pipeline(RobustScaler(), LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced', random_state=42))
scores = cross_val_score(clf, X_tr_s, y_tr, cv=5, scoring='f1')
print(f"纹理特征 5-fold F1: {scores.mean():.4f}+/-{scores.std():.4f}")

clf.fit(X_tr_s, y_tr)
test_prob = clf.predict_proba(test_X_s)[:, 1]

clean_ids = [Path(p).name for p in df_test["img_path"]]
prob_df = pd.DataFrame({'id': clean_ids, 'prob_tex': test_prob})
prob_df.to_csv(DATA_ROOT / 'texture_probs.csv', index=False)
print(f"纹理概率: min={test_prob.min():.4f} max={test_prob.max():.4f} mean={test_prob.mean():.4f}")
print(f"Saved: {DATA_ROOT}/texture_probs.csv")
