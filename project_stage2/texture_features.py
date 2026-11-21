"""
18-dim handcrafted texture features for meteorite identification.
Computed purely with cv2/numpy, no GPU needed.

1. Multi-scale local variance (15 dims):
   3 window sizes (7, 15, 31) x 5 statistics (mean, std, P50, P90, max)

2. Gradient energy (2 dims):
   Sobel gradient magnitude → mean, std

3. Frequency domain (1 dim):
   FFT high-frequency energy ratio

4. Histogram entropy (1 dim):
   Entropy of 32-bin grayscale histogram

Usage:
    extractor = TextureFeatureExtractor()
    features_18d = extractor(pil_image)  # returns np.array of shape (18,)
"""
import numpy as np
import cv2
from collections.abc import Iterable


class TextureFeatureExtractor:
    def __init__(self, window_sizes=(7, 15, 31), hist_bins=32):
        self.window_sizes = window_sizes
        self.hist_bins = hist_bins

    def __call__(self, image):
        """image: PIL Image or numpy array (H, W) or (H, W, 3)"""
        if hasattr(image, 'convert'):
            gray = np.array(image.convert('L'), dtype=np.float32)
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        features = []

        # 1. Multi-scale local variance (15 dims)
        for ws in self.window_sizes:
            kernel = np.ones((ws, ws), np.float32) / (ws * ws)
            local_mean = cv2.filter2D(gray, -1, kernel)
            local_sq_mean = cv2.filter2D(gray * gray, -1, kernel)
            local_var = local_sq_mean - local_mean * local_mean
            local_std = np.sqrt(np.maximum(local_var, 0))

            features.extend([
                float(np.mean(local_std)),
                float(np.std(local_std)),
                float(np.percentile(local_std, 50)),
                float(np.percentile(local_std, 90)),
                float(np.max(local_std)),
            ])

        # 2. Gradient energy (2 dims)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        features.append(float(np.mean(grad_mag)))
        features.append(float(np.std(grad_mag)))

        # 3. Frequency domain high-energy ratio (1 dim)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        total = np.sum(mag)
        if total > 0:
            h, w = mag.shape
            cy, cx = h // 2, w // 2
            radius = min(cy, cx) // 3
            y, x = np.ogrid[:h, :w]
            mask = ((y - cy) ** 2 + (x - cx) ** 2) >= radius ** 2
            high_freq_ratio = float(np.sum(mag[mask]) / total)
        else:
            high_freq_ratio = 0.0
        features.append(high_freq_ratio)

        # 4. Histogram entropy (1 dim)
        hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [self.hist_bins], [0, 256])
        hist = hist.flatten().astype(np.float64)
        hist = hist / (hist.sum() + 1e-9)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))
        features.append(entropy)

        arr = np.array(features, dtype=np.float32)
        # Replace inf/nan with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr


def precompute_texture_features(df, img_dir, extractor=None, desc="Texture"):
    """Pre-compute 18-dim texture features for all images in a dataframe.
    Returns (N, 18) numpy array and fitted RobustScaler."""
    from sklearn.preprocessing import RobustScaler
    from PIL import Image
    from tqdm import tqdm

    if extractor is None:
        extractor = TextureFeatureExtractor()

    features = []
    img_dir = str(img_dir)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        img_path = f"{img_dir}/{row['id']}"
        try:
            img = Image.open(img_path)
            feat = extractor(img)
            features.append(feat)
        except Exception:
            features.append(np.zeros(18, dtype=np.float32))

    X = np.stack(features, axis=0)  # (N, 18)
    scaler = RobustScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, scaler
