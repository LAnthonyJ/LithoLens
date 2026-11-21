"""
================================================================================
Handcrafted texture descriptors: standalone reference implementation
================================================================================
Extract 21 numerical descriptors from specimen images to complement semantic embeddings
      from backbones such as DINOv2. Surface roughness and depressions help distinguish dark objects
       (such as binder clips) from meteorites.

Design:
  1. Resize to 256 × 256 before extraction to reduce computation
  2. Compute scalar descriptors separately from the DINOv2 input
  3. Measure texture at three scales: fine cracks (7 × 7), intermediate edges (15 × 15) and larger depressions(31×31)

21 descriptor layout:
  Dimensions, source and interpretation
  ────  ────────────────────────  ──────────────────
  1-5   7 × 7 local standard deviation: fine texture (microcracks)
  6-10  15 × 15 local standard deviation: intermediate texture (depression edges)
  11-15 31 × 31 local standard deviation: coarse texture (large depressions)
  16    Global grayscale standard deviation: variation in brightness
  17    Global contrast (max minus min): brightness range
  18    Sobel mean gradient: average edge strength
  19    Sobel gradient standard deviation: spatial variation in edge strength
  20    FFT high-frequency energy fraction: fine surface detail
  21    Grayscale histogram entropy: intensity diversity
================================================================================
"""

from pathlib import Path
import numpy as np


def compute_texture_features(image):
    """
    Extract a 21-dimensional texture vector from a PIL RGB image.

    settings:
        image: PIL.Image (RGB mode)

    Returns:
        np.ndarray, shape=(21,), dtype=float64
        Return a zero vector on failure (len=21)

    Implementation notes:
        - resize(256,256) instead of the original resolution:
          256×256 a 31 × 31 window covers about 1.5% of the image and captures local depressions,
          and a 256-square FFT costs much less than one on a 2240-square image.
        - LANCZOS downsampling retains fine detail.
        - cv2.blur twice to avoid nested loops:
          Var = E[X²] - (E[X])², compute local moments efficiently.
    """
    import cv2

    # ========================================================
    # Preprocessing
    # ========================================================
    image = image.resize((256, 256), Image.LANCZOS)
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)

    features = []

    # ========================================================
    # Group 1: multiscale local variation (15 dimensions)
    # ========================================================
    # Identity: Var(X) = E[X²] - (E[X])²
    # cv2.blur computes a sliding-window mean, a discrete approximation to the expectation
    #
    # Extract five statistics per window size:
    #   mean(std)   → mean texture variation
    #   std(std)    → spatial spread of texture variation
    #   p90(std)    → "90th-percentile local variation
    #   p50(std)    → median texture variation (robust center)
    #   max(std)    → strongest local variation (maximum)
    for ws in [7, 15, 31]:
        mean = cv2.blur(gray, (ws, ws))           # local mean E[X]
        mean_sq = cv2.blur(gray ** 2, (ws, ws))   # local mean square E[X²]
        var = np.maximum(mean_sq - mean ** 2, 0)   # local variance; clamp roundoff below zero
        std = np.sqrt(var)                          # local standard deviation

        features.extend([
            float(np.mean(std)),
            float(np.std(std)),
            float(np.percentile(std, 90)),
            float(np.percentile(std, 50)),
            float(np.max(std)),
        ])

    # ========================================================
    # Group 2: contrast and gradients (four dimensions)
    # ========================================================
    features.append(float(np.std(gray)))                       # global brightness variation
    features.append(float(np.max(gray) - np.min(gray)))        # global contrast

    # Sobel operator (3 × 3 kernel):
    #   Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  detect vertical edges
    #   Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]  detect horizontal edges
    #   gradient_magnitude = sqrt(Gx² + Gy²)
    # Dense edges increase the mean gradient; smoother surfaces reduce it
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    features.append(float(np.mean(grad_mag)))
    features.append(float(np.std(grad_mag)))

    # ========================================================
    # Group 3: high-frequency energy fraction (one dimension)
    # ========================================================
    # fft2 → fftshift center the low frequencies
    # Use frequencies beyond a radius of min(h,w)/4
    # High-frequency energy fraction = E_high / E_total
    # Fine detail contributes more high-frequency energy than smooth surfaces
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    ch, cw = h // 2, w // 2
    radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask_high = ((y - ch) ** 2 + (x - cw) ** 2) > radius ** 2
    features.append(
        float((magnitude[mask_high] ** 2).sum() / ((magnitude ** 2).sum() + 1e-8))
    )

    # ========================================================
    # Group 4: grayscale histogram entropy (one dimension)
    # ========================================================
    # H = -Σ p_i * log2(p_i)
    # Use 32 bins and exclude empty bins before taking logarithms
    # Broad intensity distributions tend to have higher entropy
    # Concentrated intensity distributions have lower entropy
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    features.append(float(-np.sum(hist * np.log2(hist))))

    return np.array(features, dtype=np.float64)


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    from PIL import Image

    # Single image
    img = Image.open("some_meteorite.jpg").convert("RGB")
    feat = compute_texture_features(img)
    print(f"Feature shape: {feat.shape}")     # (21,)
    print(f"Feature range: {feat.min():.2f} ~ {feat.max():.2f}")

    # Batch extraction (including error handling and NaN cleanup)
    def extract_batch(image_paths, desc="texture"):
        from tqdm import tqdm
        feats = []
        for path in tqdm(image_paths, desc=desc):
            try:
                img = Image.open(path)
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img).convert("RGB")
                feats.append(compute_texture_features(img))
            except Exception:
                # Return a zero vector on failure so the batch can continue
                feats.append(np.zeros(21, dtype=np.float64))
        mat = np.stack(feats, axis=0)
        # Replace NaN and infinity
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    # ========================================================================
    # Using these features in a classifier
    # ========================================================================
    #
    # 1. Scale the texture dimensions: fit RobustScaler on training data before applying transform
    #    from sklearn.preprocessing import RobustScaler
    #    scaler = RobustScaler().fit(train_texture_feat)
    #    train_tex = scaler.transform(train_texture_feat)
    #    test_tex  = scaler.transform(test_texture_feat)
    #
    #    → Why use RobustScaler over StandardScaler?
    #      Extreme intensities can yield outlying variances; RobustScaler uses the
    #      median and IQR to limit their influence.
    #
    # 2. Concatenate with L2-normalized DINOv2 or ConvNeXt embeddings
    #    train_fused = np.concatenate([train_dino_l2, train_tex], axis=1)
    #
    # 3. Use SelectKBest to filter the expanded feature vector
    #    from sklearn.feature_selection import SelectKBest, f_classif
    #    selector = SelectKBest(f_classif, k=512).fit(train_fused, train_y)
    #    train_fused = selector.transform(train_fused)
    #
    # 4. Normalize the fused vector again using L2
    #    from sklearn.preprocessing import normalize
    #    train_fused = normalize(train_fused, norm='l2', axis=1)
    #
    # 5. Average RBF-SVM and cosine-kNN scores
    #    final = (svm_prob + knn_prob) / 2
    #
    # ========================================================================
