"""
================================================================================
手工纹理特征提取 — 独立参考模块
================================================================================
用途：从陨石/非陨石图像中提取 21 维数值纹理特征，作为 DINOv2 等语义特征
      的互补信号。专门量化"熔壳粗糙度"和"气印凹凸感"，解决黑色人造物
      （如长尾夹）被误判为陨石的问题。

设计原则：
  1. 提取前缩放到 256×256 —— 纹理统计量对分辨率不敏感，但速度提升 100+ 倍
  2. 所有特征都是标量数值，不修改 DINOv2 输入 —— 避免预训练分布偏移
  3. 多尺度覆盖 —— 微裂纹(7×7)、气印边缘(15×15)、大凹凸(31×31)

21维特征构成：
  维度  来源                      物理含义
  ────  ────────────────────────  ──────────────────
  1-5   窗口 7×7  局部标准差统计   细粒度纹理（微裂纹密度）
  6-10  窗口 15×15 局部标准差统计  中等纹理（小气印边缘）
  11-15 窗口 31×31 局部标准差统计  粗纹理（大气印凹凸）
  16    全局灰度标准差            整体亮度离散度
  17    全局对比度 (max-min)      最亮与最暗的差距
  18    Sobel梯度均值             全图平均边缘强度
  19    Sobel梯度标准差           边缘分布不均匀度
  20    FFT高频能量占比           表面细节量 ← 关键区分特征
  21    灰度直方图熵              纹理混乱度
================================================================================
"""

from pathlib import Path
import numpy as np


def compute_texture_features(image):
    """
    从一张 PIL RGB 图像中提取 21 维纹理特征向量。

    参数:
        image: PIL.Image (RGB 模式)

    返回:
        np.ndarray, shape=(21,), dtype=float64
        异常时返回全零向量 (len=21)

    设计决策:
        - resize(256,256) 而非原始分辨率:
          256×256 上一个 31×31 窗口覆盖图像面积的 ~1.5%，足以捕捉气印，
          而 FFT 在 256² 上比 2240² 快约 120 倍。
        - LANCZOS 插值降采样保留更多高频细节。
        - cv2.blur 两次而非双重循环:
          Var = E[X²] - (E[X])²，用积分图加速到毫秒级。
    """
    import cv2

    # ========================================================
    # 预处理
    # ========================================================
    image = image.resize((256, 256), Image.LANCZOS)
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)

    features = []

    # ========================================================
    # 组1: 多尺度局部方差统计量 (15维)
    # ========================================================
    # 核心恒等式: Var(X) = E[X²] - (E[X])²
    # cv2.blur 实现滑动窗口均值，等价于 E[·] 的离散近似
    #
    # 对每个窗口大小，提取 5 个统计量:
    #   mean(std)   → 平均纹理密度
    #   std(std)    → 纹理密度的空间变化程度
    #   p90(std)    → "最粗糙的 10% 区域"有多粗糙
    #   p50(std)    → 中位数纹理密度（更鲁棒的中心趋势）
    #   max(std)    → 最强纹理点（单点极值）
    for ws in [7, 15, 31]:
        mean = cv2.blur(gray, (ws, ws))           # 局部均值 E[X]
        mean_sq = cv2.blur(gray ** 2, (ws, ws))   # 局部平方均值 E[X²]
        var = np.maximum(mean_sq - mean ** 2, 0)   # 局部方差，clip 防止浮点负零
        std = np.sqrt(var)                          # 局部标准差

        features.extend([
            float(np.mean(std)),
            float(np.std(std)),
            float(np.percentile(std, 90)),
            float(np.percentile(std, 50)),
            float(np.max(std)),
        ])

    # ========================================================
    # 组2: 全局对比度 + 梯度能量 (4维)
    # ========================================================
    features.append(float(np.std(gray)))                       # 全局亮度离散度
    features.append(float(np.max(gray) - np.min(gray)))        # 全局对比度

    # Sobel 算子 (3×3核):
    #   Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  检测垂直边缘
    #   Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]  检测水平边缘
    #   gradient_magnitude = sqrt(Gx² + Gy²)
    # 陨石熔壳边缘密集 → 均值高；人造物光滑 → 均值低
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    features.append(float(np.mean(grad_mag)))
    features.append(float(np.std(grad_mag)))

    # ========================================================
    # 组3: 频域特征 — 高频能量占比 (1维)
    # ========================================================
    # fft2 → fftshift 将低频移到频谱中心
    # 高频区域 = 到中心距离 > min(h,w)/4 的环
    # 高频能量占比 = E_high / E_total
    # 陨石纹理丰富 → 高频占比高；人造物光滑 → 低频主导
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
    # 组4: 直方图熵 — 灰度分布混乱度 (1维)
    # ========================================================
    # H = -Σ p_i * log2(p_i)
    # 直方图分 32 个 bin，过滤空 bin 后计算熵
    # 陨石表面明暗变化复杂 → 分布均匀 → 熵高
    # 人造物亮度单一         → 分布集中 → 熵低
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    features.append(float(-np.sum(hist * np.log2(hist))))

    return np.array(features, dtype=np.float64)


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    from PIL import Image

    # 单张
    img = Image.open("some_meteorite.jpg").convert("RGB")
    feat = compute_texture_features(img)
    print(f"特征维度: {feat.shape}")     # (21,)
    print(f"各维度范围: {feat.min():.2f} ~ {feat.max():.2f}")

    # 批量提取的推荐封装（含异常处理和 NaN 清理）
    def extract_batch(image_paths, desc="纹理"):
        from tqdm import tqdm
        feats = []
        for path in tqdm(image_paths, desc=desc):
            try:
                img = Image.open(path)
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img).convert("RGB")
                feats.append(compute_texture_features(img))
            except Exception:
                # 异常时返回全零占位，不影响批量处理
                feats.append(np.zeros(21, dtype=np.float64))
        mat = np.stack(feats, axis=0)
        # 防御性 NaN/Inf 清除
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    # ========================================================================
    # 集成到分类管线的推荐做法
    # ========================================================================
    #
    # 1. 标准化: 纹理特征各维度量纲不同，必须 fit 训练集 RobustScaler 后 transform
    #    from sklearn.preprocessing import RobustScaler
    #    scaler = RobustScaler().fit(train_texture_feat)
    #    train_tex = scaler.transform(train_texture_feat)
    #    test_tex  = scaler.transform(test_texture_feat)
    #
    #    → 为什么 RobustScaler 而非 StandardScaler?
    #      纹理特征有离群点（极亮/极暗区域方差极大），RobustScaler 用
    #      median 和 IQR，不受极端值影响。
    #
    # 2. 拼接: 与 DINOv2/ConvNeXt 等语义特征的 L2 归一化结果拼接
    #    train_fused = np.concatenate([train_dino_l2, train_tex], axis=1)
    #
    # 3. 特征筛选: 拼接后维度膨胀，用 SelectKBest 剔除噪声维度
    #    from sklearn.feature_selection import SelectKBest, f_classif
    #    selector = SelectKBest(f_classif, k=512).fit(train_fused, train_y)
    #    train_fused = selector.transform(train_fused)
    #
    # 4. 最终归一化: 融合特征再做一次 L2
    #    from sklearn.preprocessing import normalize
    #    train_fused = normalize(train_fused, norm='l2', axis=1)
    #
    # 5. 分类器: SVM(RBF) + kNN(cosine) 均值融合
    #    final = (svm_prob + knn_prob) / 2
    #
    # ========================================================================
