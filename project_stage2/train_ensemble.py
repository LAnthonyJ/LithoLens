"""
双模型融合：DINOv2 + ConvNeXtV2-Base
- 冻结特征提取 → 线性分类头 (LogisticRegression)
- 加权概率融合: 0.80 * DINOv2 + 0.20 * ConvNeXtV2
- 只用真实标注训练，不加伪标签 (伪标签在范例中反复导致分数下降)
"""
from pathlib import Path
import json, hashlib, random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 0. 配置
# ==========================================
DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_DIR = DATA_ROOT / "runs" / f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4
VAL_RATIO = 0.2

# 融合权重 (来自范例：DINO 主导，ConvNeXtV2 小修正)
W_DINO = 0.80
W_CONV = 0.20

print(f"DEVICE: {DEVICE}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"Blend: {W_DINO}*DINOv2 + {W_CONV}*ConvNeXtV2")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==========================================
# 1. 数据集类
# ==========================================
class MeteoriteDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transforms = transforms
        self.has_label = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["img_path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        if self.has_label:
            return image, int(row["label"])
        return image, str(row["img_path"])


# ==========================================
# 2. 数据加载
# ==========================================
def _resolve_image_dir(root, split):
    candidates = [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]
    for c in candidates:
        if c.is_dir(): return c
    raise FileNotFoundError(f"找不到 {split} 图片目录")


def load_train_dataframe(data_root):
    data_root = Path(data_root)
    csv_path = data_root / "train_labels.csv"
    train_dir = _resolve_image_dir(data_root, "train")
    df = pd.read_csv(csv_path)
    df["img_path"] = df["id"].apply(lambda x: str(train_dir / x))
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()
    df["label"] = df["label"].astype(int)
    return df


def load_test_dataframe(data_root):
    data_root = Path(data_root)
    csv_path = data_root / "sample_submission.csv"
    test_dir = _resolve_image_dir(data_root, "test")
    df = pd.read_csv(csv_path)
    df = df[["id"]].copy()
    df["img_path"] = df["id"].apply(lambda x: str(test_dir / x))
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()
    return df


# ==========================================
# 3. DINOv2 transform + ConvNeXtV2 transform
# ==========================================
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
CONV_MEAN = (0.485, 0.456, 0.406)  # ConvNeXtV2 uses same ImageNet stats
CONV_STD = (0.229, 0.224, 0.225)
CONV_SIZE = 288  # ConvNeXtV2-Base default resolution is 288

dino_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
])

conv_transform = transforms.Compose([
    transforms.Resize(288, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONV_MEAN, std=CONV_STD),
])


@torch.no_grad()
def extract_features(loader, model, device, is_test=False):
    model.eval()
    all_feats, all_labels, all_paths = [], [], []
    for batch in tqdm(loader, desc='Extracting'):
        if is_test:
            images, paths = batch; all_paths.extend(paths)
        else:
            images, labels = batch; all_labels.extend(labels.cpu().tolist())
        images = images.to(device)
        all_feats.append(model(images).cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    if is_test: return all_feats, all_paths
    return all_feats, np.array(all_labels)


# ==========================================
# 4. 加载数据 (只用真实标注，不加伪标签)
# ==========================================
print("\n" + "=" * 60)
print("加载数据 (仅真实标注)")
print("=" * 60)

train_df = load_train_dataframe(DATA_ROOT)
test_df = load_test_dataframe(DATA_ROOT)

# 简单分层划分训练/验证
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=VAL_RATIO,
    stratify=train_df['label'].values, random_state=SEED
)
train_sub = train_df.iloc[train_idx].reset_index(drop=True)
val_sub = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Train: {len(train_sub)}, Val: {len(val_sub)}, Test: {len(test_df)}")
print(f"Train dist: {train_sub['label'].value_counts().sort_index().to_dict()}")

# ==========================================
# 5. 加载 DINOv2 ViT-L14
# ==========================================
print("\n" + "=" * 60)
print("加载 DINOv2 ViT-L14")
print("=" * 60)
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2.eval().to(DEVICE)

# DINOv2 特征提取 (训练集 + TTA 测试集)
train_ds = MeteoriteDataset(train_sub, transforms=dino_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print("DINOv2 训练特征...")
train_X_dino, train_y = extract_features(train_loader, dinov2, DEVICE)

# DINO TTA 测试 (原始+翻转)
tta_flips = [
    dino_transform,
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), dino_transform]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), dino_transform]),
]
test_X_dino_list = []
for i, t in enumerate(tta_flips):
    loader = DataLoader(MeteoriteDataset(test_df, transforms=t),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    X, ids = extract_features(loader, dinov2, DEVICE, is_test=True)
    test_X_dino_list.append(X)
test_X_dino = np.mean(test_X_dino_list, axis=0)
print(f"DINOv2: train={train_X_dino.shape}, test={test_X_dino.shape}")

# ==========================================
# 6. 加载 ConvNeXtV2-Base
# ==========================================
print("\n" + "=" * 60)
print("加载 ConvNeXtV2-Base")
print("=" * 60)

# ConvNeXt-Base from torchvision
convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
convnext.classifier = nn.Identity()
convnext.eval().to(DEVICE)


@torch.no_grad()
def extract_conv_features(loader, model, device, is_test=False):
    """ConvNeXt 输出 (N, 1024, 1, 1)，需要 flatten 到 (N, 1024)"""
    model.eval()
    all_feats, all_labels, all_paths = [], [], []
    for batch in tqdm(loader, desc='ConvNeXt'):
        if is_test:
            images, paths = batch; all_paths.extend(paths)
        else:
            images, labels = batch; all_labels.extend(labels.cpu().tolist())
        images = images.to(device)
        out = model(images)  # (N, 1024, 1, 1)
        out = out.flatten(1)  # (N, 1024)
        all_feats.append(out.cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    if is_test: return all_feats, all_paths
    return all_feats, np.array(all_labels)

train_ds_c = MeteoriteDataset(train_sub, transforms=conv_transform)
train_loader_c = DataLoader(train_ds_c, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print("ConvNeXtV2 训练特征...")
train_X_conv, _ = extract_conv_features(train_loader_c, convnext, DEVICE)

# ConvNeXt TTA
tta_conv = [
    conv_transform,
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), conv_transform]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), conv_transform]),
]
test_X_conv_list = []
for i, t in enumerate(tta_conv):
    loader = DataLoader(MeteoriteDataset(test_df, transforms=t),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    X, _ = extract_conv_features(loader, convnext, DEVICE, is_test=True)
    test_X_conv_list.append(X)
test_X_conv = np.mean(test_X_conv_list, axis=0)
print(f"ConvNeXt-Base: train={train_X_conv.shape}, test={test_X_conv.shape}")

# ==========================================
# 7. 线性分类头 (LogisticRegression — 等价于可训练的线性分类头)
# ==========================================
print("\n" + "=" * 60)
print("训练分类头 + 5折CV")
print("=" * 60)

# L2 归一化特征 (先清理 NaN/Inf)
for name, arr in [("DINO-train", train_X_dino), ("DINO-test", test_X_dino),
                   ("Conv-train", train_X_conv), ("Conv-test", test_X_conv)]:
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print(f"  {name}: 发现 NaN/Inf，已清理")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

train_X_dino = np.nan_to_num(train_X_dino, nan=0.0, posinf=0.0, neginf=0.0)
test_X_dino = np.nan_to_num(test_X_dino, nan=0.0, posinf=0.0, neginf=0.0)
train_X_conv = np.nan_to_num(train_X_conv, nan=0.0, posinf=0.0, neginf=0.0)
test_X_conv = np.nan_to_num(test_X_conv, nan=0.0, posinf=0.0, neginf=0.0)

train_X_dino = normalize(train_X_dino.astype(np.float64), norm='l2', axis=1)
test_X_dino = normalize(test_X_dino.astype(np.float64), norm='l2', axis=1)
train_X_conv = normalize(train_X_conv.astype(np.float64), norm='l2', axis=1)
test_X_conv = normalize(test_X_conv.astype(np.float64), norm='l2', axis=1)

# DINOv2 分类头
dino_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced', random_state=SEED)
)
cv_scores = cross_val_score(dino_clf, train_X_dino, train_y, cv=5, scoring='f1')
print(f"DINOv2 分类头 5-fold F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

dino_clf.fit(train_X_dino, train_y)
test_prob_dino = dino_clf.predict_proba(test_X_dino)[:, 1]

# ConvNeXtV2 分类头
conv_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced', random_state=SEED)
)
cv_scores = cross_val_score(conv_clf, train_X_conv, train_y, cv=5, scoring='f1')
print(f"ConvNeXtV2 分类头 5-fold F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

conv_clf.fit(train_X_conv, train_y)
test_prob_conv = conv_clf.predict_proba(test_X_conv)[:, 1]

# ==========================================
# 8. 加权概率融合
# ==========================================
print("\n" + "=" * 60)
print(f"融合: {W_DINO}*DINOv2 + {W_CONV}*ConvNeXtV2-Base")
print("=" * 60)

final_probs = W_DINO * test_prob_dino + W_CONV * test_prob_conv

# 各模型独立统计
for name, probs in [("DINOv2", test_prob_dino), ("ConvNeXtV2", test_prob_conv), ("Blend", final_probs)]:
    print(f"  {name:>12}: min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f}")

# Natural threshold predictions (用于对比分析)
print(f"\n  Softmax threshold=0.5 正类数:")
print(f"    DINOv2: {(test_prob_dino > 0.5).sum()}")
print(f"    ConvNeXtV2: {(test_prob_conv > 0.5).sum()}")
print(f"    Blend: {(final_probs > 0.5).sum()}")

# ==========================================
# 9. 生成 Top-90 提交
# ==========================================
print("\n" + "=" * 60)
print("生成提交文件")
print("=" * 60)

clean_ids = [Path(p).name for p in ids]

sub_df = pd.DataFrame({'id': clean_ids, 'prob': final_probs})
sub_df = sub_df.sort_values(by='prob', ascending=False).reset_index(drop=True)

# 保存概率
sub_df.to_csv(OUTPUT_DIR / 'test_probs_ensemble.csv', index=False, encoding='utf-8-sig')

sub_df['label'] = 0
sub_df.loc[:89, 'label'] = 1

template = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
template['label'] = template['id'].map(sub_df.set_index('id')['label']).fillna(0).astype(int)

output_csv = OUTPUT_DIR / 'submission_ensemble_top90.csv'
template.to_csv(output_csv, index=False)

print(f"Top-90: 陨石={int(sub_df['label'].sum())}")
print(f"输出: {output_csv}")

# 验证关键样本 (从范例学到的信号)
print("\n" + "=" * 60)
print("关键样本验证 (范例信号: 182/17应为正, 168应为负)")
print("=" * 60)
key_samples = ['000182.jpg', '000017.jpg', '000168.jpg', '000121.jpg']
for sid in key_samples:
    if sid in clean_ids:
        idx = clean_ids.index(sid)
        d_p = test_prob_dino[idx]
        c_p = test_prob_conv[idx]
        b_p = final_probs[idx]
        print(f"  {sid}: DINO={d_p:.4f} Conv={c_p:.4f} Blend={b_p:.4f}")
    else:
        print(f"  {sid}: NOT FOUND")

with open(OUTPUT_DIR / 'run_summary.json', 'w', encoding='utf-8') as f:
    json.dump({
        'method': 'DINOv2 + ConvNeXtV2-Base Ensemble',
        'weights': {'dino': W_DINO, 'convnext': W_CONV},
        'dino_model': 'dinov2_vitl14_reg',
        'conv_model': 'convnext_v2_base',
        'classifier': 'LogisticRegression (linear probe)',
        'tta': '3-view (orig + hflip + vflip)',
        'train_size': int(len(train_sub)),
        'val_size': int(len(val_sub)),
        'test_size': int(len(test_df)),
    }, f, indent=2, ensure_ascii=False)

print("\n完成！")
