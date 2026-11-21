"""快速提取 ConvNeXt-Base 概率，用于与 TopBot 融合"""
from pathlib import Path
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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = Path("/data/数据科学实践project")
DEVICE = torch.device('cuda')
BATCH_SIZE = 32

def _resolve_image_dir(root, split):
    candidates = [root / f"{split}_images" / f"{split}_images", root / f"{split}_images"]
    for c in candidates:
        if c.is_dir(): return c
    raise FileNotFoundError(f"找不到 {split}")

class MeteoriteDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transforms = transforms
        self.has_label = "label" in self.df.columns
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["img_path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        if self.has_label:
            return image, int(row["label"])
        return image, str(row["img_path"])

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

CONV_MEAN = (0.485, 0.456, 0.406)
CONV_STD = (0.229, 0.224, 0.225)
conv_transform = transforms.Compose([
    transforms.Resize(288, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONV_MEAN, std=CONV_STD),
])

# Load ConvNeXt
print("Loading ConvNeXt-Base...")
convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
convnext.classifier = nn.Identity()
convnext.eval().to(DEVICE)

@torch.no_grad()
def extract_conv(loader, is_test=False):
    convnext.eval()
    feats, labs, paths = [], [], []
    for batch in tqdm(loader, desc='ConvNeXt'):
        images = batch[0].to(DEVICE)
        out = convnext(images).flatten(1)
        feats.append(out.cpu().numpy())
        if is_test:
            paths.extend(batch[1])
        else:
            labs.extend(batch[1])
    feats_np = np.concatenate(feats, axis=0)
    if is_test:
        return feats_np, paths
    return feats_np, np.array(labs)

# Train data
train_idx, _ = train_test_split(np.arange(len(df_train)), test_size=0.2, stratify=df_train['label'], random_state=42)
train_sub = df_train.iloc[train_idx].reset_index(drop=True)
train_loader = DataLoader(MeteoriteDataset(train_sub, transforms=conv_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
train_X_conv, train_y = extract_conv(train_loader, is_test=False)

# Test TTA
tta_list = [
    conv_transform,
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), conv_transform]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), conv_transform]),
]
test_X_list = []
test_ids = None
for t in tta_list:
    loader = DataLoader(MeteoriteDataset(df_test, transforms=t), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    X, ids_list = extract_conv(loader, is_test=True)
    test_X_list.append(X)
    if ids_list: test_ids = ids_list
test_X_conv = np.mean(test_X_list, axis=0)

train_X_conv = np.nan_to_num(train_X_conv, nan=0.0, posinf=0.0, neginf=0.0)
test_X_conv = np.nan_to_num(test_X_conv, nan=0.0, posinf=0.0, neginf=0.0)
train_X_conv = normalize(train_X_conv.astype(np.float64), norm='l2', axis=1)
test_X_conv = normalize(test_X_conv.astype(np.float64), norm='l2', axis=1)

# Train classifier
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced', random_state=42))
clf.fit(train_X_conv, train_y)
test_prob_conv = clf.predict_proba(test_X_conv)[:, 1]

clean_ids = [Path(p).name for p in test_ids]
prob_df = pd.DataFrame({'id': clean_ids, 'prob_conv': test_prob_conv})
prob_df.to_csv(DATA_ROOT / 'conv_probs.csv', index=False)
print(f"Saved: {DATA_ROOT}/conv_probs.csv")
print(f"ConvNeXt prob: min={test_prob_conv.min():.4f} max={test_prob_conv.max():.4f} mean={test_prob_conv.mean():.4f}")
