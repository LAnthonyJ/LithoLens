"""
MLP head training + 21-dim handcrafted texture features (no SVM/kNN, no SelectKBest).

Architecture: DINOv3-L(frozen)→1024 + Texture(21) = 1045 → MLP(1045→512→2)
Dataset: original + Dataset/ folder + pseudo-labels (full from top-84)
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.backends.cudnn as cudnn, timm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from pathlib import Path
from PIL import Image, ImageOps
import cv2

DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "v13_proper" / "texture_mlp"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
TEXTURE_DIM = 21
DINO_DIM = 1024
TOTAL_DIM = DINO_DIM + TEXTURE_DIM  # 1045

SEED = 35; IMG_SIZE = 512; EPOCHS = 20; BATCH = 8
LR = 4e-5; WD = 2e-2; LS = 0.1; PATIENCE = 8

torch.manual_seed(SEED); np.random.seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED); cudnn.benchmark = True
print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 21-dim Texture Feature Extractor (from texture_feature_reference.py)
# ============================================================
def compute_texture_features(image):
    image = image.resize((256, 256), Image.LANCZOS)
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    feats = []
    for ws in [7, 15, 31]:
        mean = cv2.blur(gray, (ws, ws))
        mean_sq = cv2.blur(gray ** 2, (ws, ws))
        var = np.maximum(mean_sq - mean ** 2, 0)
        std = np.sqrt(var)
        feats.extend([float(np.mean(std)), float(np.std(std)),
                      float(np.percentile(std, 90)), float(np.percentile(std, 50)),
                      float(np.max(std))])
    feats.append(float(np.std(gray)))
    feats.append(float(np.max(gray) - np.min(gray)))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(gx**2 + gy**2)
    feats.extend([float(np.mean(gm)), float(np.std(gm))])
    f = np.fft.fft2(gray); fshift = np.fft.fftshift(f)
    mag = np.abs(fshift); h, w = mag.shape; ch, cw = h//2, w//2
    r = min(h,w)//4
    y, x = np.ogrid[:h, :w]
    mask = (y-ch)**2 + (x-cw)**2 > r**2
    feats.append(float((mag[mask]**2).sum() / ((mag**2).sum() + 1e-8)))
    hist, _ = np.histogram(gray, bins=32, range=(0,256), density=True)
    hist = hist[hist > 0]
    feats.append(float(-np.sum(hist * np.log2(hist))))
    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Dataset with pre-computed texture features
# ============================================================
class TexDataset(Dataset):
    def __init__(self, df, texture_feats, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tex = texture_feats.astype(np.float32)
        self.transform = transform; self.is_test = is_test
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        tex = torch.from_numpy(self.tex[idx])
        if self.transform: img = self.transform(img)
        if self.is_test: return img, tex, row["id"]
        return img, tex, int(row["label"])


# ============================================================
# Transforms
# ============================================================
def build_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.6),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(), transforms.RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.0), value=0),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def build_val_transform():
    return transforms.Compose([transforms.Resize(512), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def build_tta_transforms():
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    def m(extra): return transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), *extra, transforms.ToTensor(), norm])
    return {"orig": m([]), "hflip": m([transforms.RandomHorizontalFlip(p=1.0)]),
            "vflip": m([transforms.RandomVerticalFlip(p=1.0)]),
            "hvflip": m([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)])}


# ============================================================
# Model: DINOv3 + Texture MLP
# ============================================================
class DINOv3TexClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.freeze = True
        self.backbone = timm.create_model("vit_large_patch16_dinov3.lvd1689m",
            pretrained=True, num_classes=0, img_size=512,
            pretrained_cfg_overlay={"file": "/data/models/dinov3_large/model.safetensors"})
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        self.head = nn.Sequential(nn.LayerNorm(TOTAL_DIM), nn.Linear(TOTAL_DIM, 512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 2))
    def forward(self, x, tex):
        with torch.no_grad(): feat = self.backbone(x)
        return self.head(torch.cat([feat, tex], dim=1))
    def train(self, mode=True):
        super().train(mode)
        if self.freeze: self.backbone.eval()
        return self


# ============================================================
# Threshold search
# ============================================================
def threshold_search(y_true, prob, start=0.05, end=0.95, step=0.005):
    best = {"threshold": 0.5, "f1": -1.0}
    for th in np.arange(start, end + 1e-9, step):
        pred = (prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(th), "f1": float(f1),
                    "accuracy": float(accuracy_score(y_true, pred)),
                    "precision": float(precision_score(y_true, pred, zero_division=0)),
                    "recall": float(recall_score(y_true, pred, zero_division=0))}
    return best


# ============================================================
# Build full dataset
# ============================================================
train_dir = DATA_ROOT / "train_images" / "train_images"
if not train_dir.is_dir(): train_dir = DATA_ROOT / "train_images"
test_dir = DATA_ROOT / "test_images" / "test_images"
if not test_dir.is_dir(): test_dir = DATA_ROOT / "test_images"

# Original
train_df = pd.read_csv(DATA_ROOT / "train_labels.csv")
train_df["img_path"] = train_df["id"].apply(lambda x: str(train_dir / x))
train_df["source"] = "original"

# Dataset augmentation
DATASET_DIR = DATA_ROOT / "Dataset"
ds_entries = []
if DATASET_DIR.is_dir():
    for cat in sorted(DATASET_DIR.iterdir()):
        if not cat.is_dir(): continue
        if cat.name == "Meteorite":
            for f in cat.iterdir():
                if f.suffix.lower() in ('.jpg','.jpeg','.png'):
                    ds_entries.append({"id": f"ds/meteorite/{f.name}", "img_path": str(f), "label": 1, "source": "dataset"})
        else:
            for sc in cat.iterdir():
                if sc.is_dir():
                    for f in sc.iterdir():
                        if f.suffix.lower() in ('.jpg','.jpeg','.png'):
                            ds_entries.append({"id": f"ds/{cat.name}/{sc.name}/{f.name}", "img_path": str(f), "label": 0, "source": "dataset"})
ds_df = pd.DataFrame(ds_entries)

# Pseudo-labels: all test images from top-84 submission
pseudo_pos = {  # Top-84 positives from pseudo_self blend
    "000001.jpg","000005.jpg","000009.jpg","000012.jpg","000013.jpg","000014.jpg",
    "000015.jpg","000017.jpg","000019.jpg","000023.jpg","000027.jpg","000028.jpg",
    "000029.jpg","000030.jpg","000034.jpg","000038.jpg","000042.jpg","000047.jpg",
    "000048.jpg","000052.jpg","000055.jpg","000056.jpg","000058.jpg","000060.jpg",
    "000061.jpg","000063.jpg","000064.jpg","000067.jpg","000068.jpg","000076.jpg",
    "000077.jpg","000078.jpg","000080.jpg","000081.jpg","000085.jpg","000089.jpg",
    "000092.jpg","000093.jpg","000094.jpg","000098.jpg","000099.jpg","000102.jpg",
    "000103.jpg","000104.jpg","000105.jpg","000106.jpg","000108.jpg","000109.jpg",
    "000110.jpg","000111.jpg","000121.jpg","000124.jpg","000125.jpg","000126.jpg",
    "000128.jpg","000130.jpg","000133.jpg","000134.jpg","000139.jpg","000141.jpg",
    "000142.jpg","000144.jpg","000146.jpg","000147.jpg","000149.jpg","000154.jpg",
    "000155.jpg","000157.jpg","000158.jpg","000159.jpg","000161.jpg","000164.jpg",
    "000166.jpg","000169.jpg","000170.jpg","000172.jpg","000173.jpg","000175.jpg",
    "000181.jpg","000182.jpg","000184.jpg","000185.jpg","000188.jpg","000191.jpg",
}
all_ids = pd.read_csv(DATA_ROOT / "sample_submission.csv")["id"].tolist()
ps_entries = []
for tid in all_ids:
    p = test_dir / tid
    if p.exists():
        ps_entries.append({"id": f"ps/{tid}", "img_path": str(p),
                          "label": 1 if tid in pseudo_pos else 0, "source": "pseudo"})
ps_df = pd.DataFrame(ps_entries)

full_df = pd.concat([train_df, ds_df, ps_df], ignore_index=True)
full_df = full_df[full_df["img_path"].apply(lambda p: Path(p).exists())].copy()
print(f"Original: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
print(f"Dataset:  {len(ds_df)} ({ds_df['label'].value_counts().to_dict()})")
print(f"Pseudo:   {len(ps_df)} ({ps_df['label'].value_counts().to_dict()})")
print(f"Combined: {len(full_df)} ({full_df['label'].value_counts().to_dict()})")

test_df = pd.read_csv(DATA_ROOT / "sample_submission.csv")

# ============================================================
# Pre-compute texture features
# ============================================================
tex_cache = OUTPUT_BASE / "texture_cache.npz"
if tex_cache.exists():
    print("\nLoading cached texture features...")
    data = np.load(tex_cache)
    full_tex = data["full_tex"]
    test_tex = data["test_tex"]
else:
    print("\nPre-computing 21-dim texture features...")
    full_feats = []
    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Train tex"):
        try:
            img = ImageOps.exif_transpose(Image.open(row["img_path"])).convert("RGB")
            full_feats.append(compute_texture_features(img))
        except Exception:
            full_feats.append(np.zeros(TEXTURE_DIM, dtype=np.float32))
    full_tex = np.stack(full_feats); full_tex = np.nan_to_num(full_tex, nan=0.0, posinf=0.0, neginf=0.0)

    test_feats = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test tex"):
        try:
            p = test_dir / row["id"]
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            test_feats.append(compute_texture_features(img))
        except Exception:
            test_feats.append(np.zeros(TEXTURE_DIM, dtype=np.float32))
    test_tex = np.stack(test_feats); test_tex = np.nan_to_num(test_tex, nan=0.0, posinf=0.0, neginf=0.0)

    # RobustScaler
    scaler = RobustScaler().fit(full_tex)
    full_tex = scaler.transform(full_tex).astype(np.float32)
    test_tex = scaler.transform(test_tex).astype(np.float32)
    np.savez_compressed(tex_cache, full_tex=full_tex, test_tex=test_tex)
    print("Texture features cached.")

print(f"Train tex: {full_tex.shape}, Test tex: {test_tex.shape}")

# ============================================================
# Split
# ============================================================
labels = full_df["label"].values
rng = np.random.default_rng(SEED)
tr_idx, val_idx = [], []
for c in range(2):
    ci = np.where(labels == c)[0]; rng.shuffle(ci)
    nv = max(1, int(len(ci) * 0.1))
    val_idx.extend(ci[:nv]); tr_idx.extend(ci[nv:])
tr_idx = np.array(tr_idx); val_idx = np.array(val_idx)
rng.shuffle(tr_idx); rng.shuffle(val_idx)

tr_df = full_df.iloc[tr_idx].reset_index(drop=True)
val_df = full_df.iloc[val_idx].reset_index(drop=True)
tr_tex = full_tex[tr_idx]; val_tex = full_tex[val_idx]
print(f"Split: train={len(tr_df)}, val={len(val_df)}")

# ============================================================
# DataLoaders
# ============================================================
train_ds = TexDataset(tr_df, tr_tex, build_train_transform())
val_ds = TexDataset(val_df, val_tex, build_val_transform())
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

# ============================================================
# Train
# ============================================================
print(f"\nLoading DINOv3+Tex classifier (input={TOTAL_DIM}d)...")
model = DINOv3TexClassifier().to(DEVICE)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion = nn.CrossEntropyLoss(label_smoothing=LS)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda")

best_f1 = -1.0; best_epoch = 0; no_improve = 0; rows = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_n, total_correct = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"E{epoch}/{EPOCHS}")
    for x, tex, y in pbar:
        x, tex, y = x.to(DEVICE), tex.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"): logits = model(x, tex); loss = criterion(logits, y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        bs = y.size(0); total_loss += loss.item()*bs; total_n += bs
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    scheduler.step()

    model.eval()
    all_probs, all_labels, vloss, vn = [], [], 0.0, 0
    with torch.no_grad():
        for x, tex, y in val_loader:
            x, tex, y = x.to(DEVICE), tex.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast("cuda"): logits = model(x, tex); l = criterion(logits, y)
            prob = torch.softmax(logits.float(), dim=1)[:,1]
            bs = y.size(0); vloss += l.item()*bs; vn += bs
            all_probs.append(prob.cpu().numpy()); all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs); labels = np.concatenate(all_labels)
    th_best = threshold_search(labels, probs)
    pred_05 = (probs >= 0.5).astype(int)

    row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
           "train_loss": total_loss/max(total_n,1), "train_acc": total_correct/max(total_n,1),
           "val_loss": vloss/max(vn,1), "val_f1_05": f1_score(labels, pred_05, zero_division=0),
           "val_best_f1": th_best["f1"], "val_best_th": th_best["threshold"]}
    rows.append(row); pd.DataFrame(rows).to_csv(OUTPUT_BASE / "metrics.csv", index=False)
    print(f"  E{epoch}: train_loss={row['train_loss']:.4f} acc={row['train_acc']:.4f} | "
          f"val_f1@0.5={row['val_f1_05']:.4f} best_f1={th_best['f1']:.4f} th={th_best['threshold']:.3f}")

    hs = model.head.state_dict()
    torch.save({"head": hs, "epoch": epoch, "metrics": th_best}, OUTPUT_BASE / "last_head.pth")
    if th_best["f1"] > best_f1:
        best_f1 = th_best["f1"]; best_epoch = epoch; no_improve = 0
        torch.save({"head": hs, "epoch": epoch, "metrics": th_best}, OUTPUT_BASE / "best_head.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE: print(f"  Early stop at {epoch}"); break

# ============================================================
# Inference
# ============================================================
ckpt = torch.load(OUTPUT_BASE / "best_head.pth", map_location=DEVICE)
model.head.load_state_dict(ckpt["head"]); model.eval()
best_th = float(ckpt["metrics"]["threshold"])
print(f"\nBest: epoch={best_epoch}, f1={best_f1:.4f}, th={best_th:.4f}")

print("4-view TTA inference...")
tta_tfs = build_tta_transforms(); vp = {}; ids_ref = None
for vn, tfm in tta_tfs.items():
    ds = TexDataset(test_df.assign(img_path=test_df["id"].apply(lambda x: str(test_dir / x))),
                    test_tex, tfm, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    ids, pl = [], []
    with torch.no_grad():
        for x, tex, img_ids in tqdm(loader, desc=f"TTA {vn}"):
            x, tex = x.to(DEVICE), tex.to(DEVICE)
            with torch.amp.autocast("cuda"): logits = model(x, tex)
            prob = torch.softmax(logits.float(), dim=1)[:,1].cpu().numpy()
            ids.extend(list(img_ids)); pl.append(prob)
    pa = np.concatenate(pl)
    if ids_ref is None: ids_ref = ids
    vp[vn] = pa
mp = np.mean(list(vp.values()), axis=0)

out = pd.DataFrame({"id": ids_ref, "prob": mp})
for k,v in vp.items(): out[f"prob_{k}"] = v
out.to_csv(OUTPUT_BASE / "test_prob.csv", index=False)

tmpl = pd.read_csv(DATA_ROOT / "sample_submission.csv")
for th_name, th_val in [("dynamic", best_th), ("th05", 0.5)]:
    sub = tmpl.copy(); sub["label"] = sub["id"].map(dict(zip(ids_ref, mp >= th_val))).fillna(0).astype(int)
    sub.to_csv(OUTPUT_BASE / f"submission_{th_name}.csv", index=False)
    print(f"Submission {th_name} (th={th_val:.4f}): {int(sub['label'].sum())} positives")

for k in [80,82,84,85,88,90,92,95]:
    top = set(out.sort_values("prob", ascending=False).head(k)["id"].tolist())
    s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
    s.to_csv(OUTPUT_BASE / f"submission_top{k}.csv", index=False)
    print(f"  Top-{k}: {int(s['label'].sum())} positives")

# Blend
conv_path = DATA_ROOT / "runs" / "v13_proper" / "convnextv2_base_frozen_head_v2" / "test_prob.csv"
if conv_path.exists():
    print("\n=== Blend ===")
    conv = pd.read_csv(conv_path)
    df = out[["id","prob"]].rename(columns={"prob":"prob_dino_tex"})
    df = df.merge(conv[["id","prob"]].rename(columns={"prob":"prob_conv"}), on="id")
    df["prob_v13"] = 0.80*df["prob_dino_tex"] + 0.15*df["prob_conv"]
    df["label"] = (df["prob_v13"] >= 0.5).astype(int)
    bd = OUTPUT_BASE / "blend"; bd.mkdir(exist_ok=True)
    df.to_csv(bd / "test_prob_blend.csv", index=False)
    for k in [80,82,84,85,88,90,92,95]:
        top = set(df.sort_values("prob_v13", ascending=False).head(k)["id"].tolist())
        s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
        s.to_csv(bd / f"submission_blend_top{k}.csv", index=False)
        print(f"  Blend Top-{k}: {int(s['label'].sum())} positives")

summary = {"best_epoch": best_epoch, "best_val_f1": best_f1, "best_threshold": best_th,
           "train_samples": len(full_df), "texture_dim": TEXTURE_DIM}
with open(OUTPUT_BASE / "summary.json", "w") as f: json.dump(summary, f, indent=2)
print(f"\nDone! Output: {OUTPUT_BASE}")
