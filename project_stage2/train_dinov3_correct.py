"""
DINOv3-L training — EXACT replication of teammate's approach.
Key differences from previous attempt:
1. MLP head (1024->512->2) with LayerNorm+GELU+Dropout, NOT single Linear
2. LR=4e-5, weight_decay=2e-2
3. Correct data augmentation (RandomPerspective, RandomAffine, strong RandomErasing)
4. F1-based model selection
5. Dynamic threshold from validation
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
import timm
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
from PIL import Image

DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "v13_proper" / "dinov3_correct"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Hyperparameters (EXACT from hhp_main_f1.py + dino.py)
# ============================================================
SEED = 35
IMG_SIZE = 512
EPOCHS = 20
BATCH_SIZE = 8
LR = 4e-5        # ← was 3e-4 before (7.5x too high!)
WD = 2e-2         # ← was 1e-2 before
LS = 0.1
PATIENCE = 8

torch.manual_seed(SEED); np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED); cudnn.benchmark = True
print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# Data paths
# ============================================================
train_dir = DATA_ROOT / "train_images" / "train_images"
test_dir = DATA_ROOT / "test_images" / "test_images"
if not train_dir.is_dir(): train_dir = DATA_ROOT / "train_images"
if not test_dir.is_dir(): test_dir = DATA_ROOT / "test_images"

train_df = pd.read_csv(DATA_ROOT / "train_labels.csv")
test_df = pd.read_csv(DATA_ROOT / "sample_submission.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")


# ============================================================
# Dataset
# ============================================================
class ClsDataset(Dataset):
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


# ============================================================
# Transforms (EXACT from data_loader/utils.py build_all_datasets)
# ============================================================
def build_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.6),  # ← NEW
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # ← replaces Rotation
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.0), value=0),  # ← p=0.7 was 0.25
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_tta_transforms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def m(extra): return transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(IMG_SIZE),
        *extra, transforms.ToTensor(), norm])
    return {
        "orig": m([]),
        "hflip": m([transforms.RandomHorizontalFlip(p=1.0)]),
        "vflip": m([transforms.RandomVerticalFlip(p=1.0)]),
        "hvflip": m([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)]),
    }


# ============================================================
# Model: DINOv3Classifier with MLP head (EXACT from dino.py)
# ============================================================
class DINOv3Classifier(nn.Module):
    def __init__(self, num_classes=2, freeze=True, img_size=512,
                 ckpt_path="/data/models/dinov3_large/model.safetensors"):
        super().__init__()
        self.freeze = freeze
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True, num_classes=0, img_size=img_size,
            pretrained_cfg_overlay={"file": ckpt_path},
        )
        feat_dim = self.backbone.num_features  # 1024
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad = False
            self.backbone.eval()
        # MLP head (NOT single Linear!)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        if self.freeze:
            with torch.no_grad(): feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        return self.head(feat)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze: self.backbone.eval()
        return self


# ============================================================
# Threshold search for F1 (EXACT from hhp_main_f1.py)
# ============================================================
def threshold_search(y_true, prob, start=0.05, end=0.95, step=0.005):
    thresholds = np.arange(start, end + 1e-9, step)
    best = {"threshold": 0.5, "f1": -1.0}
    for th in thresholds:
        pred = (prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(th), "f1": float(f1),
                    "accuracy": float(accuracy_score(y_true, pred)),
                    "precision": float(precision_score(y_true, pred, zero_division=0)),
                    "recall": float(recall_score(y_true, pred, zero_division=0))}
    return best


# ============================================================
# Split (stratified, same as build_all_datasets)
# ============================================================
labels = train_df["label"].values
rng = np.random.default_rng(SEED)
train_indices, val_indices = [], []
for c in range(2):
    ci = np.where(labels == c)[0]
    rng.shuffle(ci)
    nv = int(len(ci) * 0.1)
    val_indices.extend(ci[:nv])
    train_indices.extend(ci[nv:])
train_indices = np.array(train_indices); val_indices = np.array(val_indices)
rng.shuffle(train_indices); rng.shuffle(val_indices)

tr_df = train_df.iloc[train_indices].reset_index(drop=True)
val_df = train_df.iloc[val_indices].reset_index(drop=True)
print(f"Split: train={len(tr_df)}, val={len(val_df)}")

# ============================================================
# DataLoaders
# ============================================================
train_ds = ClsDataset(tr_df, train_dir, build_train_transform())
val_ds = ClsDataset(val_df, train_dir, build_val_transform())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ============================================================
# Model + Optimizer
# ============================================================
print("\nLoading DINOv3 with MLP head...")
model = DINOv3Classifier().to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable}")

criterion = nn.CrossEntropyLoss(label_smoothing=LS)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda")

# ============================================================
# Training (F1-based model selection)
# ============================================================
best_f1 = -1.0; best_epoch = 0; best_metrics = None
best_val_prob = None; best_val_true = None
no_improve = 0; rows = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_n, total_correct = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"E{epoch}/{EPOCHS}")
    for x, y in pbar:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(x); loss = criterion(logits, y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        bs = y.size(0); total_loss += loss.item() * bs; total_n += bs
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    scheduler.step()

    # Validation
    model.eval()
    all_probs, all_labels, vloss, vn = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x); l = criterion(logits, y)
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            bs = y.size(0); vloss += l.item() * bs; vn += bs
            all_probs.append(prob.cpu().numpy()); all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs); labels = np.concatenate(all_labels)
    pred_05 = (probs >= 0.5).astype(int)
    th_best = threshold_search(labels, probs)

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
        best_f1 = th_best["f1"]; best_epoch = epoch; best_metrics = th_best
        best_val_prob = probs; best_val_true = labels; no_improve = 0
        torch.save({"head": hs, "epoch": epoch, "metrics": th_best}, OUTPUT_BASE / "best_head.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}"); break

# ============================================================
# Load best head + get dynamic threshold
# ============================================================
ckpt = torch.load(OUTPUT_BASE / "best_head.pth", map_location=DEVICE)
model.head.load_state_dict(ckpt["head"])
best_threshold = float(ckpt["metrics"]["threshold"])
model.eval()
print(f"\nBest epoch: {best_epoch}, best_val_f1: {best_f1:.4f}, threshold: {best_threshold:.4f}")

# ============================================================
# 4-view TTA inference
# ============================================================
print("\n4-view TTA inference...")
tta_tfs = build_tta_transforms()
view_probs = {}; ids_ref = None
for vn, tfm in tta_tfs.items():
    ds = ClsDataset(test_df, test_dir, tfm, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    ids, pl = [], []
    with torch.no_grad():
        for x, img_ids in tqdm(loader, desc=f"TTA {vn}"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"): logits = model(x)
            prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            ids.extend(list(img_ids)); pl.append(prob)
    pa = np.concatenate(pl)
    if ids_ref is None: ids_ref = ids
    assert ids_ref == ids, f"ID mismatch at {vn}"
    view_probs[vn] = pa
mp = np.mean(list(view_probs.values()), axis=0)

# ============================================================
# Save
# ============================================================
out = pd.DataFrame({"id": ids_ref, "prob": mp})
for k, v in view_probs.items(): out[f"prob_{k}"] = v
out.to_csv(OUTPUT_BASE / "test_prob.csv", index=False)

sub = pd.DataFrame({"id": ids_ref, "label": (mp >= best_threshold).astype(int)})
sub.to_csv(OUTPUT_BASE / "submission.csv", index=False)

summary = {"best_epoch": best_epoch, "best_val_f1": best_f1,
           "best_threshold": best_threshold, "n_positives": int(sub["label"].sum())}
with open(OUTPUT_BASE / "summary.json", "w") as f: json.dump(summary, f, indent=2)
print(f"\nDINOv3 Correct: f1={best_f1:.4f} th={best_threshold:.4f} positives={int(sub['label'].sum())}")
print(f"Prob: mean={mp.mean():.4f}, range=[{mp.min():.4f}, {mp.max():.4f}]")

# ============================================================
# Blend with ConvNeXtV2 (fixed v2)
# ============================================================
print("\n=== Blend with ConvNeXtV2 ===")
conv_path = DATA_ROOT / "runs" / "v13_proper" / "convnextv2_base_frozen_head_v2" / "test_prob.csv"
if conv_path.exists():
    conv = pd.read_csv(conv_path)
    df = out[["id", "prob"]].rename(columns={"prob": "prob_dino"})
    df = df.merge(conv[["id", "prob"]].rename(columns={"prob": "prob_conv"}), on="id")
    df["prob_v13"] = 0.80 * df["prob_dino"] + 0.15 * df["prob_conv"]
    df["label"] = (df["prob_v13"] >= 0.5).astype(int)

    bd = OUTPUT_BASE / "blend"
    bd.mkdir(exist_ok=True)
    df.to_csv(bd / "test_prob_blend.csv", index=False)

    tmpl = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    for label_name in ["label"]:
        sub = tmpl.copy()
        sub["label"] = sub["id"].map(df.set_index("id")[label_name]).fillna(0).astype(int)
        sub.to_csv(bd / f"submission_blend_threshold05.csv", index=False)
        print(f"Blend threshold 0.5: {int(sub['label'].sum())} positives")

    for k in [80, 82, 85, 88, 90, 92, 95]:
        top = set(df.sort_values("prob_v13", ascending=False).head(k)["id"].tolist())
        s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
        s.to_csv(bd / f"submission_blend_top{k}.csv", index=False)
        print(f"  Top-{k}: {int(s['label'].sum())} positives")
else:
    print(f"ConvNeXtV2 path not found: {conv_path}")

print("\nDone!")
