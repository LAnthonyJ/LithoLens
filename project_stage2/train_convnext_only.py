"""Train ConvNeXtV2-Base only (correct weight loading)"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
import timm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from PIL import Image

DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "v13_proper"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 35
BATCH = 8
EPOCHS = 20
LR = 3e-4
WD = 1e-2
LS = 0.1
PATIENCE = 8
IMG_SIZE = 384

torch.manual_seed(SEED); np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = True
print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")

train_dir = DATA_ROOT / "train_images" / "train_images"
test_dir = DATA_ROOT / "test_images" / "test_images"
if not train_dir.is_dir(): train_dir = DATA_ROOT / "train_images"
if not test_dir.is_dir(): test_dir = DATA_ROOT / "test_images"

train_df = pd.read_csv(DATA_ROOT / "train_labels.csv")
test_df = pd.read_csv(DATA_ROOT / "sample_submission.csv")
tr_df, val_df = train_test_split(train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"])
print(f"Train: {len(tr_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


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


def train_tf():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def tta_tfs():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def m(extra): return transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), *extra, transforms.ToTensor(), norm])
    return {"orig": m([]), "hflip": m([transforms.RandomHorizontalFlip(p=1.0)]),
            "vflip": m([transforms.RandomVerticalFlip(p=1.0)]),
            "hvflip": m([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)])}


print("\nLoading ConvNeXtV2-Base (correct method)...")
model = timm.create_model("convnextv2_base", pretrained=True, num_classes=2,
    pretrained_cfg_overlay={"file": "/data/models/convnextv2_base_22k_384_ema.pt"})
for p in model.parameters(): p.requires_grad = False
for p in model.get_classifier().parameters(): p.requires_grad = True
model = model.to(DEVICE)

p = next(model.parameters())
print(f"First param std={p.std():.6f} (should be ~0.04 for pretrained)")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable}")

if p.std() < 0.03:
    print("ERROR: Weights not loaded correctly! std too low.")
    raise RuntimeError("Weight loading failed")

train_ds = ClsDataset(tr_df, train_dir, train_tf())
val_ds = ClsDataset(val_df, train_dir, tta_tfs()["orig"])
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss(label_smoothing=LS)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda")

exp_dir = OUTPUT_BASE / "convnextv2_base_frozen_head_v2"
exp_dir.mkdir(parents=True, exist_ok=True)

best_acc = -1.0; best_epoch = 0; no_improve = 0; rows = []

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
    preds = (probs >= 0.5).astype(int)
    vm = {"loss": vloss/max(vn,1), "acc": accuracy_score(labels, preds), "f1": f1_score(labels, preds, zero_division=0)}
    row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
           "train_loss": total_loss/max(total_n,1), "train_acc": total_correct/max(total_n,1),
           "val_loss": vm["loss"], "val_acc": vm["acc"], "val_f1": vm["f1"]}
    rows.append(row); pd.DataFrame(rows).to_csv(exp_dir / "metrics.csv", index=False)
    print(f"  E{epoch}: train_loss={row['train_loss']:.4f} acc={row['train_acc']:.4f} | val_acc={vm['acc']:.4f} f1={vm['f1']:.4f}")

    hs = model.get_classifier().state_dict()
    torch.save({"head": hs, "epoch": epoch, "metrics": vm}, exp_dir / "last_head.pth")
    if vm["acc"] > best_acc:
        best_acc = vm["acc"]; best_epoch = epoch; no_improve = 0
        torch.save({"head": hs, "epoch": epoch, "metrics": vm}, exp_dir / "best_head.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE: print(f"  Early stop at {epoch}"); break

ckpt = torch.load(exp_dir / "best_head.pth", map_location=DEVICE)
model.get_classifier().load_state_dict(ckpt["head"]); model.eval()

print("\n4-view TTA inference...")
tfs = tta_tfs(); vp = {}; ids_ref = None
for vn, tfm in tfs.items():
    ds = ClsDataset(test_df, test_dir, tfm, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    ids, pl = [], []
    with torch.no_grad():
        for x, img_ids in tqdm(loader, desc=f"TTA {vn}"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"): logits = model(x)
            prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            ids.extend(list(img_ids)); pl.append(prob)
    pa = np.concatenate(pl)
    if ids_ref is None: ids_ref = ids
    vp[vn] = pa
mp = np.mean(list(vp.values()), axis=0)

out = pd.DataFrame({"id": ids_ref, "prob": mp})
for k, v in vp.items(): out[f"prob_{k}"] = v
out.to_csv(exp_dir / "test_prob.csv", index=False)
sub = pd.DataFrame({"id": ids_ref, "label": (mp >= 0.5).astype(int)})
sub.to_csv(exp_dir / "submission.csv", index=False)
summary = {"best_epoch": best_epoch, "best_val_acc": best_acc, "n_positives": int(sub["label"].sum())}
with open(exp_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)
print(f"ConvNeXtV2 v2: val_acc={best_acc:.4f} epoch={best_epoch} positives={int(sub['label'].sum())}")

# ---- Blend ----
print("\n=== Blend ===")
dino = pd.read_csv(OUTPUT_BASE / "dinov3_l_frozen_head" / "test_prob.csv")
df = dino[["id", "prob"]].rename(columns={"prob": "prob_dino"})
df = df.merge(out[["id", "prob"]].rename(columns={"prob": "prob_conv"}), on="id")
df["prob_v13"] = 0.80 * df["prob_dino"] + 0.15 * df["prob_conv"]
df["label"] = (df["prob_v13"] >= 0.5).astype(int)

bd = OUTPUT_BASE / "blend_v2"; bd.mkdir(exist_ok=True)
df.to_csv(bd / "test_prob_blend.csv", index=False)

tmpl = pd.read_csv(DATA_ROOT / "sample_submission.csv")
sub = tmpl.copy(); sub["label"] = sub["id"].map(df.set_index("id")["label"]).fillna(0).astype(int)
sub.to_csv(bd / "submission_blend_threshold05.csv", index=False)
print(f"Blend threshold 0.5: {int(sub['label'].sum())} positives")

for k in [80, 82, 85, 88, 90, 92, 95]:
    top = set(df.sort_values("prob_v13", ascending=False).head(k)["id"].tolist())
    s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
    s.to_csv(bd / f"submission_blend_top{k}.csv", index=False)
    print(f"  Top-{k}: {int(s['label'].sum())} positives")

print("Done!")
