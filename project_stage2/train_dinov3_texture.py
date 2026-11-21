"""
DINOv3-L + 18-dim texture feature augmentation.
Texture features are pre-computed (cv2, no GPU) and concatenated with DINO embeddings.
MLP head: 1043 → 512 → 2  (1024 DINO + 19 texture)
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, time, pickle, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
import timm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from pathlib import Path
from PIL import Image
import cv2

# ============================================================
# Config
# ============================================================
DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "v13_proper" / "dinov3_texture"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 35
IMG_SIZE = 512
EPOCHS = 20
BATCH_SIZE = 8
LR = 4e-5
WD = 2e-2
LS = 0.1
PATIENCE = 8
TEXTURE_DIM = 19  # 15 (multi-scale var) + 2 (gradient) + 1 (FFT) + 1 (entropy)
DINO_DIM = 1024

torch.manual_seed(SEED); np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED); cudnn.benchmark = True
print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Texture Feature Extractor
# ============================================================
class TextureFeatureExtractor:
    def __init__(self, window_sizes=(7, 15, 31), hist_bins=32):
        self.window_sizes = window_sizes
        self.hist_bins = hist_bins

    def __call__(self, image):
        if hasattr(image, 'convert'):
            gray = np.array(image.convert('L').resize((256, 256)), dtype=np.float32)
        elif image.ndim == 3:
            gray = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = cv2.resize(image, (256, 256)).astype(np.float32)

        feats = []
        # 1. Multi-scale local variance (15 dims)
        for ws in self.window_sizes:
            kernel = np.ones((ws, ws), np.float32) / (ws * ws)
            local_mean = cv2.filter2D(gray, -1, kernel)
            local_sq_mean = cv2.filter2D(gray * gray, -1, kernel)
            local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0))
            feats.extend([float(np.mean(local_std)), float(np.std(local_std)),
                         float(np.percentile(local_std, 50)), float(np.percentile(local_std, 90)),
                         float(np.max(local_std))])
        # 2. Gradient energy (2 dims)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gm = np.sqrt(gx*gx + gy*gy)
        feats.extend([float(np.mean(gm)), float(np.std(gm))])
        # 3. Frequency domain (1 dim)
        mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        total = np.sum(mag)
        if total > 0:
            h, w = mag.shape; cy, cx = h//2, w//2; r = min(cy, cx)//3
            y, x = np.ogrid[:h, :w]
            mask = (y-cy)**2 + (x-cx)**2 >= r**2
            feats.append(float(np.sum(mag[mask])/total))
        else:
            feats.append(0.0)
        # 4. Histogram entropy (1 dim)
        hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [32], [0,256]).flatten().astype(np.float64)
        hist = hist / (hist.sum() + 1e-9)
        hist = hist[hist > 0]
        feats.append(float(-np.sum(hist * np.log2(hist))))

        arr = np.array(feats, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Dataset with pre-computed texture features
# ============================================================
class TextureAugDataset(Dataset):
    def __init__(self, df, img_dir, texture_feats, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.texture_feats = texture_feats.astype(np.float32)  # (N, 18) pre-scaled
        self.transform = transform
        self.is_test = is_test

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["id"]).convert("RGB")
        tex = torch.from_numpy(self.texture_feats[idx])
        if self.transform: img = self.transform(img)
        if self.is_test: return img, tex, row["id"]
        return img, tex, int(row["label"])


# ============================================================
# Transforms (same as corrected DINOv3)
# ============================================================
def build_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.6),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.0), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_val_transform():
    return transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_tta_transforms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def m(extra): return transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(IMG_SIZE), *extra, transforms.ToTensor(), norm])
    return {"orig": m([]), "hflip": m([transforms.RandomHorizontalFlip(p=1.0)]),
            "vflip": m([transforms.RandomVerticalFlip(p=1.0)]),
            "hvflip": m([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)])}


# ============================================================
# Model: DINOv3 + Texture Features
# ============================================================
class DINOv3TextureClassifier(nn.Module):
    def __init__(self, texture_dim=TEXTURE_DIM, dino_dim=DINO_DIM):
        super().__init__()
        self.freeze = True
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True, num_classes=0, img_size=512,
            pretrained_cfg_overlay={"file": "/data/models/dinov3_large/model.safetensors"},
        )
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()

        total_dim = dino_dim + texture_dim  # 1024 + 18 = 1042
        self.head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x, texture_feat):
        if self.freeze:
            with torch.no_grad(): feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        combined = torch.cat([feat, texture_feat], dim=1)
        return self.head(combined)

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
# Main
# ============================================================
def main():
    # Data paths
    train_dir = DATA_ROOT / "train_images" / "train_images"
    test_dir = DATA_ROOT / "test_images" / "test_images"
    if not train_dir.is_dir(): train_dir = DATA_ROOT / "train_images"
    if not test_dir.is_dir(): test_dir = DATA_ROOT / "test_images"

    train_df = pd.read_csv(DATA_ROOT / "train_labels.csv")
    test_df = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Pre-compute texture features
    print("\nPre-computing texture features...")
    extractor = TextureFeatureExtractor()

    # Check cache
    cache_dir = OUTPUT_BASE / "texture_cache"
    cache_dir.mkdir(exist_ok=True)
    train_tex_path = cache_dir / "train_texture.npz"
    test_tex_path = cache_dir / "test_texture.npz"

    if train_tex_path.exists() and test_tex_path.exists():
        print("Loading cached texture features...")
        train_tex = np.load(train_tex_path)["feats"]
        test_tex = np.load(test_tex_path)["feats"]
    else:
        print("Computing training texture features...")
        train_feats = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train tex"):
            img = Image.open(f"{train_dir}/{row['id']}")
            train_feats.append(extractor(img))
        train_tex_raw = np.stack(train_feats)

        print("Computing test texture features...")
        test_feats = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test tex"):
            img = Image.open(f"{test_dir}/{row['id']}")
            test_feats.append(extractor(img))
        test_tex_raw = np.stack(test_feats)

        # Fit RobustScaler on training data
        scaler = RobustScaler()
        train_tex = scaler.fit_transform(train_tex_raw).astype(np.float32)
        test_tex = scaler.transform(test_tex_raw).astype(np.float32)

        np.savez_compressed(train_tex_path, feats=train_tex)
        np.savez_compressed(test_tex_path, feats=test_tex)
        print("Texture features cached.")

    print(f"Train texture: {train_tex.shape}, Test texture: {test_tex.shape}")

    # Stratified split
    labels = train_df["label"].values
    rng = np.random.default_rng(SEED)
    tr_idx, val_idx = [], []
    for c in range(2):
        ci = np.where(labels == c)[0]
        rng.shuffle(ci)
        nv = int(len(ci) * 0.1)
        val_idx.extend(ci[:nv]); tr_idx.extend(ci[nv:])
    tr_idx = np.array(tr_idx); val_idx = np.array(val_idx)
    rng.shuffle(tr_idx); rng.shuffle(val_idx)
    print(f"Split: train={len(tr_idx)}, val={len(val_idx)}")

    # Datasets
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    tr_tex = train_tex[tr_idx]
    val_tex = train_tex[val_idx]

    train_ds = TextureAugDataset(tr_df, train_dir, tr_tex, build_train_transform())
    val_ds = TextureAugDataset(val_df, train_dir, val_tex, build_val_transform())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    print("\nLoading DINOv3 + Texture classifier...")
    model = DINOv3TextureClassifier().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LS)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Training
    best_f1 = -1.0; best_epoch = 0; no_improve = 0; rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_n, total_correct = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"E{epoch}/{EPOCHS}")
        for x, tex, y in pbar:
            x, tex, y = x.to(DEVICE, non_blocking=True), tex.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(x, tex); loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            bs = y.size(0); total_loss += loss.item()*bs; total_n += bs
            total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # Val
        model.eval()
        all_probs, all_labels, vloss, vn = [], [], 0.0, 0
        with torch.no_grad():
            for x, tex, y in val_loader:
                x, tex, y = x.to(DEVICE, non_blocking=True), tex.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(x, tex); l = criterion(logits, y)
                prob = torch.softmax(logits.float(), dim=1)[:, 1]
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

    # Load best head
    ckpt = torch.load(OUTPUT_BASE / "best_head.pth", map_location=DEVICE)
    model.head.load_state_dict(ckpt["head"])
    best_th = float(ckpt["metrics"]["threshold"])
    model.eval()
    print(f"\nBest: epoch={best_epoch}, f1={best_f1:.4f}, th={best_th:.4f}")

    # TTA inference with texture features
    print("\n4-view TTA inference...")
    tta_tfs = build_tta_transforms()
    view_probs = {}; ids_ref = None

    for vn, tfm in tta_tfs.items():
        ds = TextureAugDataset(test_df, test_dir, test_tex, tfm, is_test=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        ids, pl = [], []
        with torch.no_grad():
            for x, tex, img_ids in tqdm(loader, desc=f"TTA {vn}"):
                x, tex = x.to(DEVICE, non_blocking=True), tex.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda"): logits = model(x, tex)
                prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                ids.extend(list(img_ids)); pl.append(prob)
        pa = np.concatenate(pl)
        if ids_ref is None: ids_ref = ids
        view_probs[vn] = pa
    mp = np.mean(list(view_probs.values()), axis=0)

    # Save
    out = pd.DataFrame({"id": ids_ref, "prob": mp})
    for k, v in view_probs.items(): out[f"prob_{k}"] = v
    out.to_csv(OUTPUT_BASE / "test_prob.csv", index=False)

    # Submissions
    tmpl = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    for th_name, th_val in [("dynamic", best_th), ("th05", 0.5)]:
        sub = tmpl.copy()
        sub["label"] = sub["id"].map(lambda x: 1 if dict(zip(ids_ref, mp >= th_val)).get(x, 0) else 0).astype(int)
        sub.to_csv(OUTPUT_BASE / f"submission_{th_name}.csv", index=False)
        print(f"Submission {th_name} (th={th_val:.4f}): {int(sub['label'].sum())} positives")

    # Top-K
    for k in [80, 82, 85, 88, 90, 92, 95]:
        top = set(out.sort_values("prob", ascending=False).head(k)["id"].tolist())
        s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
        s.to_csv(OUTPUT_BASE / f"submission_top{k}.csv", index=False)
        print(f"  Top-{k}: {int(s['label'].sum())} positives")

    # Blend with ConvNeXtV2
    conv_path = DATA_ROOT / "runs" / "v13_proper" / "convnextv2_base_frozen_head_v2" / "test_prob.csv"
    if conv_path.exists():
        print("\n=== Blend with ConvNeXtV2 ===")
        conv = pd.read_csv(conv_path)
        df = out[["id","prob"]].rename(columns={"prob":"prob_dino_tex"})
        df = df.merge(conv[["id","prob"]].rename(columns={"prob":"prob_conv"}), on="id")
        df["prob_v13"] = 0.80*df["prob_dino_tex"] + 0.15*df["prob_conv"]
        df["label"] = (df["prob_v13"] >= 0.5).astype(int)

        bd = OUTPUT_BASE / "blend"; bd.mkdir(exist_ok=True)
        df.to_csv(bd / "test_prob_blend.csv", index=False)
        for k in [80, 82, 85, 88, 90, 92, 95]:
            top = set(df.sort_values("prob_v13", ascending=False).head(k)["id"].tolist())
            s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
            s.to_csv(bd / f"submission_blend_top{k}.csv", index=False)
            print(f"  Blend Top-{k}: {int(s['label'].sum())} positives")

    summary = {"best_epoch": best_epoch, "best_val_f1": best_f1, "best_threshold": best_th}
    with open(OUTPUT_BASE / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(f"\nDone! Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
