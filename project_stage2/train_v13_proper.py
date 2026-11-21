"""
Proper v13 replication: frozen backbone + trainable classifier head + 4-view TTA
Based on teammate's train_timm_frozen.py and infer_tta_dinov3.py

Pipeline:
1. DINOv3-L: frozen + train head → 4-view TTA → prob_dino.csv
2. ConvNeXtV2-Base: frozen + train head → 4-view TTA → prob_convbase.csv
3. Blend: 0.80*prob_dino + 0.15*prob_convbase + 0.05*prob_convlarge (if available)
4. Threshold: 0.5 → submission
"""
import os, sys, time, json, argparse
os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline for timm
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import timm

# ============================================================
# 0. Config
# ============================================================
DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "v13_proper"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Models to train
MODELS = {
    "dinov3_l": {
        "model_name": "vit_large_patch16_dinov3.lvd1689m",
        "local_weights": "/data/models/dinov3_large/model.safetensors",
        "img_size": 512,
        "epochs": 20,
        "batch_size": 8,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "label_smoothing": 0.1,
        "seed": 35,
        "patience": 8,
        "use_amp": True,
        "blend_weight": 0.80,
    },
    "convnextv2_base": {
        "model_name": "convnextv2_base",
        "pretrained_path": "/data/models/convnextv2_base_22k_384_ema.pt",
        "img_size": 384,
        "epochs": 20,
        "batch_size": 8,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "label_smoothing": 0.1,
        "seed": 35,
        "patience": 8,
        "use_amp": True,
        "blend_weight": 0.15,
    },
}


# ============================================================
# 1. Dataset
# ============================================================
class ClsDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["id"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, row["id"]
        return img, int(row["label"])


# ============================================================
# 2. Transforms
# ============================================================
def build_train_transform(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def build_test_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_tta_transforms(img_size):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def make(extra):
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            *extra,
            transforms.ToTensor(),
            norm,
        ])

    return {
        "orig": make([]),
        "hflip": make([transforms.RandomHorizontalFlip(p=1.0)]),
        "vflip": make([transforms.RandomVerticalFlip(p=1.0)]),
        "hvflip": make([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ]),
    }


# ============================================================
# 3. Model helpers
# ============================================================
def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    return model


def load_backbone(cfg, device):
    """Load a pretrained timm model, fully offline."""
    model_name = cfg["model_name"]
    local_weights = cfg.get("local_weights")
    pretrained_path = cfg.get("pretrained_path")

    print(f"  Loading {model_name}...")

    if local_weights:
        # DINOv3 from local safetensors
        model = timm.create_model(model_name, pretrained=False, num_classes=2)
        from safetensors.torch import load_file
        state = load_file(local_weights)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded local weights from {local_weights}")
    elif pretrained_path:
        # ConvNeXtV2 from local checkpoint
        # MUST use pretrained_cfg_overlay because checkpoint uses Facebook key names
        # (e.g. downsample_layers.0.0.weight) that timm internally converts
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=2,
            pretrained_cfg_overlay={"file": pretrained_path},
        )
        print(f"  Loaded from {pretrained_path}")
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=2)

    model = freeze_backbone(model)
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable}")
    return model


# ============================================================
# 4. Train / Eval / Predict
# ============================================================
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    all_probs, all_labels = [], []
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            all_probs.append(prob.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": float(total_loss / max(total_n, 1)),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def predict_single_view(model, loader, device, use_amp=True):
    model.eval()
    ids, probs = [], []
    with torch.no_grad():
        for x, img_ids in tqdm(loader, desc="Test", leave=False):
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits = model(x)
            prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            ids.extend(list(img_ids))
            probs.append(prob)
    return ids, np.concatenate(probs)


def predict_tta(model, test_df, img_dir, img_size, batch_size, num_workers, device):
    tta_tfs = build_tta_transforms(img_size)
    view_probs = {}
    ids_ref = None

    for view_name, tfm in tta_tfs.items():
        ds = ClsDataset(test_df, img_dir, transform=tfm, is_test=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
        ids, probs = predict_single_view(model, loader, device)
        if ids_ref is None:
            ids_ref = ids
        else:
            assert ids_ref == ids, f"TTA order mismatch at {view_name}"
        view_probs[view_name] = probs

    mean_prob = np.mean(list(view_probs.values()), axis=0)
    return ids_ref, mean_prob, view_probs


# ============================================================
# 5. Train one model
# ============================================================
def train_one_model(cfg, exp_name, train_df, test_df, train_dir, test_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Training: {exp_name}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    exp_dir = OUTPUT_BASE / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Seed
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

    # Split
    tr_df, val_df = train_test_split(
        train_df, test_size=0.10, random_state=seed, stratify=train_df["label"])
    print(f"Train: {len(tr_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    img_size = cfg["img_size"]
    train_tf = build_train_transform(img_size)
    val_tf = build_test_transform(img_size)

    train_ds = ClsDataset(tr_df, train_dir, train_tf, is_test=False)
    val_ds = ClsDataset(val_df, train_dir, val_tf, is_test=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=min(4, cfg["batch_size"] // 2), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                           num_workers=min(4, cfg["batch_size"] // 2), pin_memory=True)

    # Model
    model = load_backbone(cfg, device)

    # Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg["use_amp"]))

    best_acc = -1.0
    best_epoch = 0
    no_improve = 0
    rows = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total_n, total_correct = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [Train]")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda" and cfg["use_amp"])):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(exp_dir / "metrics.csv", index=False)

        print(f"  E{epoch} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}")

        head_state = model.get_classifier().state_dict()
        torch.save({"head": head_state, "epoch": epoch, "metrics": val_metrics}, exp_dir / "last_head.pth")

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            no_improve = 0
            torch.save({"head": head_state, "epoch": epoch, "metrics": val_metrics}, exp_dir / "best_head.pth")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best head and predict
    ckpt = torch.load(exp_dir / "best_head.pth", map_location=device)
    model.get_classifier().load_state_dict(ckpt["head"])
    model.eval()

    print(f"\n  Running 4-view TTA inference...")
    t0 = time.time()
    ids, prob, view_probs = predict_tta(
        model, test_df, test_dir, img_size, cfg["batch_size"],
        min(4, cfg["batch_size"] // 2), device)

    out_prob = pd.DataFrame({"id": ids, "prob": prob})
    for k, v in view_probs.items():
        out_prob[f"prob_{k}"] = v
    out_prob.to_csv(exp_dir / "test_prob.csv", index=False)

    sub = pd.DataFrame({"id": ids, "label": (prob >= 0.5).astype(int)})
    sub.to_csv(exp_dir / "submission.csv", index=False)

    summary = {
        "exp_name": exp_name,
        "model_name": cfg["model_name"],
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "n_positives": int(sub["label"].sum()),
        "test_prob_path": str(exp_dir / "test_prob.csv"),
        "elapsed_s": time.time() - t0,
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Best val_acc={best_acc:.4f} at epoch {best_epoch}")
    print(f"  Positives at threshold 0.5: {int(sub['label'].sum())}")
    print(f"  Output: {exp_dir}")

    return exp_dir / "test_prob.csv", summary


# ============================================================
# 6. Main
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_dir = DATA_ROOT / "train_images" / "train_images"
    test_dir = DATA_ROOT / "test_images" / "test_images"
    train_csv = DATA_ROOT / "train_labels.csv"
    sample_csv = DATA_ROOT / "sample_submission.csv"

    # Resolve paths (handle different structures)
    if not train_dir.is_dir():
        alt = DATA_ROOT / "train_images"
        if alt.is_dir():
            train_dir = alt
    if not test_dir.is_dir():
        alt = DATA_ROOT / "test_images"
        if alt.is_dir():
            test_dir = alt

    print(f"Train dir: {train_dir}")
    print(f"Test dir: {test_dir}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(sample_csv)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Labels: {train_df['label'].value_counts().sort_index().to_dict()}")

    # Train each model
    prob_files = {}
    summaries = {}

    for model_key, cfg in MODELS.items():
        exp_name = f"{model_key}_frozen_head"
        prob_file, summary = train_one_model(
            cfg, exp_name, train_df, test_df, train_dir, test_dir)
        prob_files[model_key] = prob_file
        summaries[model_key] = summary

    # ---- Blend ----
    print(f"\n{'='*60}")
    print("Blending probabilities")
    print(f"{'='*60}")

    blend_dir = OUTPUT_BASE / "blend"
    blend_dir.mkdir(exist_ok=True)

    # Read all probability files
    all_probs = {}
    for key, path in prob_files.items():
        df = pd.read_csv(path)
        all_probs[key] = df.set_index("id")["prob"]
        cfg = MODELS[key]
        print(f"  {cfg['model_name']}: weight={cfg['blend_weight']}, prob range=[{df['prob'].min():.4f}, {df['prob'].max():.4f}]")

    # Compute weighted blend
    prob_blend = pd.DataFrame(index=all_probs[list(all_probs.keys())[0]].index)
    prob_blend["prob_dino"] = all_probs["dinov3_l"]
    prob_blend["prob_convbase"] = all_probs["convnextv2_base"]

    prob_blend["prob_v13"] = (
        MODELS["dinov3_l"]["blend_weight"] * prob_blend["prob_dino"]
        + MODELS["convnextv2_base"]["blend_weight"] * prob_blend["prob_convbase"]
    )

    # Scale to match the missing 5%: renormalize weights
    total_w = MODELS["dinov3_l"]["blend_weight"] + MODELS["convnextv2_base"]["blend_weight"]
    prob_blend["prob_v13_renorm"] = prob_blend["prob_v13"] / total_w  # Renormalize to sum-to-1

    prob_blend["label"] = (prob_blend["prob_v13_renorm"] >= 0.5).astype(int)
    prob_blend.index.name = "id"
    prob_blend.reset_index().to_csv(blend_dir / "test_prob_blend.csv", index=False)

    # Also save with just dinov3+convbase (no renormalization)
    prob_blend["label_orig"] = (prob_blend["prob_v13"] >= 0.5).astype(int)

    template = pd.read_csv(sample_csv)
    for label_col, suffix in [("label", "renorm"), ("label_orig", "raw")]:
        sub = template.copy()
        sub["label"] = sub["id"].map(prob_blend[label_col]).fillna(0).astype(int)
        sub.to_csv(blend_dir / f"submission_blend_{suffix}.csv", index=False)
        print(f"  submission_blend_{suffix}.csv: {int(sub['label'].sum())} positives")

    # Also save Top-K variants
    prob_sorted = prob_blend.sort_values("prob_v13_renorm", ascending=False)
    for k in [80, 82, 85, 88, 90, 92, 95]:
        top_ids = set(prob_sorted.head(k).index.tolist())
        sub = template.copy()
        sub["label"] = sub["id"].map(lambda x: 1 if x in top_ids else 0).astype(int)
        sub.to_csv(blend_dir / f"submission_blend_top{k}.csv", index=False)
        print(f"  Top-{k}: {int(sub['label'].sum())} positives")

    print(f"\nDone! All outputs in: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
