"""Run ConvNeXtV2-Base training only, then blend with existing DINOv3 results"""
import paramiko, os, time

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]
SERVER_PROJECT = '/data/数据科学实践project'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
print('Connected.')

# Upload fixed script
sftp = client.open_sftp()
sftp.put(os.path.join(LOCAL_DIR, 'train_v13_proper.py'), f'{SERVER_PROJECT}/train_v13_proper.py')
sftp.close()
print('Uploaded train_v13_proper.py (fixed)')

# Run ConvNeXtV2-Base ONLY
blend_script = '''
import os, json, time
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import timm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
from PIL import Image

# Setup
DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "v13_proper"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 35
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
PATIENCE = 8
IMG_SIZE = 384

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = True

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Data
train_dir = DATA_ROOT / "train_images" / "train_images"
test_dir = DATA_ROOT / "test_images" / "test_images"
if not train_dir.is_dir():
    train_dir = DATA_ROOT / "train_images"
if not test_dir.is_dir():
    test_dir = DATA_ROOT / "test_images"

train_df = pd.read_csv(DATA_ROOT / "train_labels.csv")
test_df = pd.read_csv(DATA_ROOT / "sample_submission.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

tr_df, val_df = train_test_split(train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"])
print(f"Split: train={len(tr_df)}, val={len(val_df)}")

# Dataset
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

# Transforms
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
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def make(extra):
        return transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), *extra, transforms.ToTensor(), norm])
    return {
        "orig": make([]),
        "hflip": make([transforms.RandomHorizontalFlip(p=1.0)]),
        "vflip": make([transforms.RandomVerticalFlip(p=1.0)]),
        "hvflip": make([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)]),
    }

# Load model CORRECTLY with pretrained_cfg_overlay
print("\\nLoading ConvNeXtV2-Base with local weights...")
model = timm.create_model(
    "convnextv2_base",
    pretrained=True,
    num_classes=2,
    pretrained_cfg_overlay={"file": "/data/models/convnextv2_base_22k_384_ema.pt"},
)
# Freeze backbone
for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True
model = model.to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable}")
# Verify weights loaded
p = next(model.parameters())
print(f"First param: mean={p.mean():.6f}, std={p.std():.6f}")

# Data loaders
train_tf = build_train_transform(IMG_SIZE)
val_tf = build_test_transform(IMG_SIZE)["orig"]
train_ds = ClsDataset(tr_df, train_dir, train_tf)
val_ds = ClsDataset(val_df, train_dir, val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Training
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda")

exp_dir = OUTPUT_BASE / "convnextv2_base_frozen_head_v2"
exp_dir.mkdir(parents=True, exist_ok=True)

best_acc = -1.0
best_epoch = 0
no_improve = 0
rows = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_n, total_correct = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for x, y in pbar:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
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

    # Val
    model.eval()
    all_probs, all_labels = [], []
    val_loss, val_n = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                l = criterion(logits, y)
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            bs = y.size(0)
            val_loss += l.item() * bs
            val_n += bs
            all_probs.append(prob.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)
    val_metrics = {
        "loss": float(val_loss / max(val_n, 1)),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    row = {
        "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
        "train_loss": total_loss / max(total_n, 1),
        "train_acc": total_correct / max(total_n, 1),
        "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"], "val_f1": val_metrics["f1"],
    }
    rows.append(row)
    pd.DataFrame(rows).to_csv(exp_dir / "metrics.csv", index=False)
    print(f"  E{epoch}: train_loss={row['train_loss']:.4f} acc={row['train_acc']:.4f} | val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}")

    head_state = model.get_classifier().state_dict()
    torch.save({"head": head_state, "epoch": epoch, "metrics": val_metrics}, exp_dir / "last_head.pth")
    if val_metrics["acc"] > best_acc:
        best_acc = val_metrics["acc"]
        best_epoch = epoch
        no_improve = 0
        torch.save({"head": head_state, "epoch": epoch, "metrics": val_metrics}, exp_dir / "best_head.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

# Load best head
ckpt = torch.load(exp_dir / "best_head.pth", map_location=DEVICE)
model.get_classifier().load_state_dict(ckpt["head"])
model.eval()

# 4-view TTA
print("\\nRunning 4-view TTA...")
tta_tfs = build_test_transform(IMG_SIZE)
view_probs = {}
ids_ref = None
for view_name, tfm in tta_tfs.items():
    ds = ClsDataset(test_df, test_dir, tfm, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    ids, probs_list = [], []
    with torch.no_grad():
        for x, img_ids in tqdm(loader, desc=f"TTA {view_name}"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
            prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            ids.extend(list(img_ids))
            probs_list.append(prob)
    probs_arr = np.concatenate(probs_list)
    if ids_ref is None:
        ids_ref = ids
    view_probs[view_name] = probs_arr
mean_prob = np.mean(list(view_probs.values()), axis=0)

out_prob = pd.DataFrame({"id": ids_ref, "prob": mean_prob})
for k, v in view_probs.items():
    out_prob[f"prob_{k}"] = v
out_prob.to_csv(exp_dir / "test_prob.csv", index=False)

sub = pd.DataFrame({"id": ids_ref, "label": (mean_prob >= 0.5).astype(int)})
sub.to_csv(exp_dir / "submission.csv", index=False)

summary = {"exp_name": "convnextv2_base_frozen_head_v2", "best_epoch": best_epoch,
           "best_val_acc": best_acc, "n_positives": int(sub["label"].sum())}
with open(exp_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\\nConvNeXtV2-Base v2: best_val_acc={best_acc:.4f} at epoch {best_epoch}, {int(sub['label'].sum())} positives")

# ---- Blend with existing DINOv3 ----
print("\\n=== Blending with DINOv3 ===")
dino = pd.read_csv(OUTPUT_BASE / "dinov3_l_frozen_head" / "test_prob.csv")
conv = pd.read_csv(exp_dir / "test_prob.csv")

df = dino[["id", "prob"]].rename(columns={"prob": "prob_dino"})
df = df.merge(conv[["id", "prob"]].rename(columns={"prob": "prob_conv"}), on="id")
df["prob_v13"] = 0.80 * df["prob_dino"] + 0.15 * df["prob_conv"]
df["label"] = (df["prob_v13"] >= 0.5).astype(int)

blend_dir = OUTPUT_BASE / "blend_v2"
blend_dir.mkdir(exist_ok=True)
df.to_csv(blend_dir / "test_prob_blend.csv", index=False)

template = pd.read_csv(DATA_ROOT / "sample_submission.csv")
for label_name in ["label"]:
    sub = template.copy()
    sub["label"] = sub["id"].map(df.set_index("id")[label_name]).fillna(0).astype(int)
    sub.to_csv(blend_dir / f"submission_blend_threshold05.csv", index=False)
    print(f"Blend threshold 0.5: {int(sub['label'].sum())} positives")

for k in [80, 82, 85, 88, 90, 92, 95]:
    df_sorted = df.sort_values("prob_v13", ascending=False)
    top_ids = set(df_sorted.head(k)["id"].tolist())
    sub = template.copy()
    sub["label"] = sub["id"].map(lambda x: 1 if x in top_ids else 0).astype(int)
    sub.to_csv(blend_dir / f"submission_blend_top{k}.csv", index=False)
    print(f"  Top-{k}: {int(sub['label'].sum())} positives")

print("\\nDone!")
'''

# Write and upload the fix script
with open('train_convnext_fix.py', 'w') as f:
    f.write(blend_script)

sftp = client.open_sftp()
sftp.put('train_convnext_fix.py', f'{SERVER_PROJECT}/train_convnext_fix.py')
sftp.close()
print('Uploaded train_convnext_fix.py')

# Run
print(f'Running ConvNeXtV2-Base training (start: {time.strftime("%H:%M:%S")})...')
stdin, stdout, stderr = client.exec_command(f'cd {SERVER_PROJECT} && python train_convnext_fix.py 2>&1', timeout=7200)
for line in iter(stdout.readline, ''):
    if line:
        print(line.rstrip())
err = stderr.read().decode().strip()
if err:
    print(f'STDERR: {err[:1000]}')

# Download
local_out = os.path.join(LOCAL_DIR, 'v13_proper_output')
os.makedirs(local_out, exist_ok=True)
sftp = client.open_sftp()

# Download v2 results
for subdir in ['convnextv2_base_frozen_head_v2', 'blend_v2']:
    remote_dir = f'{SERVER_PROJECT}/runs/v13_proper/{subdir}'
    local_sub = os.path.join(local_out, subdir)
    os.makedirs(local_sub, exist_ok=True)
    try:
        for item in sftp.listdir_attr(remote_dir):
            rp = f'{remote_dir}/{item.filename}'
            lp = os.path.join(local_sub, item.filename)
            sftp.get(rp, lp)
            print(f'  Downloaded: {subdir}/{item.filename}')
    except Exception as e:
        print(f'  Error: {e}')

sftp.close()
client.close()
print('\nDone!')
