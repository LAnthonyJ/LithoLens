
import os; os.environ["HF_HUB_OFFLINE"] = "1"
import json, numpy as np, pandas as pd, torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from PIL import Image

DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "v13_proper" / "pseudo_self"
DEVICE = "cuda"; BATCH = 8

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True); self.transform = transform
        d = DATA_ROOT / "test_images" / "test_images"
        self.test_dir = d if d.is_dir() else DATA_ROOT / "test_images"
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.test_dir / row["id"]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, row["id"]

class DINOv3Classifier(nn.Module):
    def __init__(self):
        super().__init__(); self.freeze = True
        self.backbone = timm.create_model("vit_large_patch16_dinov3.lvd1689m",
            pretrained=True, num_classes=0, img_size=512,
            pretrained_cfg_overlay={"file": "/data/models/dinov3_large/model.safetensors"})
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        self.head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 2))
    def forward(self, x):
        with torch.no_grad(): feat = self.backbone(x)
        return self.head(feat)
    def train(self, mode=True):
        super().train(mode)
        if self.freeze: self.backbone.eval()
        return self

def build_tta():
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    def m(extra): return transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), *extra, transforms.ToTensor(), norm])
    return {"orig": m([]), "hflip": m([transforms.RandomHorizontalFlip(p=1.0)]),
            "vflip": m([transforms.RandomVerticalFlip(p=1.0)]),
            "hvflip": m([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)])}

test_df = pd.read_csv(DATA_ROOT / "sample_submission.csv")
model = DINOv3Classifier().to(DEVICE)
ckpt = torch.load(OUTPUT_BASE / "best_head.pth", map_location=DEVICE)
model.head.load_state_dict(ckpt["head"]); model.eval()
best_th = float(ckpt["metrics"]["threshold"])
print(f"Threshold: {best_th:.4f}")

print("TTA...")
tta_tfs = build_tta(); vp = {}; ids_ref = None
for vn, tfm in tta_tfs.items():
    ds = TestDataset(test_df, tfm)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    ids, pl = [], []
    with torch.no_grad():
        for x, img_ids in tqdm(loader, desc=f"TTA {vn}"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"): logits = model(x)
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

for k in [80,82,85,88,90,92,95]:
    top = set(out.sort_values("prob", ascending=False).head(k)["id"].tolist())
    s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
    s.to_csv(OUTPUT_BASE / f"submission_top{k}.csv", index=False)
    print(f"  Top-{k}: {int(s['label'].sum())} positives")

conv_path = DATA_ROOT / "runs" / "v13_proper" / "convnextv2_base_frozen_head_v2" / "test_prob.csv"
if conv_path.exists():
    print("\n=== Blend ===")
    conv = pd.read_csv(conv_path)
    df = out[["id","prob"]].rename(columns={"prob":"prob_dino"})
    df = df.merge(conv[["id","prob"]].rename(columns={"prob":"prob_conv"}), on="id")
    df["prob_v13"] = 0.80*df["prob_dino"] + 0.15*df["prob_conv"]
    df["label"] = (df["prob_v13"] >= 0.5).astype(int)
    bd = OUTPUT_BASE / "blend"; bd.mkdir(exist_ok=True)
    df.to_csv(bd / "test_prob_blend.csv", index=False)
    for k in [80,82,85,88,90,92,95]:
        top = set(df.sort_values("prob_v13", ascending=False).head(k)["id"].tolist())
        s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
        s.to_csv(bd / f"submission_blend_top{k}.csv", index=False)
        print(f"  Blend Top-{k}: {int(s['label'].sum())} positives")

json.dump({"best_threshold": best_th}, open(OUTPUT_BASE / "summary.json", "w"))
print("Done!")
