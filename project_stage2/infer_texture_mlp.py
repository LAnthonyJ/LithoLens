"""TTA inference for texture_mlp checkpoint"""
import os; os.environ["HF_HUB_OFFLINE"] = "1"
import json, numpy as np, pandas as pd, torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from PIL import Image

DATA_ROOT = Path("/data/数据科学实践project")
OUT = DATA_ROOT / "runs" / "v13_proper" / "texture_mlp"
DEV = "cuda"; BS = 8; IMG = 512; TDIM = 21

class Cls(Dataset):
    def __init__(self, df, tx, tf=None, it=False):
        self.df = df.reset_index(drop=True); self.tx = tx.astype(np.float32)
        self.tf = tf; self.it = it
        self.td = DATA_ROOT / "test_images" / "test_images"
        if not self.td.is_dir(): self.td = DATA_ROOT / "test_images"
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.td / r["id"]).convert("RGB")
        if self.tf: img = self.tf(img)
        if self.it: return img, torch.from_numpy(self.tx[i]), r["id"]
        return img, torch.from_numpy(self.tx[i]), int(r["label"])

class M(nn.Module):
    def __init__(self):
        super().__init__(); self.f = True
        self.bb = timm.create_model("vit_large_patch16_dinov3.lvd1689m",
            pretrained=True, num_classes=0, img_size=IMG,
            pretrained_cfg_overlay={"file": "/data/models/dinov3_large/model.safetensors"})
        for p in self.bb.parameters(): p.requires_grad = False; self.bb.eval()
        self.h = nn.Sequential(nn.LayerNorm(1045), nn.Linear(1045, 512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 2))
    def forward(self, x, tx):
        with torch.no_grad(): f = self.bb(x)
        return self.h(torch.cat([f, tx], 1))
    def train(self, m=True):
        super().train(m)
        if self.f: self.bb.eval()
        return self

norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
def mk(extra): return transforms.Compose([
    transforms.Resize(512), transforms.CenterCrop(512), *extra, transforms.ToTensor(), norm])
tta = {
    "orig": mk([]),
    "hflip": mk([transforms.RandomHorizontalFlip(p=1.0)]),
    "vflip": mk([transforms.RandomVerticalFlip(p=1.0)]),
    "hvflip": mk([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)]),
}

tdf = pd.read_csv(DATA_ROOT / "sample_submission.csv")
tx = np.load(OUT / "texture_cache.npz")["test_tex"]

m = M().to(DEV)
ck = torch.load(OUT / "best_head.pth", map_location=DEV)
m.h.load_state_dict(ck["head"]); m.eval()
bt = float(ck["metrics"]["threshold"])
print(f"Threshold: {bt:.4f}")

vp = {}; ir = None
for vn, tf in tta.items():
    ds = Cls(tdf, tx, tf, True)
    dl = DataLoader(ds, BS, shuffle=False, num_workers=4, pin_memory=True)
    ids, pl = [], []
    with torch.no_grad():
        for x, t, img_ids in tqdm(dl, desc=f"TTA {vn}"):
            x, t = x.to(DEV), t.to(DEV)
            with torch.amp.autocast("cuda"): lg = m(x, t)
            prob = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy()
            ids.extend(list(img_ids)); pl.append(prob)
    pa = np.concatenate(pl)
    if ir is None: ir = ids
    vp[vn] = pa
mp = np.mean(list(vp.values()), 0)

out = pd.DataFrame({"id": ir, "prob": mp})
for k, v in vp.items(): out[f"prob_{k}"] = v
out.to_csv(OUT / "test_prob.csv", index=False)

tmpl = pd.read_csv(DATA_ROOT / "sample_submission.csv")
for tn, tv in [("dynamic", bt), ("th05", 0.5)]:
    sub = tmpl.copy()
    sub["label"] = sub["id"].map(dict(zip(ir, mp >= tv))).fillna(0).astype(int)
    sub.to_csv(OUT / f"submission_{tn}.csv", index=False)
    print(f"Submission {tn} (th={tv:.4f}): {int(sub['label'].sum())} pos")

for k in [80, 82, 84, 85, 88, 90, 92, 95]:
    top = set(out.sort_values("prob", ascending=False).head(k)["id"].tolist())
    s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
    s.to_csv(OUT / f"submission_top{k}.csv", index=False)
    print(f"  Top-{k}: {int(s['label'].sum())} pos")

# Blend
cp = DATA_ROOT / "runs" / "v13_proper" / "convnextv2_base_frozen_head_v2" / "test_prob.csv"
if cp.exists():
    cv = pd.read_csv(cp)
    df = out[["id", "prob"]].rename(columns={"prob": "pd"})
    df = df.merge(cv[["id", "prob"]].rename(columns={"prob": "pc"}), on="id")
    df["pv"] = 0.80 * df["pd"] + 0.15 * df["pc"]
    df["label"] = (df["pv"] >= 0.5).astype(int)
    bd = OUT / "blend"; bd.mkdir(exist_ok=True)
    df.to_csv(bd / "test_prob_blend.csv", index=False)
    for k in [80, 82, 84, 85, 88, 90, 92, 95]:
        top = set(df.sort_values("pv", ascending=False).head(k)["id"].tolist())
        s = tmpl.copy(); s["label"] = s["id"].map(lambda x: 1 if x in top else 0).astype(int)
        s.to_csv(bd / f"submission_blend_top{k}.csv", index=False)
        print(f"  Blend Top-{k}: {int(s['label'].sum())} pos")

json.dump({"threshold": bt}, open(OUT / "summary.json", "w"))
print("Done!")
