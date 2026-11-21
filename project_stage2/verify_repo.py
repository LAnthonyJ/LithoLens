"""Verify all files needed for reproducibility exist on server"""
from pathlib import Path
import torch, pandas as pd, numpy as np

print("1. Environment check")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")

print("\n2. Checkpoint files")
checks = {
    "DINOv3+Tex MLP head": Path("/data/数据科学实践project/runs/final_top86/best_head.pth"),
    "ConvNeXtV2 head": Path("/data/数据科学实践project/runs/v13_proper/convnextv2_base_frozen_head_v2/best_head.pth"),
    "ConvNeXtV2 probs": Path("/data/数据科学实践project/runs/v13_proper/convnextv2_base_frozen_head_v2/test_prob.csv"),
    "Texture cache": Path("/data/数据科学实践project/runs/final_top86/texture_cache.npz"),
    "DINOv3 backbone": Path("/data/models/dinov3_large/model.safetensors"),
    "Dataset folder": Path("/data/数据科学实践project/Dataset"),
    "Train labels": Path("/data/数据科学实践project/train_labels.csv"),
    "Sample submission": Path("/data/数据科学实践project/sample_submission.csv"),
    "Test images": Path("/data/数据科学实践project/test_images"),
}
all_ok = True
for name, p in checks.items():
    ok = p.exists()
    if not ok: all_ok = False
    print(f"   {name}: {'OK' if ok else 'MISSING'}")

print(f"\n3. Result: {'ALL FILES PRESENT - Reproducible!' if all_ok else 'SOME FILES MISSING'}")

if Path("/data/数据科学实践project/Dataset").is_dir():
    ds = Path("/data/数据科学实践project/Dataset")
    n_jpg = sum(1 for _ in ds.rglob("*.jpg"))
    n_png = sum(1 for _ in ds.rglob("*.png"))
    print(f"\n4. Dataset stats: {n_jpg + n_png} images ({n_jpg} jpg, {n_png} png)")
