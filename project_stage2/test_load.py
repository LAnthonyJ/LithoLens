import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch, timm

print("Test 1: pretrained=False + pretrained_cfg_overlay")
try:
    model = timm.create_model(
        "convnextv2_base", pretrained=False, num_classes=2,
        pretrained_cfg_overlay={"file": "/data/models/convnextv2_base_22k_384_ema.pt"},
    )
    print(f"Created: {type(model).__name__}")
    p = next(model.parameters())
    print(f"First param mean={p.mean():.6f}, std={p.std():.6f}")
except Exception as e:
    print(f"Error: {e}")

print()
print("Test 2: pretrained=True + pretrained_cfg_overlay (offline)")
try:
    model = timm.create_model(
        "convnextv2_base", pretrained=True, num_classes=2,
        pretrained_cfg_overlay={"file": "/data/models/convnextv2_base_22k_384_ema.pt"},
    )
    print(f"Created: {type(model).__name__}")
    p = next(model.parameters())
    print(f"First param mean={p.mean():.6f}, std={p.std():.6f}")
except Exception as e:
    print(f"Error: {e}")
