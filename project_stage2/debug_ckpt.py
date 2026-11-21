"""Debug convnext checkpoint structure"""
import torch
import timm

ckpt_path = "/data/models/convnextv2_base_22k_384_ema.pt"
print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"Type: {type(ckpt)}")

if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print(f"Keys ({len(keys)}): {keys}")

    # Check if it's a model_ema checkpoint
    if "model_ema" in ckpt:
        print("\nFound model_ema key!")
        ema = ckpt["model_ema"]
        if isinstance(ema, dict):
            ema_keys = list(ema.keys())
            print(f"EMA keys ({len(ema_keys)}): first 20:")
            for k in sorted(ema_keys)[:20]:
                print(f"  {k}")
            print(f"  ...")
            for k in sorted(ema_keys)[-5:]:
                print(f"  {k}")
    elif "model" in ckpt:
        print("\nFound model key!")
        model_state = ckpt["model"]
        if isinstance(model_state, dict):
            mk = list(model_state.keys())
            print(f"Model keys ({len(mk)}): first 20:")
            for k in sorted(mk)[:20]:
                print(f"  {k}")
    elif "state_dict" in ckpt:
        print("\nFound state_dict key!")
        sd = ckpt["state_dict"]
        mk = list(sd.keys())
        print(f"State dict keys ({len(mk)}): first 20:")
        for k in sorted(mk)[:20]:
            print(f"  {k}")
    else:
        # Check if it's a raw state dict
        print("\nRaw dict, first 20 keys:")
        for k in sorted(keys)[:20]:
            print(f"  {k}")

# Try to load with timm
print("\n\n=== Trying timm create_model with local file ===")
try:
    model = timm.create_model(
        "convnextv2_base",
        pretrained=True,
        num_classes=2,
        pretrained_cfg_overlay={"file": ckpt_path},
    )
    print("Success! Model loaded with pretrained_cfg_overlay")
except Exception as e:
    print(f"Failed: {e}")
    print("\nTrying manual state_dict loading...")
    # Try to figure out the state dict format
    model = timm.create_model("convnextv2_base", pretrained=False, num_classes=2)
    model_keys = set(model.state_dict().keys())
    print(f"Model state dict keys ({len(model_keys)}): first 20:")
    for k in sorted(model_keys)[:20]:
        print(f"  {k}")
    print(f"  ...")
    for k in sorted(model_keys)[-5:]:
        print(f"  {k}")
