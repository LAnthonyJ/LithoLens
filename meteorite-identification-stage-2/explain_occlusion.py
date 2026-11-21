import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from stage2_utils import build_model, build_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Occlusion explanation for a trained fold")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--view", type=str, default="full", choices=["full", "crop"])
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-float(x)))


def colorize(mask):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    heat = np.zeros((*mask.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(255 * mask, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(180 * (1.0 - np.abs(mask - 0.55) / 0.55), 0, 180).astype(np.uint8)
    heat[..., 2] = np.clip(80 * (1.0 - mask), 0, 80).astype(np.uint8)
    return heat


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_tf = build_transforms(args.img_size, view=args.view)
    model = build_model(args.model, pretrained=args.pretrained).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with Image.open(args.image) as img:
        raw = img.convert("RGB")
    tensor = test_tf(raw).unsqueeze(0).to(device)

    with torch.inference_mode():
        base_logit = model(tensor).view(-1)[0].float().cpu().item()
    base_prob = sigmoid(base_logit)

    _, _, height, width = tensor.shape
    grid = int(max(4, args.grid))
    patch_h = int(np.ceil(height / grid))
    patch_w = int(np.ceil(width / grid))
    importance = np.zeros((grid, grid), dtype=np.float32)

    with torch.inference_mode():
        for gy in range(grid):
            for gx in range(grid):
                y1 = gy * patch_h
                x1 = gx * patch_w
                y2 = min(height, y1 + patch_h)
                x2 = min(width, x1 + patch_w)
                occluded = tensor.clone()
                occluded[:, :, y1:y2, x1:x2] = 0.0
                logit = model(occluded).view(-1)[0].float().cpu().item()
                prob = sigmoid(logit)
                importance[gy, gx] = max(0.0, base_prob - prob)

    heat_small = Image.fromarray(colorize(importance), mode="RGB")
    heat = heat_small.resize((args.img_size, args.img_size), Image.Resampling.BICUBIC)
    raw_square = ImageOps.pad(raw, (args.img_size, args.img_size), color=(255, 255, 255))
    overlay = Image.blend(raw_square, heat, alpha=0.42)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, args.img_size - 1, 28), fill=(255, 255, 255))
    draw.text(
        (8, 7),
        f"{args.image.name}  meteorite_prob={base_prob:.4f}  checkpoint={args.checkpoint.name}",
        fill=(0, 0, 0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output, quality=92)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
