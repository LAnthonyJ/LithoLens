from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from make_review_boards import ROOT, OUT, TEST_DIR, TRAIN_DIR, make_sheet, read_labels


def crop_object(img: Image.Image) -> Image.Image:
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    rgb.thumbnail((420, 420), Image.Resampling.LANCZOS)
    arr = np.asarray(rgb)
    bright = arr.mean(axis=2)
    spread = arr.max(axis=2) - arr.min(axis=2)
    mask = (bright < 245) | (spread > 18)
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return rgb
    pad = 8
    left = max(0, int(xs.min()) - pad)
    right = min(rgb.width, int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(rgb.height, int(ys.max()) + pad + 1)
    return rgb.crop((left, top, right, bottom))


def feature(path: Path) -> np.ndarray:
    img = crop_object(Image.open(path))
    img.thumbnail((96, 96), Image.Resampling.LANCZOS)
    arr = np.asarray(img.convert("HSV"), dtype=np.float32)
    bright = arr[:, :, 2]
    sat = arr[:, :, 1]
    mask = (bright < 245) | (sat > 18)
    if mask.sum() < 20:
        mask = np.ones(bright.shape, dtype=bool)
    pix = arr[mask]
    h = pix[:, 0] / 256.0
    s = pix[:, 1] / 256.0
    v = pix[:, 2] / 256.0
    hist, _ = np.histogramdd(
        np.stack([h, s, v], axis=1),
        bins=(8, 4, 4),
        range=((0, 1), (0, 1), (0, 1)),
    )
    hist = hist.reshape(-1).astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    stats = np.array(
        [
            h.mean(),
            s.mean(),
            v.mean(),
            h.std(),
            s.std(),
            v.std(),
            float(mask.mean()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([hist, stats * 0.35])


def nearest_refs(target_ids: list[str], per_class: int = 5, max_train_per_class: int = 450) -> Path:
    labels = read_labels(ROOT / "train_labels.csv")
    rng = random.Random(531)
    positives = [image_id for image_id, label in labels.items() if label == 1]
    negatives = [image_id for image_id, label in labels.items() if label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    train_ids = sorted(positives[:max_train_per_class] + negatives[:max_train_per_class])
    train_features = np.stack([feature(TRAIN_DIR / image_id) for image_id in train_ids])
    train_features /= np.maximum(1e-8, np.linalg.norm(train_features, axis=1, keepdims=True))

    items: list[tuple[Path, str, int | None]] = []
    for target_id in target_ids:
        target_feature = feature(TEST_DIR / target_id)
        target_feature = target_feature / max(1e-8, float(np.linalg.norm(target_feature)))
        distances = 1.0 - train_features @ target_feature
        items.append((TEST_DIR / target_id, f"TEST {target_id}", None))
        for wanted_label in (1, 0):
            class_indices = [idx for idx, image_id in enumerate(train_ids) if labels[image_id] == wanted_label]
            class_indices.sort(key=lambda idx: float(distances[idx]))
            for idx in class_indices[:per_class]:
                image_id = train_ids[idx]
                items.append((TRAIN_DIR / image_id, f"tr{wanted_label} {image_id}", wanted_label))

    out = OUT / "round5_similarity_refs.jpg"
    make_sheet(
        items,
        out,
        cols=11,
        cell=190,
        label_h=40,
        title="target + nearest train positives then nearest train negatives",
    )
    return out


if __name__ == "__main__":
    ids = [
        "000037.jpg",
        "000111.jpg",
        "000164.jpg",
        "000172.jpg",
        "000124.jpg",
        "000126.jpg",
        "000141.jpg",
        "000188.jpg",
    ]
    print(nearest_refs(ids))
