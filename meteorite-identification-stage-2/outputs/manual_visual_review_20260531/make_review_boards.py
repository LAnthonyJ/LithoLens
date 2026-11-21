from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "manual_visual_review_20260531"
TEST_DIR = ROOT / "test_images" / "test_images"
TRAIN_DIR = ROOT / "train_images" / "train_images"
TRAIN_WHITE_DIR = ROOT / "train_images_whitebackground"


def read_labels(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: int(row["label"]) for row in csv.DictReader(f)}


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = load_font(18)
SMALL = load_font(14)


def make_sheet(
    items: list[tuple[Path, str, int | None]],
    out_path: Path,
    *,
    cols: int = 5,
    cell: int = 260,
    label_h: int = 42,
    title: str = "",
    old_labels: dict[str, int] | None = None,
) -> None:
    rows = math.ceil(len(items) / cols)
    title_h = 34 if title else 0
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h) + title_h), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((8, 6), title, fill=(0, 0, 0), font=FONT)

    for index, (path, image_id, label) in enumerate(items):
        row, col = divmod(index, cols)
        x = col * cell
        y = title_h + row * (cell + label_h)
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            img = Image.new("RGB", (cell - 10, cell - 10), (235, 235, 235))

        thumb = ImageOps.contain(img, (cell - 12, cell - 12), method=Image.Resampling.LANCZOS)
        px = x + (cell - thumb.width) // 2
        py = y + (cell - thumb.height) // 2
        sheet.paste(thumb, (px, py))

        color = (35, 150, 75) if label == 1 else (160, 160, 160)
        if old_labels and image_id in old_labels and label is not None and old_labels[image_id] != label:
            color = (195, 50, 160)
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=color, width=4)

        text = image_id
        if label is not None:
            old_note = ""
            if old_labels and image_id in old_labels and old_labels[image_id] != label:
                old_note = f" old {old_labels[image_id]}->"
            text += f"  {old_note}{label}"
        draw.text((x + 8, y + cell + 4), text, fill=(220, 0, 0), font=FONT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def mean_brightness(path: Path) -> float:
    img = Image.open(path).convert("L")
    img.thumbnail((96, 96))
    pix = list(img.getdata())
    return sum(pix) / len(pix)


def main() -> None:
    current = read_labels(ROOT / "submission_0.757.csv")
    old = read_labels(ROOT / "submission_0.732.csv")
    train = read_labels(ROOT / "train_labels.csv")

    test_items = [(TEST_DIR / image_id, image_id, current[image_id]) for image_id in current]
    for page, start in enumerate(range(0, len(test_items), 25), start=1):
        make_sheet(
            test_items[start : start + 25],
            OUT / f"test_all_page_{page:02d}.jpg",
            title=f"test images page {page} - border green=1 gray=0 magenta=changed from 0.732",
            old_labels=old,
        )

    pos_items = [item for item in test_items if item[2] == 1]
    neg_items = [item for item in test_items if item[2] == 0]
    make_sheet(pos_items, OUT / "test_current_positive_0757.jpg", cols=7, cell=220, title="current positives")
    make_sheet(neg_items, OUT / "test_current_negative_0757.jpg", cols=7, cell=220, title="current negatives")
    make_sheet(
        [item for item in test_items if old[item[1]] != item[2]],
        OUT / "test_changed_0732_to_0757.jpg",
        title="changed from 0.732 to 0.757",
        old_labels=old,
    )

    rng = random.Random(31)
    pos_ids = [image_id for image_id, label in train.items() if label == 1]
    neg_ids = [image_id for image_id, label in train.items() if label == 0]
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    make_sheet(
        [(TRAIN_DIR / image_id, image_id, 1) for image_id in pos_ids[:70]],
        OUT / "train_positive_reference_random.jpg",
        cols=7,
        cell=220,
        title="train positive reference random",
    )
    make_sheet(
        [(TRAIN_DIR / image_id, image_id, 0) for image_id in neg_ids[:70]],
        OUT / "train_negative_reference_random.jpg",
        cols=7,
        cell=220,
        title="train negative reference random",
    )

    dark_pos = sorted(pos_ids, key=lambda image_id: mean_brightness(TRAIN_DIR / image_id))[:70]
    dark_neg = sorted(neg_ids, key=lambda image_id: mean_brightness(TRAIN_DIR / image_id))[:70]
    make_sheet(
        [(TRAIN_DIR / image_id, image_id, 1) for image_id in dark_pos],
        OUT / "train_positive_reference_dark.jpg",
        cols=7,
        cell=220,
        title="dark train positives",
    )
    make_sheet(
        [(TRAIN_DIR / image_id, image_id, 0) for image_id in dark_neg],
        OUT / "train_negative_reference_dark.jpg",
        cols=7,
        cell=220,
        title="dark train negatives",
    )

    # White-background versions make texture and shape easier to compare.
    make_sheet(
        [(TRAIN_WHITE_DIR / image_id, image_id, 1) for image_id in pos_ids[:70]],
        OUT / "train_positive_reference_white_random.jpg",
        cols=7,
        cell=220,
        title="train positive white-background reference",
    )
    make_sheet(
        [(TRAIN_WHITE_DIR / image_id, image_id, 0) for image_id in neg_ids[:70]],
        OUT / "train_negative_reference_white_random.jpg",
        cols=7,
        cell=220,
        title="train negative white-background reference",
    )

    with (OUT / "board_index.txt").open("w", encoding="utf-8") as f:
        for path in sorted(OUT.glob("*.jpg")):
            f.write(str(path) + "\n")


if __name__ == "__main__":
    main()
