import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "domain_diagnosis_20260602"


def resolve_nested_dir(path: Path, first_id: str) -> Path:
    if (path / first_id).exists():
        return path
    nested = path / path.name
    if (nested / first_id).exists():
        return nested
    for child in path.iterdir():
        if child.is_dir() and (child / first_id).exists():
            return child
    raise FileNotFoundError(f"Could not resolve {first_id} under {path}")


def image_features(path: Path) -> dict:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        small = rgb.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)

    border = max(2, int(round(min(arr.shape[:2]) * 0.04)))
    border_pixels = np.concatenate(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border_pixels, axis=0)
    diff = np.mean(np.abs(arr - bg), axis=2)
    gray = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    mask = (diff > 18.0) | ((saturation > 35.0) & (gray < 245.0))
    object_area_ratio = float(mask.mean())
    if mask.any():
        ys, xs = np.where(mask)
        bbox_w = float((xs.max() - xs.min() + 1) / mask.shape[1])
        bbox_h = float((ys.max() - ys.min() + 1) / mask.shape[0])
        bbox_area_ratio = float(bbox_w * bbox_h)
        center_x = float((xs.min() + xs.max() + 1) / 2 / mask.shape[1])
        center_y = float((ys.min() + ys.max() + 1) / 2 / mask.shape[0])
    else:
        bbox_w = bbox_h = bbox_area_ratio = 0.0
        center_x = center_y = 0.5

    return {
        "width": int(width),
        "height": int(height),
        "aspect": float(width / max(1, height)),
        "file_size_kb": float(path.stat().st_size / 1024.0),
        "mean_luma": float(gray.mean()),
        "std_luma": float(gray.std()),
        "mean_saturation": float(saturation.mean()),
        "bg_luma": float(bg.mean()),
        "bg_saturation": float(bg.max() - bg.min()),
        "border_white_frac": float((border_pixels.mean(axis=1) > 240).mean()),
        "border_black_frac": float((border_pixels.mean(axis=1) < 35).mean()),
        "object_area_ratio": object_area_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_width_ratio": bbox_w,
        "bbox_height_ratio": bbox_h,
        "center_offset": float(((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5),
    }


def dhash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as img:
        gray = img.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
    arr = np.asarray(gray, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return int(value)


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def summarize_numeric(df: pd.DataFrame, columns: list[str]) -> dict:
    summary = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        summary[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p10": float(series.quantile(0.10)),
            "p90": float(series.quantile(0.90)),
        }
    return summary


def write_board(rows, image_dir: Path, out_path: Path, title: str, cols: int = 5) -> None:
    rows = list(rows)
    if not rows:
        return
    cell_w, cell_h = 260, 235
    label_h = 34
    rows_n = int(np.ceil(len(rows) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows_n * cell_h + 44), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, row in enumerate(rows):
        x0 = (i % cols) * cell_w
        y0 = (i // cols) * cell_h + 44
        path = image_dir / str(row["id"])
        with Image.open(path) as img:
            img = ImageOps.contain(img.convert("RGB"), (cell_w - 8, cell_h - label_h - 8))
        canvas.paste(img, (x0 + (cell_w - img.width) // 2, y0 + 4))
        color = (30, 130, 70) if int(row.get("v13_label", row.get("label", 0))) == 1 else (120, 120, 120)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=color, width=4)
        text = str(row.get("caption", row["id"]))
        draw.text((x0 + 8, y0 + cell_h - label_h + 7), text[:42], fill=(160, 20, 20), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(ROOT / "train_labels.csv")
    sample_df = pd.read_csv(ROOT / "sample_submission.csv")
    train_dir = resolve_nested_dir(ROOT / "train_images", str(train_df.iloc[0]["id"]))
    test_dir = resolve_nested_dir(ROOT / "test_images", str(sample_df.iloc[0]["id"]))

    train_rows = []
    for _, row in train_df.iterrows():
        img_id = str(row["id"])
        path = train_dir / img_id
        train_rows.append(
            {
                "split": "train",
                "id": img_id,
                "label": int(row["label"]),
                **image_features(path),
                "dhash": dhash(path),
            }
        )

    v13_path = Path("D:/v13.csv")
    v13_map = {}
    if v13_path.exists():
        v13_df = pd.read_csv(v13_path)
        v13_map = dict(zip(v13_df["id"].astype(str), v13_df["label"].astype(int)))

    test_rows = []
    for _, row in sample_df.iterrows():
        img_id = str(row["id"])
        path = test_dir / img_id
        test_rows.append(
            {
                "split": "test",
                "id": img_id,
                "v13_label": v13_map.get(img_id, None),
                **image_features(path),
                "dhash": dhash(path),
            }
        )

    train_features = pd.DataFrame(train_rows)
    test_features = pd.DataFrame(test_rows)

    train_hashes = train_features[["id", "label", "dhash"]].to_dict("records")
    nearest_rows = []
    for row in test_features.to_dict("records"):
        distances = [
            (hamming(int(row["dhash"]), int(t["dhash"])), t["id"], int(t["label"]))
            for t in train_hashes
        ]
        distances.sort(key=lambda x: (x[0], x[1]))
        nearest_rows.append(
            {
                "id": row["id"],
                "nearest_train_dhash_distance": distances[0][0],
                "nearest_train_id": distances[0][1],
                "nearest_train_label": distances[0][2],
            }
        )
    nearest_df = pd.DataFrame(nearest_rows)
    test_features = test_features.merge(nearest_df, on="id", how="left")

    submissions = [
        "submission_0.732.csv",
        "submission_0.757.csv",
        "submission_0.76571.csv",
        "submission_visual_v5.csv",
        "outputs/final_submission/submission.csv",
        "outputs/stage2/submission.csv",
        "outputs/stage2_effv2/submission.csv",
    ]
    for sub in submissions:
        path = ROOT / sub
        if not path.exists():
            continue
        sub_df = pd.read_csv(path)
        col = sub.replace("/", "_").replace("\\", "_").replace(".csv", "").replace(".", "_")
        sub_map = dict(zip(sub_df["id"].astype(str), sub_df["label"].astype(int)))
        test_features[col] = test_features["id"].map(sub_map)
        if v13_map:
            test_features[f"{col}_diff_v13"] = (
                test_features[col].astype("Int64") != test_features["v13_label"].astype("Int64")
            )

    numeric_cols = [
        "width",
        "height",
        "aspect",
        "file_size_kb",
        "mean_luma",
        "std_luma",
        "mean_saturation",
        "bg_luma",
        "border_white_frac",
        "border_black_frac",
        "object_area_ratio",
        "bbox_area_ratio",
        "center_offset",
        "nearest_train_dhash_distance",
    ]
    train_numeric_cols = [c for c in numeric_cols if c in train_features.columns]
    test_numeric_cols = [c for c in numeric_cols if c in test_features.columns]

    summary = {
        "train_count": int(len(train_features)),
        "test_count": int(len(test_features)),
        "train_label_counts": {
            str(k): int(v) for k, v in train_features["label"].value_counts().sort_index().items()
        },
        "v13_label_counts": {
            str(k): int(v)
            for k, v in test_features["v13_label"].value_counts(dropna=True).sort_index().items()
        },
        "train_overall": summarize_numeric(train_features, train_numeric_cols),
        "test_overall": summarize_numeric(test_features, test_numeric_cols),
        "train_by_label": {
            str(label): summarize_numeric(part, train_numeric_cols)
            for label, part in train_features.groupby("label")
        },
    }
    if v13_map:
        summary["test_by_v13_label"] = {
            str(label): summarize_numeric(part, test_numeric_cols)
            for label, part in test_features.groupby("v13_label")
        }
        diff_col = "submission_0_757_diff_v13"
        if diff_col in test_features.columns:
            summary["v13_vs_submission_0757_diff_count"] = int(test_features[diff_col].sum())
            summary["v13_vs_submission_0757_diff_by_direction"] = {
                "v13_1_sub_0": int(
                    ((test_features["v13_label"] == 1) & (test_features["submission_0_757"] == 0)).sum()
                ),
                "v13_0_sub_1": int(
                    ((test_features["v13_label"] == 0) & (test_features["submission_0_757"] == 1)).sum()
                ),
            }

    train_features.to_csv(OUT_DIR / "train_image_features.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    test_features.to_csv(OUT_DIR / "test_image_features.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with (OUT_DIR / "image_domain_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if v13_map and "submission_0_757" in test_features.columns:
        diff = test_features[test_features["submission_0_757_diff_v13"]].copy()
        diff["caption"] = diff.apply(
            lambda r: f'{r["id"]} v13={int(r["v13_label"])} sub757={int(r["submission_0_757"])}',
            axis=1,
        )
        write_board(
            diff.to_dict("records"),
            test_dir,
            OUT_DIR / "v13_vs_submission_0757_diffs.jpg",
            "v13 vs submission_0.757 differences",
            cols=4,
        )

    near = test_features.sort_values("nearest_train_dhash_distance").head(32).copy()
    near["caption"] = near.apply(
        lambda r: f'{r["id"]} near {r["nearest_train_id"]} d={int(r["nearest_train_dhash_distance"])}',
        axis=1,
    )
    write_board(
        near.to_dict("records"),
        test_dir,
        OUT_DIR / "test_nearest_train_by_dhash.jpg",
        "test images with closest train dhash neighbors",
        cols=4,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False)[:6000])
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
