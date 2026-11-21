# from pathlib import Path
# from typing import Optional

# import pandas as pd
# from PIL import Image, ImageOps
# from torch.utils.data import Dataset


# class MeteoriteDataset(Dataset):
#     """
#     Generic image dataset.

#     Usage:
#     - Training and validation require a label column
#     - Test inference accepts unlabeled rows
#     """

#     def __init__(self, dataframe: pd.DataFrame, transforms=None):
#         self.df = dataframe.reset_index(drop=True).copy()
#         self.transforms = transforms
#         self.has_label = "label" in self.df.columns

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         row = self.df.iloc[index]
#         img_path = row["img_path"]

#         # EXIF orientation correction and RGB conversion
#         image = Image.open(img_path)
#         image = ImageOps.exif_transpose(image).convert("RGB")

#         if self.transforms is not None:
#             image = self.transforms(image)

#         if self.has_label:
#             return image, int(row["label"])
#         return image, str(img_path)


# def _resolve_image_dir(root: Path, split: str) -> Path:
#     """
#     Support both directory layouts:
#     1) split_images/split_images/*.jpg
#     2) split_images/*.jpg
#     """
#     candidates = [
#         root / f"{split}_images" / f"{split}_images",
#         root / f"{split}_images",
#     ]
#     for candidate in candidates:
#         if candidate.is_dir():
#             return candidate
#     raise FileNotFoundError(f"Missing {split} image directory; tried: {candidates}")


# def load_train_dataframe(data_root: Path) -> pd.DataFrame:
#     """Read training labels and resolve image paths."""
#     data_root = Path(data_root)
#     csv_path = data_root / "train_labels.csv"
#     if not csv_path.exists():
#         raise FileNotFoundError(f"Training labels missing: {csv_path}")

#     train_dir = _resolve_image_dir(data_root, "train")
#     df = pd.read_csv(csv_path)
#     if "id" not in df.columns or "label" not in df.columns:
#         raise ValueError("train_labels.csv must contain id and label columns")

#     df["img_path"] = df["id"].apply(lambda x: str(train_dir / x))

#     # Skip missing images before training
#     exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
#     missing_count = int((~exists_mask).sum())
#     if missing_count > 0:
#         print(f"[Warning] {missing_count} training images missing; skipping")
#     df = df[exists_mask].copy()

#     df["label"] = df["label"].astype(int)
#     return df


# def load_test_dataframe(data_root: Path) -> pd.DataFrame:
#     """Read test IDs and resolve image paths."""
#     data_root = Path(data_root)
#     csv_path = data_root / "sample_submission.csv"
#     if not csv_path.exists():
#         raise FileNotFoundError(f"Missing sample_submission.csv: {csv_path}")

#     test_dir = _resolve_image_dir(data_root, "test")
#     df = pd.read_csv(csv_path)
#     if "id" not in df.columns:
#         raise ValueError("sample_submission.csv must contain an id column")

#     df = df[["id"]].copy()
#     df["img_path"] = df["id"].apply(lambda x: str(test_dir / x))

#     exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
#     missing_count = int((~exists_mask).sum())
#     if missing_count > 0:
#         print(f"[Warning] {missing_count} test images missing; skipping")
#     df = df[exists_mask].copy()

#     return df


# def stratified_sample_dataframe(
#     df: pd.DataFrame,
#     label_col: str,
#     sample_size: Optional[int],
#     seed: int = 42,
# ) -> pd.DataFrame:
#     """
#     Sample in proportion to class size.

#     - sample_size of None or <=0 returns the original df
#     - sample_size larger than the dataset also returns the original df
#     """
#     if sample_size is None or sample_size <= 0 or sample_size >= len(df):
#         return df.copy()

#     sampled_parts = []
#     class_counts = df[label_col].value_counts().sort_index()

#     # Allocate samples to each class proportionally
#     allocated = {}
#     for cls, count in class_counts.items():
#         ratio = count / len(df)
#         allocated[cls] = max(1, int(round(sample_size * ratio)))

#     # Adjust the total to match sample_size
#     total_alloc = sum(allocated.values())
#     classes = list(class_counts.index)
#     idx = 0
#     while total_alloc > sample_size:
#         cls = classes[idx % len(classes)]
#         if allocated[cls] > 1:
#             allocated[cls] -= 1
#             total_alloc -= 1
#         idx += 1
#     idx = 0
#     while total_alloc < sample_size:
#         cls = classes[idx % len(classes)]
#         if allocated[cls] < int(class_counts[cls]):
#             allocated[cls] += 1
#             total_alloc += 1
#         idx += 1

#     for cls, n in allocated.items():
#         cls_df = df[df[label_col] == cls]
#         n = min(n, len(cls_df))
#         sampled_parts.append(cls_df.sample(n=n, random_state=seed))

#     sampled_df = pd.concat(sampled_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
#     return sampled_df


# def stratified_train_val_split(
#     df: pd.DataFrame,
#     label_col: str,
#     val_ratio: float = 0.2,
#     seed: int = 42,
# ):
#     """
#     Stratify by class when splitting train/val.
#     Keep each class represented in both partitions.
#     """
#     if not (0.0 < val_ratio < 1.0):
#         raise ValueError("val_ratio must be within (0, 1)")

#     train_parts = []
#     val_parts = []

#     for cls, cls_df in df.groupby(label_col):
#         cls_df = cls_df.sample(frac=1.0, random_state=seed)
#         n_total = len(cls_df)
#         n_val = max(1, int(round(n_total * val_ratio)))

#         # Keep at least one training sample per class
#         if n_val >= n_total:
#             n_val = n_total - 1

#         val_part = cls_df.iloc[:n_val]
#         train_part = cls_df.iloc[n_val:]
#         train_parts.append(train_part)
#         val_parts.append(val_part)

#     train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
#     val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
#     return train_df, val_df

####Version 2: split by duplicate group####
from pathlib import Path
from typing import Optional
import hashlib
import random

import pandas as pd
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MeteoriteDataset(Dataset):
    """
    Generic image dataset.

    Usage:
    - Training and validation require a label column
    - Test inference accepts unlabeled rows
    """

    def __init__(self, dataframe: pd.DataFrame, transforms=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transforms = transforms
        self.has_label = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row["img_path"]

        # EXIF orientation correction and RGB conversion
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        if self.has_label:
            return image, int(row["label"])
        return image, str(img_path)


def _resolve_image_dir(root: Path, split: str) -> Path:
    """
    Support both directory layouts:
    1) split_images/split_images/*.jpg
    2) split_images/*.jpg
    """
    candidates = [
        root / f"{split}_images" / f"{split}_images",
        root / f"{split}_images",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing {split} image directory; tried: {candidates}")


def load_train_dataframe(data_root: Path) -> pd.DataFrame:
    """Read training labels and resolve image paths."""
    data_root = Path(data_root)
    csv_path = data_root / "train_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training labels missing: {csv_path}")

    train_dir = _resolve_image_dir(data_root, "train")
    df = pd.read_csv(csv_path)
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("train_labels.csv must contain id and label columns")

    df["img_path"] = df["id"].apply(lambda x: str(train_dir / x))

    exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"[Warning] {missing_count} training images missing; skipping")
    df = df[exists_mask].copy()

    df["label"] = df["label"].astype(int)
    return df


def load_test_dataframe(data_root: Path) -> pd.DataFrame:
    """Read test IDs and resolve image paths."""
    data_root = Path(data_root)
    csv_path = data_root / "sample_submission.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sample_submission.csv: {csv_path}")

    test_dir = _resolve_image_dir(data_root, "test")
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        raise ValueError("sample_submission.csv must contain an id column")

    df = df[["id"]].copy()
    df["img_path"] = df["id"].apply(lambda x: str(test_dir / x))

    exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"[Warning] {missing_count} test images missing; skipping")
    df = df[exists_mask].copy()

    return df


def stratified_sample_dataframe(
    df: pd.DataFrame,
    label_col: str,
    sample_size: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample in proportion to class size.
    """
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.copy()

    sampled_parts = []
    class_counts = df[label_col].value_counts().sort_index()

    allocated = {}
    for cls, count in class_counts.items():
        ratio = count / len(df)
        allocated[cls] = max(1, int(round(sample_size * ratio)))

    total_alloc = sum(allocated.values())
    classes = list(class_counts.index)

    idx = 0
    while total_alloc > sample_size:
        cls = classes[idx % len(classes)]
        if allocated[cls] > 1:
            allocated[cls] -= 1
            total_alloc -= 1
        idx += 1

    idx = 0
    while total_alloc < sample_size:
        cls = classes[idx % len(classes)]
        if allocated[cls] < int(class_counts[cls]):
            allocated[cls] += 1
            total_alloc += 1
        idx += 1

    for cls, n in allocated.items():
        cls_df = df[df[label_col] == cls]
        n = min(n, len(cls_df))
        sampled_parts.append(cls_df.sample(n=n, random_state=seed))

    sampled_df = pd.concat(sampled_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled_df


def stratified_train_val_split(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Original stratified random split.
    Retained for comparison experiments.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be within (0, 1)")

    train_parts = []
    val_parts = []

    for cls, cls_df in df.groupby(label_col):
        cls_df = cls_df.sample(frac=1.0, random_state=seed)
        n_total = len(cls_df)
        n_val = max(1, int(round(n_total * val_ratio)))

        if n_val >= n_total:
            n_val = n_total - 1

        val_part = cls_df.iloc[:n_val]
        train_part = cls_df.iloc[n_val:]
        train_parts.append(train_part)
        val_parts.append(val_part)

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def compute_file_md5(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute a file's MD5.

    Purpose:
    - Find byte-identical images
    - Assign identical files to the same group
    """
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def attach_md5_group_ids(
    df: pd.DataFrame,
    path_col: str = "img_path",
    group_col: str = "group_id",
) -> pd.DataFrame:
    """
    Add a group_id column to the dataframe.

    Group exact duplicates by file content:
    - Same MD5 -> same group
    - Different MD5 -> different groups

    The first pass hashes each training image.
    """
    df = df.copy()
    df[group_col] = df[path_col].apply(compute_file_md5)
    return df


def group_aware_train_val_split(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "group_id",
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Split by group to keep duplicate images in one partition.

    Procedure:
    - Stratify by label
    - Split the groups within each class
    - Each group belongs to only one partition: train or val

    This is a partial implementation of GroupKFold,
    Grouping exact duplicates addresses one source of leakage.
    """
    if group_col not in df.columns:
        raise ValueError(f"df is missing the {group_col} column; first call attach_md5_group_ids()")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be within (0, 1)")

    rng = random.Random(seed)
    train_parts = []
    val_parts = []

    for cls, cls_df in df.groupby(label_col):
        group_items = []
        for gid, gdf in cls_df.groupby(group_col):
            group_items.append((gid, gdf.index.tolist()))

        rng.shuffle(group_items)

        target_val_count = max(1, int(round(len(cls_df) * val_ratio)))
        val_indices = []
        train_indices = []
        current_val_count = 0

        for _, indices in group_items:
            if current_val_count < target_val_count:
                val_indices.extend(indices)
                current_val_count += len(indices)
            else:
                train_indices.extend(indices)

        # Avoid assigning an entire class to val
        if len(train_indices) == 0 and len(group_items) > 0:
            _, last_indices = group_items[-1]
            val_indices = [idx for idx in val_indices if idx not in last_indices]
            train_indices.extend(last_indices)

        train_parts.append(df.loc[train_indices])
        val_parts.append(df.loc[val_indices])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df
