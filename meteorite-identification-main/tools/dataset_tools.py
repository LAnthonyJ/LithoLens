# from pathlib import Path
# from typing import Optional

# import pandas as pd
# from PIL import Image, ImageOps
# from torch.utils.data import Dataset


# class MeteoriteDataset(Dataset):
#     """
#     通用数据集类。

#     使用方式:
#     - 训练/验证: dataframe 里需要有 label 列
#     - 测试推理: dataframe 可以没有 label 列
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

#         # EXIF 矫正 + RGB 统一，避免方向和通道格式问题
#         image = Image.open(img_path)
#         image = ImageOps.exif_transpose(image).convert("RGB")

#         if self.transforms is not None:
#             image = self.transforms(image)

#         if self.has_label:
#             return image, int(row["label"])
#         return image, str(img_path)


# def _resolve_image_dir(root: Path, split: str) -> Path:
#     """
#     同时兼容两种目录:
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
#     raise FileNotFoundError(f"找不到 {split} 图片目录，已尝试: {candidates}")


# def load_train_dataframe(data_root: Path) -> pd.DataFrame:
#     """读取训练标签并拼接图片完整路径。"""
#     data_root = Path(data_root)
#     csv_path = data_root / "train_labels.csv"
#     if not csv_path.exists():
#         raise FileNotFoundError(f"未找到训练标签文件: {csv_path}")

#     train_dir = _resolve_image_dir(data_root, "train")
#     df = pd.read_csv(csv_path)
#     if "id" not in df.columns or "label" not in df.columns:
#         raise ValueError("train_labels.csv 必须包含 id 和 label 两列")

#     df["img_path"] = df["id"].apply(lambda x: str(train_dir / x))

#     # 过滤掉路径不存在的样本，防止训练中断
#     exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
#     missing_count = int((~exists_mask).sum())
#     if missing_count > 0:
#         print(f"[警告] 有 {missing_count} 条训练样本找不到图片，将自动跳过")
#     df = df[exists_mask].copy()

#     df["label"] = df["label"].astype(int)
#     return df


# def load_test_dataframe(data_root: Path) -> pd.DataFrame:
#     """读取测试集 id，并拼接图片完整路径。"""
#     data_root = Path(data_root)
#     csv_path = data_root / "sample_submission.csv"
#     if not csv_path.exists():
#         raise FileNotFoundError(f"未找到 sample_submission.csv: {csv_path}")

#     test_dir = _resolve_image_dir(data_root, "test")
#     df = pd.read_csv(csv_path)
#     if "id" not in df.columns:
#         raise ValueError("sample_submission.csv 必须包含 id 列")

#     df = df[["id"]].copy()
#     df["img_path"] = df["id"].apply(lambda x: str(test_dir / x))

#     exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
#     missing_count = int((~exists_mask).sum())
#     if missing_count > 0:
#         print(f"[警告] 有 {missing_count} 条测试样本找不到图片，将自动跳过")
#     df = df[exists_mask].copy()

#     return df


# def stratified_sample_dataframe(
#     df: pd.DataFrame,
#     label_col: str,
#     sample_size: Optional[int],
#     seed: int = 42,
# ) -> pd.DataFrame:
#     """
#     按类别比例抽样，避免抽样后类别极度失衡。

#     - sample_size 为 None 或 <=0 时，返回原始 df
#     - sample_size 大于总样本数时，也返回原始 df
#     """
#     if sample_size is None or sample_size <= 0 or sample_size >= len(df):
#         return df.copy()

#     sampled_parts = []
#     class_counts = df[label_col].value_counts().sort_index()

#     # 先按比例计算每类最少抽取数量
#     allocated = {}
#     for cls, count in class_counts.items():
#         ratio = count / len(df)
#         allocated[cls] = max(1, int(round(sample_size * ratio)))

#     # 调整总数，使其精确等于 sample_size
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
#     按类别分层切分 train/val。
#     这样每个类别在训练集和验证集里都尽量有代表性。
#     """
#     if not (0.0 < val_ratio < 1.0):
#         raise ValueError("val_ratio 必须在 (0, 1) 范围内")

#     train_parts = []
#     val_parts = []

#     for cls, cls_df in df.groupby(label_col):
#         cls_df = cls_df.sample(frac=1.0, random_state=seed)
#         n_total = len(cls_df)
#         n_val = max(1, int(round(n_total * val_ratio)))

#         # 保证至少留 1 个样本在训练集
#         if n_val >= n_total:
#             n_val = n_total - 1

#         val_part = cls_df.iloc[:n_val]
#         train_part = cls_df.iloc[n_val:]
#         train_parts.append(train_part)
#         val_parts.append(val_part)

#     train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
#     val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
#     return train_df, val_df

####第二版：按重复图分组切分####
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
    通用数据集类。

    使用方式:
    - 训练/验证: dataframe 里需要有 label 列
    - 测试推理: dataframe 可以没有 label 列
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

        # EXIF 矫正 + RGB 统一，避免方向和通道格式问题
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        if self.has_label:
            return image, int(row["label"])
        return image, str(img_path)


def _resolve_image_dir(root: Path, split: str) -> Path:
    """
    同时兼容两种目录:
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
    raise FileNotFoundError(f"找不到 {split} 图片目录，已尝试: {candidates}")


def load_train_dataframe(data_root: Path) -> pd.DataFrame:
    """读取训练标签并拼接图片完整路径。"""
    data_root = Path(data_root)
    csv_path = data_root / "train_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到训练标签文件: {csv_path}")

    train_dir = _resolve_image_dir(data_root, "train")
    df = pd.read_csv(csv_path)
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("train_labels.csv 必须包含 id 和 label 两列")

    df["img_path"] = df["id"].apply(lambda x: str(train_dir / x))

    exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"[警告] 有 {missing_count} 条训练样本找不到图片，将自动跳过")
    df = df[exists_mask].copy()

    df["label"] = df["label"].astype(int)
    return df


def load_test_dataframe(data_root: Path) -> pd.DataFrame:
    """读取测试集 id，并拼接图片完整路径。"""
    data_root = Path(data_root)
    csv_path = data_root / "sample_submission.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 sample_submission.csv: {csv_path}")

    test_dir = _resolve_image_dir(data_root, "test")
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        raise ValueError("sample_submission.csv 必须包含 id 列")

    df = df[["id"]].copy()
    df["img_path"] = df["id"].apply(lambda x: str(test_dir / x))

    exists_mask = df["img_path"].apply(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"[警告] 有 {missing_count} 条测试样本找不到图片，将自动跳过")
    df = df[exists_mask].copy()

    return df


def stratified_sample_dataframe(
    df: pd.DataFrame,
    label_col: str,
    sample_size: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    按类别比例抽样，避免抽样后类别极度失衡。
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
    原始版本的分层随机切分。
    现在仍然保留，方便对照实验。
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio 必须在 (0, 1) 范围内")

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
    计算单个文件的 MD5。

    用途:
    - 找完全重复的图片
    - 把完全一样的图片放进同一个 group
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
    给 dataframe 增加 group_id 列。

    这里先用最稳妥的 exact duplicate 分组方式：
    - 同一个 md5 -> 同一组
    - 不同 md5 -> 不同组

    第一次跑会稍微慢一点，因为要对训练图片逐张算哈希。
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
    按 group 做 train / val 切分，避免完全重复图泄漏。

    逻辑:
    - 先按 label 分层
    - 每个类别内部再按 group 划分
    - 同一个 group 不允许同时出现在 train 和 val

    这不是完整的 GroupKFold，
    但已经比普通随机分层切分更可靠，适合作为第一步改进。
    """
    if group_col not in df.columns:
        raise ValueError(f"df 中缺少 {group_col} 列，请先调用 attach_md5_group_ids()")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio 必须在 (0, 1) 范围内")

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

        # 防止极端情况下某一类全部被分进 val
        if len(train_indices) == 0 and len(group_items) > 0:
            _, last_indices = group_items[-1]
            val_indices = [idx for idx in val_indices if idx not in last_indices]
            train_indices.extend(last_indices)

        train_parts.append(df.loc[train_indices])
        val_parts.append(df.loc[val_indices])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df
