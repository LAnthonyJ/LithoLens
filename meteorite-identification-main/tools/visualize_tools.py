from pathlib import Path

import matplotlib

# 在无图形界面的环境中保存图片，避免报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_label_distribution(df, output_path):
    """
    绘制类别分布柱状图。

    这个函数主要用于训练前的数据检查：
    - 看类别是否平衡
    - 看 train / val 的切分是否合理
    """
    counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar([str(k) for k in counts.index], counts.values)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_sample_grid(dataset, output_path, max_images=16):
    """
    保存样本网格图，用来快速检查图片和标签是否读对了。

    参数:
    - dataset: 需要能返回 (image_tensor, label)
    - output_path: 图片保存路径
    - max_images: 最多展示多少张样本
    """
    max_images = min(max_images, len(dataset))
    cols = 4
    rows = (max_images + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(max_images):
        image, label = dataset[i]

        # Dataset 返回的是标准化后的 Tensor。
        # 为了方便显示，这里做一个近似反归一化。
        img = image.permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"label={label}")
        ax.axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_curves(history, output_path):
    """
    根据每个 epoch 的训练记录，绘制训练过程折线图。

    生成的图包含 4 个子图：
    - Loss
    - Accuracy
    - F1 Score
    - Learning Rate

    其中 history 应该是一个列表，列表中的每个元素是一个字典，
    至少包含以下字段：
    epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, lr
    """
    if not history:
        raise ValueError("history 为空，无法绘制训练曲线")

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    train_f1 = [row["train_f1"] for row in history]
    val_f1 = [row["val_f1"] for row in history]
    lrs = [row.get("lr", 0.0) for row in history]

    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, train_acc, label="Train Accuracy")
    ax2.plot(epochs, val_acc, label="Val Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(epochs, train_f1, label="Train F1")
    ax3.plot(epochs, val_f1, label="Val F1")
    ax3.set_title("F1 Score")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(epochs, lrs, label="Learning Rate")
    ax4.set_title("Learning Rate")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
