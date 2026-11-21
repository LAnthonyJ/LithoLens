from pathlib import Path

import matplotlib

# Save plots in headless environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_label_distribution(df, output_path):
    """
    Plot class counts.

    Inspect data before training:
    - Check class balance
    - Compare training and validation distributions
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
    Save a grid for checking images and labels.

    settings:
    - dataset: must return (image_tensor, label)
    - output_path: output image path
    - max_images: maximum samples to show
    """
    max_images = min(max_images, len(dataset))
    cols = 4
    rows = (max_images + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(max_images):
        image, label = dataset[i]

        # Dataset returns normalized Tensor.
        # Approximately undo normalization for display.
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
    Plot training curves from per-epoch records.

    The figure has four panels:
    - Loss
    - Accuracy
    - F1 Score
    - Learning Rate

    Pass history as a list of dictionaries,
    with at least these fields:
    epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, lr
    """
    if not history:
        raise ValueError("history is empty; cannot plot training curves")

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
