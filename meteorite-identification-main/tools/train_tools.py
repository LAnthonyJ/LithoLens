import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def set_seed(seed: int = 42):
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_bce_pos_weight(class_counts, device=None):
    """根据各类别样本数，计算 BCEWithLogitsLoss 所需的 pos_weight。"""
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if device is not None:
        counts = counts.to(device)

    negatives = counts.sum() - counts
    return negatives / counts.clamp_min(1.0)


class IntegerLabelBCEWithLogitsLoss(nn.Module):
    """让 BCEWithLogitsLoss 兼容整数类别标签。"""

    def __init__(self, pos_weight=None, reduction="mean"):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits, labels):
        if logits.ndim != 2:
            raise ValueError("BCEWithLogitsLoss 需要二维 logits，形状应为 [batch_size, num_classes]")

        targets = F.one_hot(labels.to(torch.long), num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return self.loss_fn(logits, targets)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证集上评估 loss/acc/f1。"""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": acc,
        "f1": f1,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练 1 个 epoch，返回该 epoch 的平均指标。"""
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.detach().cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": acc,
        "f1": f1,
    }
