import torch


def binary_classification_report(logits: torch.Tensor, labels: torch.Tensor):
    """
    Compute binary classification metrics: accuracy and F1.

    Notes:
    - logits: model output of shape [B, 2]
    - labels: ground-truth labels of shape [B]
    """
    preds = torch.argmax(logits, dim=1)

    correct = (preds == labels).sum().item()
    total = labels.numel()
    acc = correct / max(total, 1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "acc": acc,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
