"""Single-label classification metrics; missing-class F1 is defined as zero."""

import torch
from torch import Tensor


def classification_metrics(predicted: Tensor, target: Tensor, num_classes: int) -> dict[str, float]:
    predicted, target = predicted.cpu(), target.cpu()
    if predicted.ndim != 1 or predicted.shape != target.shape or target.numel() == 0:
        raise ValueError("predicted and target must be matching nonempty vectors")
    if predicted.dtype != torch.long or target.dtype != torch.long:
        raise ValueError("class labels must be int64")
    if num_classes < 2 or any(
        torch.any((labels < 0) | (labels >= num_classes)) for labels in (predicted, target)
    ):
        raise ValueError("class labels outside configured range")
    confusion = (
        torch.bincount(target * num_classes + predicted, minlength=num_classes**2)
        .reshape(num_classes, num_classes)
        .double()
    )
    true_positive = confusion.diag()
    support = confusion.sum(1)
    denominator = support + confusion.sum(0)
    f1 = torch.where(denominator > 0, 2 * true_positive / denominator, 0.0)
    accuracy = float(true_positive.sum() / confusion.sum())
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "micro_f1": accuracy,
        "weighted_f1": float((f1 * support).sum() / support.sum()),
    }
