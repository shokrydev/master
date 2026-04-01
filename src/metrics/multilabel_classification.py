# Metrics for multi-label classification tasks

from typing import Literal

import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelPrecision,
    MultilabelRecall,
)


class MultiLabelClassificationMetrics(Metric):
    """Collection of metrics for multi-label classification.

    Computes:
    - Subset Accuracy (exact match)
    - Hamming Loss
    - Precision (macro, micro)
    - Recall (macro, micro)
    - F1 Score (macro, micro)
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Literal["micro", "macro"] = "macro",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.add_state("exact_matches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        self.hamming = MultilabelHammingDistance(
            num_labels=num_labels,
            threshold=threshold,
        )
        self.precision = MultilabelPrecision(
            num_labels=num_labels,
            threshold=threshold,
            average=average,
        )
        self.recall = MultilabelRecall(
            num_labels=num_labels,
            threshold=threshold,
            average=average,
        )
        self.f1 = MultilabelF1Score(
            num_labels=num_labels,
            threshold=threshold,
            average=average,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update all metrics.

        Args:
            preds: Predictions, shape (N, L) probabilities or binary
            target: Ground truth labels, shape (N, L) binary
        """
        pred_labels = preds
        if pred_labels.dtype.is_floating_point:
            pred_labels = pred_labels >= self.threshold
        else:
            pred_labels = pred_labels.bool()
        target_labels = target.bool()

        self.exact_matches += (pred_labels == target_labels).all(dim=1).sum()
        self.total_samples += target_labels.shape[0]
        self.hamming.update(preds, target)
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.f1.update(preds, target)

    def compute(self) -> dict:
        subset_accuracy = torch.tensor(0.0, device=self.exact_matches.device)
        if self.total_samples > 0:
            subset_accuracy = self.exact_matches.float() / self.total_samples
        return {
            "subset_accuracy": subset_accuracy,
            "hamming_loss": self.hamming.compute(),
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "f1": self.f1.compute(),
        }

    def reset(self):
        self.exact_matches.zero_()
        self.total_samples.zero_()
        self.hamming.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()


class MeanAveragePrecision(Metric):
    """Mean Average Precision (mAP) for multi-label classification."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_labels: int, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.add_state("preds_list", default=[], dist_reduce_fx="cat")
        self.add_state("target_list", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Accumulate prediction probabilities and binary targets."""
        self.preds_list.append(preds)
        self.target_list.append(target)

    def compute(self) -> torch.Tensor:
        """Compute mean average precision across labels."""
        if len(self.preds_list) == 0:
            return torch.tensor(0.0)

        preds = torch.cat(self.preds_list, dim=0)
        target = torch.cat(self.target_list, dim=0)

        aps = []
        for i in range(self.num_labels):
            label_preds = preds[:, i]
            label_target = target[:, i]

            if label_target.sum() == 0:
                continue

            sorted_indices = torch.argsort(label_preds, descending=True)
            sorted_target = label_target[sorted_indices]

            tp = sorted_target.cumsum(dim=0)
            fp = (~sorted_target.bool()).cumsum(dim=0)
            precision = tp / (tp + fp)
            recall = tp / label_target.sum()

            recall_diff = torch.cat([recall[:1], recall[1:] - recall[:-1]])
            ap = (precision * recall_diff).sum()
            aps.append(ap)

        if len(aps) == 0:
            return torch.tensor(0.0)

        return torch.stack(aps).mean()
