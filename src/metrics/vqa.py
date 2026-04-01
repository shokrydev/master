# Metrics for Visual Question Answering (VQA) Tasks

import re
from typing import List, Optional, Union

import torch
from torchmetrics import Metric


class VQAAccuracy(Metric):
    """
    VQA Accuracy metric following the VQA evaluation protocol.

    For each question, accuracy is computed as:
        min(1, #humans that provided that answer / 3)

    For simple evaluation without human annotations, this metric
    supports exact match and relaxed matching modes.

    Usage:
        metric = VQAAccuracy()
        metric.update(predictions=["cat"], targets=["cat"])
        accuracy = metric.compute()
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        normalize: bool = True,
        relaxed_match: bool = False,
        **kwargs,
    ):
        """
        Initialize VQA Accuracy metric.

        Args:
            normalize: Whether to normalize text (lowercase, strip whitespace)
            relaxed_match: If True, check if prediction contains target or vice versa
        """
        super().__init__(**kwargs)
        self.normalize = normalize
        self.relaxed_match = relaxed_match

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _match(self, prediction: str, target: str) -> bool:
        """Check if prediction matches target."""
        if self.normalize:
            prediction = self._normalize_text(prediction)
            target = self._normalize_text(target)

        if self.relaxed_match:
            return prediction in target or target in prediction
        return prediction == target

    def update(self, predictions: List[str], targets: List[str]):
        """
        Update metric state.

        Args:
            predictions: List of predicted answers
            targets: List of ground truth answers
        """
        for pred, target in zip(predictions, targets):
            if self._match(pred, target):
                self.correct += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        """Compute accuracy."""
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct / self.total


class VQAAccuracyMultiRef(Metric):
    """
    VQA Accuracy with multiple reference answers.

    Computes accuracy as the proportion of predictions that match
    at least one of the reference answers.

    Usage:
        metric = VQAAccuracyMultiRef()
        metric.update(
            predictions=["cat"],
            targets=[["cat", "kitten", "feline"]]
        )
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def update(self, predictions: List[str], targets: List[List[str]]):
        """
        Update metric state.

        Args:
            predictions: List of predicted answers
            targets: List of lists of acceptable answers
        """
        for pred, target_list in zip(predictions, targets):
            pred_norm = self._normalize_text(pred) if self.normalize else pred

            matched = False
            for target in target_list:
                target_norm = self._normalize_text(target) if self.normalize else target
                if pred_norm == target_norm:
                    matched = True
                    break

            if matched:
                self.correct += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct / self.total


class ExactMatchAccuracy(Metric):
    """
    Simple exact match accuracy for text generation tasks.

    Useful for tasks where the output should exactly match the target.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _normalize_text(self, text: str) -> str:
        return text.lower().strip()

    def update(self, predictions: List[str], targets: List[str]):
        for pred, target in zip(predictions, targets):
            if self.normalize:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)

            if pred == target:
                self.correct += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct / self.total


class TokenF1Score(Metric):
    """
    Token-level F1 score for text generation.

    Computes F1 based on token overlap between prediction and target.
    Useful for open-ended VQA where partial matches are meaningful.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

        self.add_state("f1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _tokenize(self, text: str) -> set:
        if self.normalize:
            text = text.lower().strip()
        tokens = re.findall(r'\w+', text)
        return set(tokens)

    def _compute_f1(self, pred_tokens: set, target_tokens: set) -> float:
        if len(pred_tokens) == 0 and len(target_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(target_tokens) == 0:
            return 0.0

        common = pred_tokens & target_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(target_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def update(self, predictions: List[str], targets: List[str]):
        for pred, target in zip(predictions, targets):
            pred_tokens = self._tokenize(pred)
            target_tokens = self._tokenize(target)
            f1 = self._compute_f1(pred_tokens, target_tokens)
            self.f1_sum += f1
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.f1_sum / self.total
