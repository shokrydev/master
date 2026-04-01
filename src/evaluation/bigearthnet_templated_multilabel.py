# BigEarthNet-v2 prompted multilabel evaluation workflow

import re
from typing import Dict, List, Optional

import torch

from src.metrics.multilabel_classification import MultiLabelClassificationMetrics

# BigEarthNet 19-class CORINE Land Cover labels (canonical names)
BIGEARTHNET_19_LABELS = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# Common short-forms and aliases -> canonical label
BIGEARTHNET_19_ALIASES: Dict[str, str] = {
    "urban": "Urban fabric",
    "industrial": "Industrial or commercial units",
    "commercial": "Industrial or commercial units",
    "arable": "Arable land",
    "crops": "Permanent crops",
    "permanent crop": "Permanent crops",
    "pasture": "Pastures",
    "cultivation": "Complex cultivation patterns",
    "complex cultivation": "Complex cultivation patterns",
    "agriculture": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "agro-forestry": "Agro-forestry areas",
    "agroforestry": "Agro-forestry areas",
    "broadleaf forest": "Broad-leaved forest",
    "broad-leaved": "Broad-leaved forest",
    "coniferous": "Coniferous forest",
    "mixed forest": "Mixed forest",
    "grassland": "Natural grassland and sparsely vegetated areas",
    "natural grassland": "Natural grassland and sparsely vegetated areas",
    "sparsely vegetated": "Natural grassland and sparsely vegetated areas",
    "moors": "Moors, heathland and sclerophyllous vegetation",
    "heathland": "Moors, heathland and sclerophyllous vegetation",
    "sclerophyllous": "Moors, heathland and sclerophyllous vegetation",
    "transitional woodland": "Transitional woodland, shrub",
    "shrub": "Transitional woodland, shrub",
    "beaches": "Beaches, dunes, sands",
    "dunes": "Beaches, dunes, sands",
    "inland wetland": "Inland wetlands",
    "inland wetlands": "Inland wetlands",
    "coastal wetland": "Coastal wetlands",
    "coastal wetlands": "Coastal wetlands",
    "inland water": "Inland waters",
    "inland waters": "Inland waters",
    "marine water": "Marine waters",
    "marine waters": "Marine waters",
}


class BigEarthNetMultilabelEvaluator:
    """Dataset-specific workflow for prompted BigEarthNet multilabel evaluation.

    This module owns the task adapter for BigEarthNet:
    - label vocabulary
    - alias handling
    - generated-text parsing
    - target vectorization
    - metric invocation

    The underlying metric computations remain in `src.metrics.multilabel_classification`.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
    ):
        self.labels = labels or BIGEARTHNET_19_LABELS
        self.aliases = aliases or BIGEARTHNET_19_ALIASES
        self.label_to_idx = {name: i for i, name in enumerate(self.labels)}
        self._matchers = self._build_matchers()

    def _build_matchers(self) -> List[tuple]:
        entries: List[tuple] = []
        for name in self.labels:
            entries.append((name, self.label_to_idx[name]))
        for alias, canonical in self.aliases.items():
            if canonical in self.label_to_idx:
                entries.append((alias, self.label_to_idx[canonical]))

        entries.sort(key=lambda x: len(x[0]), reverse=True)

        matchers = []
        for pattern_str, idx in entries:
            regex = re.compile(r"\b" + re.escape(pattern_str) + r"\b", re.IGNORECASE)
            matchers.append((regex, idx))
        return matchers

    def parse_prediction(self, text: str) -> torch.Tensor:
        """Convert generated text into a binary BigEarthNet label vector."""
        vector = torch.zeros(len(self.labels))
        for regex, idx in self._matchers:
            if regex.search(text):
                vector[idx] = 1.0
        return vector

    def vectorize_target(self, label_names: List[str]) -> torch.Tensor:
        """Convert canonical label names into a binary target vector."""
        vector = torch.zeros(len(self.labels))
        for name in label_names:
            if name in self.label_to_idx:
                vector[self.label_to_idx[name]] = 1.0
        return vector

    def evaluate(
        self,
        predictions: List[str],
        targets: List[List[str]],
    ) -> Dict[str, torch.Tensor]:
        """Evaluate prompted BigEarthNet predictions with multilabel metrics."""
        if not predictions:
            zero = torch.tensor(0.0)
            return {
                "micro/subset_accuracy": zero,
                "micro/hamming_loss": zero,
                "micro/precision": zero,
                "micro/recall": zero,
                "micro/f1": zero,
                "macro/subset_accuracy": zero,
                "macro/hamming_loss": zero,
                "macro/precision": zero,
                "macro/recall": zero,
                "macro/f1": zero,
            }

        pred_vectors = torch.stack([self.parse_prediction(text) for text in predictions])
        target_vectors = torch.stack([self.vectorize_target(label_names) for label_names in targets])

        micro_metrics = MultiLabelClassificationMetrics(
            num_labels=len(self.labels),
            average="micro",
        )
        micro_metrics.update(pred_vectors, target_vectors)
        micro_scores = micro_metrics.compute()

        macro_metrics = MultiLabelClassificationMetrics(
            num_labels=len(self.labels),
            average="macro",
        )
        macro_metrics.update(pred_vectors, target_vectors)
        macro_scores = macro_metrics.compute()

        scores = {}
        for name, value in micro_scores.items():
            scores[f"micro/{name}"] = value
        for name, value in macro_scores.items():
            scores[f"macro/{name}"] = value
        return scores
