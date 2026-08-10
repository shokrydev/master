"""Small alignment bridge for additive scene-level location conditioning."""

import math

import torch
from torch import nn


class AdditiveLocationProjection(nn.Module):
    """RMS-normalize location features and project them into Qwen space."""

    projection_type = "linear"
    normalization_type = "rms"

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        *,
        scale: float = 0.1,
        normalization_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a finite positive number")
        if not math.isfinite(normalization_eps) or normalization_eps <= 0:
            raise ValueError("normalization_eps must be a finite positive number")

        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.scale_value = float(scale)
        self.normalization_eps = float(normalization_eps)
        self.projection = nn.Linear(
            self.feature_dim,
            self.hidden_size,
            bias=False,
        )
        self.register_buffer(
            "scale",
            torch.tensor(self.scale_value, dtype=torch.float32),
        )

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "Location features must have shape "
                f"(batch, {self.feature_dim}), got {tuple(features.shape)}"
            )
        features = features.float()
        if not torch.isfinite(features).all():
            raise ValueError("Location features must contain only finite values")

        mean_square = features.square().mean(dim=-1, keepdim=True)
        if (mean_square <= self.normalization_eps).any():
            raise ValueError("Location features must have non-zero RMS")
        return features * torch.rsqrt(mean_square + self.normalization_eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_features(features)
        return self.scale * self.projection(normalized)

    def manifest(
        self,
        *,
        feature_source: str,
        scope: str,
        source_config: dict[str, object],
    ) -> dict[str, object]:
        return {
            "version": 1,
            "feature_source": feature_source,
            "scope": scope,
            "feature_dim": self.feature_dim,
            "hidden_size": self.hidden_size,
            "feature_normalization": self.normalization_type,
            "normalization_eps": self.normalization_eps,
            "projection_type": self.projection_type,
            "projection_bias": False,
            "scale": self.scale_value,
            "source_config": source_config,
        }
