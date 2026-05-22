"""Frozen BigEarthNet S1/S2 encoder wrapper."""

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

BIGEARTHNET_S1S2_10M20M_BANDS = [
    "VV",
    "VH",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

NonRGBFeatureMode = Literal["spatial_4x4", "pooled_prelogit"]


class BigEarthNetS1S2Encoder(nn.Module):
    """Frozen wrapper around the official BigEarthNet MobileViT encoder.

    The wrapper intentionally exposes encoder features, not BigEarthNet class
    logits. `spatial_4x4` preserves a fixed 4x4 grid of MobileViT features,
    while `pooled_prelogit` returns the encoder's pooled representation before
    the classifier layer.

    The local checkpoint loader uses the `model.vision_encoder.` subtree and
    loads it strictly into a timm MobileViT encoder.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        model: nn.Module | None = None,
        feature_mode: NonRGBFeatureMode = "spatial_4x4",
        spatial_pool_size: int = 4,
        expected_bands: Sequence[str] = BIGEARTHNET_S1S2_10M20M_BANDS,
    ) -> None:
        super().__init__()
        if feature_mode not in {"spatial_4x4", "pooled_prelogit"}:
            raise ValueError(f"Unsupported non-RGB feature mode: {feature_mode}")
        if spatial_pool_size <= 0:
            raise ValueError("spatial_pool_size must be positive")
        if model is None and model_dir is None:
            raise ValueError("Either model or model_dir must be provided")

        self.feature_mode = feature_mode
        self.spatial_pool_size = spatial_pool_size
        self.expected_bands = list(expected_bands)
        self.model = model if model is not None else self._load_model(model_dir)
        self.vision_encoder = self._get_vision_encoder(self.model)
        self.feature_dim = self._infer_feature_dim()

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _load_model(model_dir: str | Path | None) -> nn.Module:
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "BigEarthNetS1S2Encoder requires timm to load a local model_dir."
            ) from exc

        if model_dir is None:
            raise ValueError("model_dir is required when no model is injected")
        model_path = Path(model_dir)
        config_path = model_path / "config.json"
        weights_path = model_path / "model.safetensors"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                "BigEarthNetS1S2Encoder expected config.json and model.safetensors "
                f"under {model_path}"
            )

        with config_path.open() as f:
            config = json.load(f)

        vision_encoder = timm.create_model(
            config.get("timm_model_name", "mobilevit_s"),
            in_chans=int(config.get("channels", len(BIGEARTHNET_S1S2_10M20M_BANDS))),
            num_classes=int(config.get("classes", 19)),
            drop_rate=float(config.get("drop_rate", 0.0)),
            drop_path_rate=float(config.get("drop_path_rate", 0.0)),
        )
        state_dict = load_file(weights_path)
        non_vision_keys = [
            key for key in state_dict if not key.startswith("model.vision_encoder.")
        ]
        if non_vision_keys:
            warnings.warn(
                "BigEarthNetS1S2Encoder ignores non-vision checkpoint weights "
                f"such as {non_vision_keys[:5]}",
                stacklevel=2,
            )
        vision_state_dict = {
            key.removeprefix("model.vision_encoder."): value
            for key, value in state_dict.items()
            if key.startswith("model.vision_encoder.")
        }
        missing, unexpected = vision_encoder.load_state_dict(vision_state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load BigEarthNet vision encoder weights: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return vision_encoder

    @staticmethod
    def _get_vision_encoder(model: nn.Module) -> nn.Module:
        if hasattr(model, "forward_features"):
            return model
        if hasattr(model, "vision_encoder"):
            return model.vision_encoder
        inner_model = getattr(model, "model", None)
        if inner_model is not None and hasattr(inner_model, "vision_encoder"):
            return inner_model.vision_encoder
        raise AttributeError("BigEarthNet model does not expose a vision_encoder")

    def _infer_feature_dim(self) -> int | None:
        head = getattr(self.vision_encoder, "head", None)
        fc = getattr(head, "fc", None)
        if fc is not None and hasattr(fc, "in_features"):
            return int(fc.in_features)

        if hasattr(self.vision_encoder, "num_features"):
            return int(self.vision_encoder.num_features)

        return None

    def _validate_bands(self, bands: Sequence[str] | Sequence[Sequence[str]] | None) -> None:
        if bands is None:
            return
        if list(bands) == self.expected_bands:
            return
        if len(bands) > 0 and all(
            isinstance(item, Sequence) and not isinstance(item, str) for item in bands
        ):
            invalid = [list(item) for item in bands if list(item) != self.expected_bands]
            if not invalid:
                return
        raise ValueError(
            "BigEarthNetS1S2Encoder expected non_rgb_bands="
            f"{self.expected_bands}, got {bands}"
        )

    def _forward_features(self, imagery: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.vision_encoder, "forward_features"):
            raise AttributeError("vision_encoder does not expose forward_features")
        features = self.vision_encoder.forward_features(imagery)
        if not torch.is_tensor(features):
            raise TypeError("vision_encoder.forward_features must return a tensor")
        return features

    def _forward_pooled_prelogits(self, features: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.vision_encoder, "forward_head"):
            raise AttributeError("vision_encoder does not expose forward_head")
        pooled = self.vision_encoder.forward_head(features, pre_logits=True)
        if not torch.is_tensor(pooled):
            raise TypeError("vision_encoder.forward_head(..., pre_logits=True) must return a tensor")
        if pooled.ndim != 2:
            raise ValueError(f"Expected pooled pre-logit features with shape (B, D), got {tuple(pooled.shape)}")
        return pooled

    def forward(
        self,
        imagery: torch.Tensor,
        non_rgb_bands: Sequence[str] | Sequence[Sequence[str]] | None = None,
    ) -> torch.Tensor:
        """Encode normalized S1/S2 imagery into pooled or spatial features."""
        self._validate_bands(non_rgb_bands)
        if imagery.ndim != 4:
            raise ValueError(f"Expected non_rgb_imagery with shape (B, C, H, W), got {tuple(imagery.shape)}")
        if imagery.shape[1] != len(self.expected_bands):
            raise ValueError(
                f"Expected {len(self.expected_bands)} non-RGB channels, got {imagery.shape[1]}"
            )

        with torch.no_grad():
            features = self._forward_features(imagery)
            if self.feature_mode == "pooled_prelogit":
                pooled = self._forward_pooled_prelogits(features)
                self.feature_dim = pooled.shape[-1]
                return pooled

            if features.ndim != 4:
                raise ValueError(
                    "spatial_4x4 mode requires forward_features to return "
                    f"(B, C, H, W), got {tuple(features.shape)}"
                )
            pooled_grid = F.adaptive_avg_pool2d(
                features,
                output_size=(self.spatial_pool_size, self.spatial_pool_size),
            )
            tokens = pooled_grid.flatten(2).transpose(1, 2).contiguous()
            self.feature_dim = tokens.shape[-1]
            return tokens
