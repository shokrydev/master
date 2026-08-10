import math

import torch
from torch import nn


class SceneLocationFeatures(nn.Module):
    """Parameter-free latitude/longitude sine-cosine features."""

    encoding_type = "prithvi_sincos_2d_v1"

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0 or feature_dim % 4 != 0:
            raise ValueError("feature_dim must be positive and divisible by four")

        self.feature_dim = int(feature_dim)
        coordinate_feature_size = self.feature_dim // 2
        frequency_count = coordinate_feature_size // 2
        frequencies = 1.0 / (
            10000.0
            ** (
                torch.arange(frequency_count, dtype=torch.float32)
                / float(frequency_count)
            )
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _encode_coordinate(self, coordinate: torch.Tensor) -> torch.Tensor:
        phases = coordinate.float().unsqueeze(-1) * self.frequencies.unsqueeze(0)
        return torch.cat([phases.sin(), phases.cos()], dim=-1)

    def forward(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("lat and lon must both have shape (batch,)")
        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have matching shapes")
        if not torch.isfinite(lat).all() or not torch.isfinite(lon).all():
            raise ValueError("lat and lon must contain only finite values")
        if (lat < -90).any() or (lat > 90).any():
            raise ValueError("latitude values must be within [-90, 90] degrees")
        if (lon < -180).any() or (lon > 180).any():
            raise ValueError("longitude values must be within [-180, 180] degrees")

        lat_encoding = self._encode_coordinate(lat)
        lon_encoding = self._encode_coordinate(lon)
        return torch.cat([lat_encoding, lon_encoding], dim=-1)


class SceneLocationEncoding(nn.Module):
    """Deterministic latitude/longitude sine-cosine encoding with one scale."""

    encoding_type = "prithvi_sincos_2d_v1"

    def __init__(
        self,
        hidden_size: int,
        *,
        scale_init: float = 0.1,
        learned_scale: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(scale_init) or scale_init <= 0:
            raise ValueError("scale_init must be a finite positive number")

        self.hidden_size = int(hidden_size)
        self.learned_scale = bool(learned_scale)
        self.scale_initialization = float(scale_init)
        self.features = SceneLocationFeatures(self.hidden_size)

        scale = torch.tensor(float(scale_init), dtype=torch.float32)
        if self.learned_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def forward(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        return self.scale * self.features(lat, lon)

    def manifest(self, *, scope: str) -> dict[str, object]:
        return {
            "version": 1,
            "encoding_type": self.encoding_type,
            "scope": scope,
            "hidden_size": self.hidden_size,
            "learned_scale": self.learned_scale,
            "scale_initialization": self.scale_initialization,
            "coordinate_order": ["latitude", "longitude"],
            "coordinate_units": "degrees",
            "coordinate_ranges": {
                "latitude": [-90.0, 90.0],
                "longitude": [-180.0, 180.0],
            },
        }
