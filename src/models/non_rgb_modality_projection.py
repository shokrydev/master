"""Projection from non-RGB imagery encoder features to VLM token space."""

import torch
import torch.nn as nn


class NonRGBModalityProjection(nn.Module):
    """Projects frozen S1/S2 encoder features into the VLM hidden space."""

    def __init__(self, encoder_dim: int = 512, hidden_size: int = 2048, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * num_tokens),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return projected non-RGB imagery tokens with shape ``(B, num_tokens, hidden_size)``."""
        out = self.proj(features)
        return out.view(-1, self.num_tokens, self.hidden_size)
