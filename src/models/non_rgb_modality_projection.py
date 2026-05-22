"""Projection from non-RGB imagery encoder features to VLM token space."""

import torch
import torch.nn as nn


class NonRGBModalityProjection(nn.Module):
    """Projects frozen S1/S2 encoder features into the VLM hidden space."""

    def __init__(self, encoder_dim: int, hidden_size: int = 2048, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.pooled_proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * num_tokens),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return projected non-RGB imagery tokens.

        Accepts either pooled encoder features with shape ``(B, D)`` or a fixed
        sequence of encoder tokens with shape ``(B, num_tokens, D)``.
        """
        if features.ndim == 2:
            out = self.pooled_proj(features)
            return out.view(-1, self.num_tokens, self.hidden_size)

        if features.ndim == 3:
            if features.shape[1] != self.num_tokens:
                raise ValueError(
                    "Spatial non-RGB features must have shape "
                    f"(B, {self.num_tokens}, D), got {tuple(features.shape)}"
                )
            return self.token_proj(features)

        raise ValueError(
            "Non-RGB encoder features must have shape (B, D) or "
            f"(B, {self.num_tokens}, D), got {tuple(features.shape)}"
        )
