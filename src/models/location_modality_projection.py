"""Learned projection from SatCLIP embeddings to VLM token space."""

import torch
import torch.nn as nn


class LocationModalityProjection(nn.Module):
    """Projects SatCLIP location embeddings into the VLM hidden space.

    The default ``mlp`` architecture preserves the original
    Linear-GELU-Linear implementation. The ``linear`` architecture is a
    compact capacity ablation with one direct map to all output tokens.

    Args:
        satclip_dim: SatCLIP embedding dimension (256 for all variants)
        hidden_size: VLM hidden dimension (2048/2560/3584 for Qwen3-VL 2B/4B/8B)
        num_tokens: Number of location tokens to produce
        architecture: ``mlp`` (original) or ``linear`` (compact ablation)
    """

    def __init__(
        self,
        satclip_dim: int = 256,
        hidden_size: int = 2048,
        num_tokens: int = 1,
        *,
        architecture: str = "mlp",
    ) -> None:
        super().__init__()
        if satclip_dim <= 0:
            raise ValueError("satclip_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if architecture not in {"mlp", "linear"}:
            raise ValueError("architecture must be 'mlp' or 'linear'")

        self.satclip_dim = int(satclip_dim)
        self.hidden_size = int(hidden_size)
        self.num_tokens = int(num_tokens)
        self.architecture = architecture
        output_dim = self.hidden_size * self.num_tokens
        if self.architecture == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(self.satclip_dim, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, output_dim),
            )
        else:
            self.proj = nn.Linear(self.satclip_dim, output_dim, bias=True)

    def forward(self, loc_embed: torch.Tensor) -> torch.Tensor:
        """Project location embeddings to token space.

        Args:
            loc_embed: (B, satclip_dim) SatCLIP embeddings

        Returns:
            (B, num_tokens, hidden_size) location tokens
        """
        out = self.proj(loc_embed)  # (B, hidden_size * num_tokens)
        return out.view(-1, self.num_tokens, self.hidden_size)

    def manifest(self) -> dict[str, object]:
        """Return the projection architecture needed for strict reload."""
        return {
            "version": 1,
            "architecture": self.architecture,
            "satclip_dim": self.satclip_dim,
            "hidden_size": self.hidden_size,
            "num_tokens": self.num_tokens,
            "bias": True,
        }
