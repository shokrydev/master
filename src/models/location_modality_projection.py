# Learned projection from SatCLIP embeddings to VLM token space

import torch
import torch.nn as nn


class LocationModalityProjection(nn.Module):
    """Projects SatCLIP location embeddings into the VLM hidden space.

    Architecture: Linear → GELU → Linear, producing one or more "location tokens"
    that are inserted before the visual block in the decoder sequence.

    Args:
        satclip_dim: SatCLIP embedding dimension (256 for all variants)
        hidden_size: VLM hidden dimension (2048/2560/3584 for Qwen3-VL 2B/4B/8B)
        num_tokens: Number of location tokens to produce
    """

    def __init__(self, satclip_dim: int = 256, hidden_size: int = 2048, num_tokens: int = 1):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(satclip_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * num_tokens),
        )
        self.hidden_size = hidden_size

    def forward(self, loc_embed: torch.Tensor) -> torch.Tensor:
        """Project location embeddings to token space.

        Args:
            loc_embed: (B, satclip_dim) SatCLIP embeddings

        Returns:
            (B, num_tokens, hidden_size) location tokens
        """
        out = self.proj(loc_embed)  # (B, hidden_size * num_tokens)
        return out.view(-1, self.num_tokens, self.hidden_size)
