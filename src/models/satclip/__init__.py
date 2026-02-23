# Vendored SatCLIP location encoder from github.com/microsoft/satclip (MIT License)
# Only the location encoder branch is included (no vision encoder).
#
# Usage:
#   from src.models.satclip import get_satclip
#   loc_encoder = get_satclip("path/to/satclip.ckpt", device="cuda")
#   embeddings = loc_encoder(coords)  # (B, 2) float64 [lon, lat] → (B, embed_dim)

import torch

from .location_encoder import (
    LocationEncoder,
    get_neural_network,
    get_positional_encoding,
)


def get_satclip(ckpt_path: str, device="cpu") -> LocationEncoder:
    """Load a pretrained SatCLIP location encoder from checkpoint.

    Args:
        ckpt_path: Path to SatCLIP checkpoint (.ckpt file)
        device: Device to load the model on

    Returns:
        LocationEncoder in eval mode, float64. Input: (B, 2) [lon, lat] degrees.
        Output: (B, embed_dim) embeddings.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt["hyper_parameters"]

    posenc = get_positional_encoding(
        hp["le_type"],
        hp["legendre_polys"],
        hp["harmonics_calculation"],
        hp.get("min_radius", 1),
        hp.get("max_radius", 360),
        hp.get("frequency_num", 16),
    )

    nnet = get_neural_network(
        hp["pe_type"],
        posenc.embedding_dim,
        hp["embed_dim"],
        hp["capacity"],
        hp["num_hidden_layers"],
    )

    # Extract only the nnet state dict from the full checkpoint
    state_dict = ckpt["state_dict"]
    state_dict = {
        k[k.index("nnet"):]: state_dict[k]
        for k in state_dict.keys()
        if "nnet" in k
    }

    loc_encoder = LocationEncoder(posenc, nnet).double()
    loc_encoder.load_state_dict(state_dict)
    loc_encoder.eval()

    return loc_encoder


__all__ = ["get_satclip", "LocationEncoder"]
