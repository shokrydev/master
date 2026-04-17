#!/usr/bin/env python3
"""Download pretrained SatCLIP checkpoint from HuggingFace Hub.

Usage:
    python scripts/download_satclip.py
    python scripts/download_satclip.py --model SatCLIP-ViT16-L10 --output_dir data/satclip
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


MODELS = {
    "SatCLIP-ViT16-L40": ("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
    "SatCLIP-ViT16-L10": ("microsoft/SatCLIP-ViT16-L10", "satclip-vit16-l10.ckpt"),
    "SatCLIP-ResNet50-L40": ("microsoft/SatCLIP-ResNet50-L40", "satclip-resnet50-l40.ckpt"),
    "SatCLIP-ResNet50-L10": ("microsoft/SatCLIP-ResNet50-L10", "satclip-resnet50-l10.ckpt"),
    "SatCLIP-ResNet18-L40": ("microsoft/SatCLIP-ResNet18-L40", "satclip-resnet18-l40.ckpt"),
    "SatCLIP-ResNet18-L10": ("microsoft/SatCLIP-ResNet18-L10", "satclip-resnet18-l10.ckpt"),
}


def main():
    parser = argparse.ArgumentParser(description="Download SatCLIP checkpoint")
    parser.add_argument("--model", type=str, default="SatCLIP-ViT16-L10",
                        choices=list(MODELS.keys()),
                        help="SatCLIP model variant (default: SatCLIP-ViT16-L10)")
    parser.add_argument("--output_dir", type=str, default="data/satclip",
                        help="Output directory (default: data/satclip)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id, filename = MODELS[args.model]
    output_path = output_dir / filename

    if output_path.exists():
        print(f"Checkpoint already exists at {output_path}")
        return

    print(f"Downloading {args.model} from {repo_id}...")
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=output_dir)
    print(f"Saved to {downloaded}")


if __name__ == "__main__":
    main()
