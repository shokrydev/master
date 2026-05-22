#!/usr/bin/env python3
"""Download the local files needed for a BigEarthNet encoder.

The official reBEN model loader can read a local Hugging Face-style directory.
For the MobileViT encoder asset we only need the model config and weights:
`config.json` and `model.safetensors`.

Usage:
    python scripts/download_bigearthnet_encoder.py
    python scripts/download_bigearthnet_encoder.py --output-dir data/bigearthnet_encoders/mobilevit_s-all-v0.2.0
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_MODEL = "mobilevit_s-all-v0.2.0"
MODELS = {
    DEFAULT_MODEL: "BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.2.0",
}
REQUIRED_FILES = ("config.json", "model.safetensors")


def _default_output_dir(model_name: str) -> Path:
    return Path("data/bigearthnet_encoders") / model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BigEarthNet encoder files")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Known BigEarthNet encoder variant (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Override Hugging Face repo id. If set, --model is used only for the default output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to data/bigearthnet_encoders/<model>.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if required files already exist locally.",
    )
    args = parser.parse_args()

    repo_id = args.repo_id or MODELS[args.model]
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {repo_id}")
    print(f"Output directory: {output_dir}")

    for filename in REQUIRED_FILES:
        output_path = output_dir / filename
        if output_path.exists() and not args.force:
            print(f"Already exists: {output_path}")
            continue

        print(f"Downloading {filename}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=args.revision,
            local_dir=output_dir,
        )
        print(f"Saved to {downloaded}")

    print()
    print(f"export BIGEARTHNET_ENCODER_DIR={output_dir}")


if __name__ == "__main__":
    main()
