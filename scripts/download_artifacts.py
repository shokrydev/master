#!/usr/bin/env python3
"""Download model artifacts needed for BigEarthNet.txt finetuning.

The script reads machine-local paths from the repo-root `.env`. It downloads
only model artifacts; BigEarthNet.txt dataset preparation is handled separately
by the dataset's Hugging Face instructions.

Default behavior downloads the first smoke-run artifacts:
    - Qwen3-VL 2B 4-bit
    - SatCLIP ViT16-L10 checkpoint
    - BigEarthNet MobileViT S1/S2 encoder

Use `--all` to prefetch all Qwen sizes.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

QWEN_REPOS = {
    "2B": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    "4B": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
    "8B": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
}

SATCLIP_REPO = "microsoft/SatCLIP-ViT16-L10"
SATCLIP_FILENAME = "satclip-vit16-l10.ckpt"

BIGEARTHNET_ENCODER_REPO = "BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.2.0"
BIGEARTHNET_ENCODER_FILES = ("config.json", "model.safetensors")


def load_env(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(
            "Missing .env. Copy .env.example to .env and fill in the server-local paths."
        )

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise SystemExit(f"Invalid .env line: {raw_line}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit(f"Invalid .env line: {raw_line}")
        parsed_value = shlex.split(value, comments=False, posix=True)
        raw_value = parsed_value[0] if parsed_value else ""
        os.environ[key] = os.path.expanduser(os.path.expandvars(raw_value))


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required env var in .env: {name}")
    return value


def download_qwen(sizes: list[str], dry_run: bool) -> None:
    hf_home = require_env("HF_HOME")
    print(f"Hugging Face cache: {hf_home}")
    if dry_run:
        for size in sizes:
            print(f"[dry-run] Would download Qwen {size}: {QWEN_REPOS[size]}")
        return

    from huggingface_hub import snapshot_download

    for size in sizes:
        repo_id = QWEN_REPOS[size]
        print(f"Downloading Qwen {size}: {repo_id}")
        local_path = snapshot_download(repo_id=repo_id)
        print(f"Cached at: {local_path}")


def download_satclip(dry_run: bool) -> None:
    checkpoint_path = Path(require_env("SATCLIP_CHECKPOINT_PATH"))
    if checkpoint_path.name != SATCLIP_FILENAME:
        raise SystemExit(
            "SATCLIP_CHECKPOINT_PATH must end with "
            f"{SATCLIP_FILENAME!r}; got {checkpoint_path}"
        )

    print(f"SatCLIP checkpoint: {checkpoint_path}")
    if checkpoint_path.is_file() and not dry_run:
        print("Already exists.")
        return
    if dry_run:
        print(f"[dry-run] Would download {SATCLIP_REPO}/{SATCLIP_FILENAME}")
        return

    from huggingface_hub import hf_hub_download

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=SATCLIP_REPO,
        filename=SATCLIP_FILENAME,
        local_dir=checkpoint_path.parent,
    )
    print(f"Saved to: {downloaded}")


def download_bigearthnet_encoder(dry_run: bool) -> None:
    output_dir = Path(require_env("BIGEARTHNET_ENCODER_DIR"))
    print(f"BigEarthNet encoder directory: {output_dir}")

    if dry_run:
        for filename in BIGEARTHNET_ENCODER_FILES:
            print(f"[dry-run] Would download {BIGEARTHNET_ENCODER_REPO}/{filename}")
        return

    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in BIGEARTHNET_ENCODER_FILES:
        output_path = output_dir / filename
        if output_path.is_file():
            print(f"Already exists: {output_path}")
            continue
        print(f"Downloading BigEarthNet encoder file: {filename}")
        downloaded = hf_hub_download(
            repo_id=BIGEARTHNET_ENCODER_REPO,
            filename=filename,
            local_dir=output_dir,
        )
        print(f"Saved to: {downloaded}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all Qwen sizes plus SatCLIP and BigEarthNet encoder.",
    )
    parser.add_argument(
        "--qwen",
        nargs="*",
        choices=sorted(QWEN_REPOS),
        metavar="SIZE",
        help="Download selected Qwen sizes. Use --qwen without sizes for all sizes.",
    )
    parser.add_argument(
        "--satclip",
        action="store_true",
        help="Download the SatCLIP ViT16-L10 checkpoint.",
    )
    parser.add_argument(
        "--bigearthnet-encoder",
        action="store_true",
        help="Download the BigEarthNet MobileViT encoder config and weights.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without contacting Hugging Face.",
    )
    return parser.parse_args()


def selected_artifacts(args: argparse.Namespace) -> tuple[list[str], bool, bool]:
    if args.all:
        return sorted(QWEN_REPOS), True, True

    has_explicit_selection = (
        args.qwen is not None or args.satclip or args.bigearthnet_encoder
    )
    if not has_explicit_selection:
        return ["2B"], True, True

    qwen_sizes: list[str] = []
    if args.qwen is not None:
        qwen_sizes = sorted(QWEN_REPOS) if not args.qwen else args.qwen
    return qwen_sizes, args.satclip, args.bigearthnet_encoder


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)

    qwen_sizes, include_satclip, include_bigearthnet_encoder = selected_artifacts(args)

    if qwen_sizes:
        download_qwen(qwen_sizes, args.dry_run)
    if include_satclip:
        download_satclip(args.dry_run)
    if include_bigearthnet_encoder:
        download_bigearthnet_encoder(args.dry_run)

    print("Artifact download step complete.")


if __name__ == "__main__":
    main()
