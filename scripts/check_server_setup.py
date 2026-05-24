#!/usr/bin/env python3
"""Check local prerequisites for BigEarthNet.txt server runs."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

REQUIRED_ENV_VARS = (
    "BIGEARTHNET_V2_LMDB_ROOT",
    "BIGEARTHNET_TXT_PARQUET_PATH",
    "BIGEARTHNET_ENCODER_DIR",
    "SATCLIP_CHECKPOINT_PATH",
    "HF_HOME",
    "FINETUNING_OUTPUT_ROOT",
)


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
        parsed_value = shlex.split(value.strip(), comments=False, posix=True)
        os.environ[key.strip()] = parsed_value[0] if parsed_value else ""


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required env var in .env: {name}")
    return value


def require_path(path: Path, label: str, *, directory: bool) -> None:
    if directory:
        if not path.is_dir():
            raise SystemExit(f"{label} is not a directory: {path}")
    elif not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    print(f"OK {label}: {path}")


def check_env_paths() -> None:
    for name in REQUIRED_ENV_VARS:
        require_env(name)

    require_path(Path(require_env("BIGEARTHNET_V2_LMDB_ROOT")), "BigEarthNet-v2 LMDB root", directory=True)
    require_path(Path(require_env("BIGEARTHNET_TXT_PARQUET_PATH")), "BigEarthNet.txt parquet", directory=False)

    encoder_dir = Path(require_env("BIGEARTHNET_ENCODER_DIR"))
    require_path(encoder_dir, "BigEarthNet encoder directory", directory=True)
    require_path(encoder_dir / "config.json", "BigEarthNet encoder config", directory=False)
    require_path(encoder_dir / "model.safetensors", "BigEarthNet encoder weights", directory=False)

    require_path(Path(require_env("SATCLIP_CHECKPOINT_PATH")), "SatCLIP checkpoint", directory=False)

    hf_home = Path(require_env("HF_HOME"))
    hf_home.mkdir(parents=True, exist_ok=True)
    require_path(hf_home, "Hugging Face cache", directory=True)

    output_root = Path(require_env("FINETUNING_OUTPUT_ROOT"))
    output_root.mkdir(parents=True, exist_ok=True)
    require_path(output_root, "finetuning output root", directory=True)


def check_config_composition() -> None:
    config_files = [
        "configs/finetuning/bigearthnet_txt_shared.yaml",
        "configs/finetuning/loc_embed.yaml",
        "configs/finetuning/bigearthnet_txt_smoke.yaml",
    ]
    config = OmegaConf.merge(
        *[OmegaConf.load(REPO_ROOT / config_file) for config_file in config_files]
    )
    resolved = OmegaConf.to_container(config, resolve=True)

    model_args = resolved["model"]["init_args"]
    data_args = resolved["data"]["init_args"]

    expected = {
        "loc_mode": "loc_embed",
        "non_rgb_conditioning": "enabled",
        "non_rgb_feature_mode": "spatial_4x4",
    }
    for key, value in expected.items():
        if model_args.get(key) != value:
            raise SystemExit(f"Unexpected model.init_args.{key}: {model_args.get(key)!r}")

    if data_args.get("bands") != "S1S2-10m20m":
        raise SystemExit(f"Unexpected data.init_args.bands: {data_args.get('bands')!r}")

    print("OK BigEarthNet loc_embed smoke config composes and resolves.")


def main() -> None:
    load_env(ENV_PATH)
    check_env_paths()
    check_config_composition()
    print("Server setup check passed.")


if __name__ == "__main__":
    main()
