#!/usr/bin/env python3
"""Check local prerequisites for BigEarthNet.txt server runs."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
LIGHTNING_CONFIG_KEYS = {
    "seed_everything",
    "trainer",
    "model",
    "data",
    "paths",
    "optimizer",
    "lr_scheduler",
    "ckpt_path",
    "weights_only",
}

REQUIRED_ENV_VARS = (
    "BIGEARTHNET_V2_LMDB_ROOT",
    "BIGEARTHNET_TXT_PARQUET_PATH",
    "BIGEARTHNET_ENCODER_DIR",
    "SATCLIP_CHECKPOINT_PATH",
    "HF_HOME",
    "FINETUNING_OUTPUT_ROOT",
    "SLURM_DEFAULT_PARTITION",
)
BENTXT_REQUIRED_COLUMNS = {
    "ID",
    "s1_name",
    "patch_id",
    "input",
    "output",
    "type",
    "category",
    "split",
    "latitude",
    "longitude",
    "country",
    "season",
    "climate_zone",
}
LOCATION_REDACTED_CAPTION_COLUMNS = {"patch_id", "refined_caption"}


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
        if not key:
            raise SystemExit(f"Invalid .env line: {raw_line}")
        parsed_value = shlex.split(value.strip(), comments=False, posix=True)
        raw_value = parsed_value[0] if parsed_value else ""
        os.environ[key] = os.path.expanduser(os.path.expandvars(raw_value))


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


def require_parquet_columns(path: Path, required_columns: set[str], label: str) -> None:
    columns = set(pq.ParquetFile(path).schema_arrow.names)
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        raise SystemExit(
            f"{label} is missing required parquet columns: {missing_columns}"
        )
    print(f"OK {label} columns: {path}")


def load_lightning_config(config_file: Path) -> object:
    config = OmegaConf.load(config_file)
    unknown_keys = set(config.keys()) - LIGHTNING_CONFIG_KEYS
    if unknown_keys:
        relative_path = config_file.relative_to(REPO_ROOT)
        raise SystemExit(
            f"{relative_path} contains unsupported top-level keys: "
            f"{', '.join(sorted(unknown_keys))}"
        )
    unresolved = OmegaConf.to_container(config, resolve=False)
    if "paths" in unresolved and not contains_paths_reference(
        {key: value for key, value in unresolved.items() if key != "paths"}
    ):
        relative_path = config_file.relative_to(REPO_ROOT)
        raise SystemExit(f"{relative_path} defines paths but does not use them locally.")
    OmegaConf.to_container(config, resolve=True)
    return config


def contains_paths_reference(value: object) -> bool:
    if isinstance(value, str):
        return "${paths." in value
    if isinstance(value, dict):
        return any(contains_paths_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_paths_reference(item) for item in value)
    return False


def check_env_paths() -> None:
    for name in REQUIRED_ENV_VARS:
        require_env(name)

    require_path(Path(require_env("BIGEARTHNET_V2_LMDB_ROOT")), "BigEarthNet-v2 LMDB environment", directory=True)
    bentxt_path = Path(require_env("BIGEARTHNET_TXT_PARQUET_PATH"))
    require_path(bentxt_path, "BigEarthNet.txt parquet", directory=False)
    require_parquet_columns(bentxt_path, BENTXT_REQUIRED_COLUMNS, "BigEarthNet.txt parquet")

    location_redacted_path = os.environ.get("BENTXT_LOCATION_REDACTED_CAPTION_FILE")
    if location_redacted_path:
        caption_path = Path(location_redacted_path)
        require_path(caption_path, "location-redacted caption parquet", directory=False)
        require_parquet_columns(
            caption_path,
            LOCATION_REDACTED_CAPTION_COLUMNS,
            "location-redacted caption parquet",
        )

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

    print(f"OK Slurm default partition: {require_env('SLURM_DEFAULT_PARTITION')}")


def check_config_composition() -> None:
    for config_file in sorted((REPO_ROOT / "configs").glob("**/*.yaml")):
        load_lightning_config(config_file)
    print("OK config files contain only LightningCLI-supported top-level keys.")

    config_files = [
        "configs/finetuning/bigearthnet_txt_shared.yaml",
        "configs/finetuning/loc_embed.yaml",
        "configs/finetuning/bigearthnet_txt_smoke.yaml",
    ]
    configs = []
    for config_file in config_files:
        configs.append(load_lightning_config(REPO_ROOT / config_file))

    config = OmegaConf.merge(*configs)
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
    if data_args.get("use_location_redacted_captions") is not False:
        raise SystemExit(
            "Smoke config should use original BigEarthNet.txt caption targets by default."
        )
    if data_args.get("location_redacted_caption_file") is not None:
        raise SystemExit(
            "Smoke config should not set data.init_args.location_redacted_caption_file."
        )

    print("OK BigEarthNet loc_embed smoke config composes and resolves.")


def main() -> None:
    load_env(ENV_PATH)
    check_env_paths()
    check_config_composition()
    print("Server setup check passed.")


if __name__ == "__main__":
    main()
