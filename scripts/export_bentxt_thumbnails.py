"""Export RGB thumbnails for samples recorded in a generation JSONL file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data_modules.ben_txt_datamodule import (
    BENImageReader,
    _sentinel2_rgb_tensor_to_pil,
)

RGB_BANDS = ["B04", "B03", "B02"]


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


def load_samples(path: Path) -> dict[str, dict]:
    samples = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            samples.setdefault(sample_id, row)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations",
        type=Path,
        required=True,
        help="Generation JSONL containing sample_id and patch_id fields.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lmdb-root",
        type=Path,
        default=None,
        help="Defaults to BIGEARTHNET_V2_LMDB_ROOT.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Defaults to BIGEARTHNET_TXT_PARQUET_PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lmdb_root = args.lmdb_root or Path(require_env("BIGEARTHNET_V2_LMDB_ROOT"))
    metadata_file = args.metadata_file or Path(
        require_env("BIGEARTHNET_TXT_PARQUET_PATH")
    )
    samples = load_samples(args.generations)
    if not samples:
        raise ValueError(f"No samples found in {args.generations}")

    reader = BENImageReader(lmdb_root, metadata_file, RGB_BANDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for sample_id, row in sorted(samples.items()):
        patch_id = str(row["patch_id"])
        image = _sentinel2_rgb_tensor_to_pil(reader[patch_id])
        filename = f"{sample_id}.png"
        image.save(args.output_dir / filename)
        manifest.append(
            {
                "sample_id": sample_id,
                "patch_id": patch_id,
                "filename": filename,
                "task_type": row.get("task_type"),
                "task_category": row.get("task_category"),
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} thumbnails to {args.output_dir}")


if __name__ == "__main__":
    main()
