#!/usr/bin/env python3
"""Profile feasible BigEarthNet.txt microbatch sizes on a real GPU.

This script is a server-side profiling tool. It uses the same dataset,
collator, Qwen3-VL module, loc_embed path and non-RGB path as finetuning, but
runs only a few optimizer updates on selected worst-text samples.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401  # Must be imported before transformers for Unsloth optimizations
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

MODEL_NAMES = {
    "2B": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    "4B": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
    "8B": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile feasible BigEarthNet.txt microbatch sizes.",
    )
    parser.add_argument("--size", choices=MODEL_NAMES, default="8B")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Profile all core model sizes and location conditions in separate subprocesses.",
    )
    parser.add_argument(
        "--full-sizes",
        choices=MODEL_NAMES,
        nargs="+",
        default=list(MODEL_NAMES),
        help="Model sizes used by --full.",
    )
    parser.add_argument(
        "--full-conditions",
        choices=("no_loc", "loc_text", "loc_embed"),
        nargs="+",
        default=["no_loc", "loc_text", "loc_embed"],
        help="Location conditions used by --full.",
    )
    parser.add_argument(
        "--condition",
        choices=("no_loc", "loc_text", "loc_embed"),
        default="loc_embed",
        help="Location condition to profile. loc_embed is the heaviest core condition.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12, 14, 16],
        help="Per-device microbatch sizes to test.",
    )
    parser.add_argument(
        "--target-effective-batch",
        type=int,
        default=None,
        help=(
            "If set, run enough gradient-accumulation microsteps to approximate "
            "this effective batch for each microbatch size."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--selection",
        choices=("worst_text", "random"),
        default="worst_text",
        help="Use longest input+target rows or a random sample from the split.",
    )
    parser.add_argument("--candidate-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1, help="Measured optimizer updates.")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch_profiles"),
        help="Directory used by --full for per-scenario JSON outputs.",
    )
    parser.add_argument(
        "--continue-after-oom",
        action="store_true",
        help="Try later batch sizes after an OOM. By default profiling stops at first OOM.",
    )
    args = parser.parse_args()
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("--batch-sizes must contain positive integers.")
    if args.target_effective_batch is not None and args.target_effective_batch <= 0:
        parser.error("--target-effective-batch must be positive when set.")
    if args.steps <= 0:
        parser.error("--steps must be positive.")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative.")
    if args.candidate_count <= 0:
        parser.error("--candidate-count must be positive.")
    return args


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def candidate_indices(dataset: Any, args: argparse.Namespace) -> list[int]:
    frame = dataset.text_data
    if args.selection == "random":
        rng = random.Random(args.seed)
        indices = list(range(len(frame)))
        rng.shuffle(indices)
        return indices[: args.candidate_count]

    scores = (
        frame["input"].astype(str).str.len()
        + frame["output"].astype(str).str.len()
    )
    return scores.sort_values(ascending=False).head(args.candidate_count).index.tolist()


def accumulation_steps(microbatch_size: int, target_effective_batch: int | None) -> int:
    if target_effective_batch is None:
        return 1
    if microbatch_size >= target_effective_batch:
        return 1
    return (target_effective_batch + microbatch_size - 1) // microbatch_size


def default_json_output(args: argparse.Namespace) -> Path:
    return Path("outputs/batch_profiles") / f"profile_{args.size}_{args.condition}.json"


def make_batch(
    *,
    dataset: Any,
    collator: Any,
    indices: list[int],
    offset: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[int]]:
    selected = [
        indices[(offset + item_idx) % len(indices)]
        for item_idx in range(batch_size)
    ]
    samples = [dataset[index] for index in selected]
    batch = collator(samples)
    return move_to_device(batch, device), selected


def run_update(
    *,
    module: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    dataset: Any,
    collator: Any,
    indices: list[int],
    batch_size: int,
    accumulation: int,
    device: torch.device,
    offset: int,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    sequence_lengths: list[int] = []
    selected_indices: list[int] = []

    for microstep in range(accumulation):
        batch, selected = make_batch(
            dataset=dataset,
            collator=collator,
            indices=indices,
            offset=offset + microstep * batch_size,
            batch_size=batch_size,
            device=device,
        )
        selected_indices.extend(selected)
        sequence_lengths.append(int(batch["input_ids"].shape[1]))

        try:
            model_inputs, *_ = module._prepare_model_inputs(batch)
            outputs = module.model(**model_inputs)
            loss = outputs.loss / accumulation
            loss.backward()
        finally:
            module._reset_projected_token_state()
        losses.append(float(outputs.loss.detach().cpu()))

    optimizer.step()
    scheduler.step()

    return {
        "loss": sum(losses) / len(losses),
        "sequence_lengths": sequence_lengths,
        "selected_indices": selected_indices,
    }


def profile_batch_size(
    *,
    module: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    dataset: Any,
    collator: Any,
    indices: list[int],
    batch_size: int,
    accumulation: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for warmup_idx in range(args.warmup_steps):
        run_update(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset=dataset,
            collator=collator,
            indices=indices,
            batch_size=batch_size,
            accumulation=accumulation,
            device=device,
            offset=warmup_idx * batch_size * accumulation,
        )

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()

    sequence_lengths: list[int] = []
    losses: list[float] = []
    selected_indices: list[int] = []
    for step_idx in range(args.steps):
        result = run_update(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset=dataset,
            collator=collator,
            indices=indices,
            batch_size=batch_size,
            accumulation=accumulation,
            device=device,
            offset=(args.warmup_steps + step_idx) * batch_size * accumulation,
        )
        sequence_lengths.extend(result["sequence_lengths"])
        losses.append(result["loss"])
        selected_indices.extend(result["selected_indices"])

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    samples = args.steps * batch_size * accumulation
    padded_tokens = batch_size * sum(sequence_lengths)

    return {
        "batch_size": batch_size,
        "accumulation_steps": accumulation,
        "effective_batch": batch_size * accumulation,
        "steps": args.steps,
        "samples": samples,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_optimizer_step": elapsed_seconds / args.steps,
        "samples_per_second": samples / elapsed_seconds,
        "padded_tokens_per_second": padded_tokens / elapsed_seconds,
        "loss": sum(losses) / len(losses),
        "sequence_length_max": max(sequence_lengths),
        "sequence_length_min": min(sequence_lengths),
        "sequence_lengths": sequence_lengths,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "selected_indices": selected_indices[: min(len(selected_indices), 32)],
    }


def main_single(args: argparse.Namespace) -> dict[str, Any]:
    load_env(ENV_PATH)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for batch-size profiling.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    from src.data_modules.ben_txt_datamodule import BENTxTDataset
    from src.data_modules.geo_aware_collator import (
        DEFAULT_LOCATION_EMBED_MARKER,
        DEFAULT_LOCATION_TEXT_TEMPLATE,
    )
    from src.lightning_modules.qwen3_vl_module import Qwen3VLModule

    metadata_file = require_env("BIGEARTHNET_TXT_PARQUET_PATH")
    lmdb_file = require_env("BIGEARTHNET_V2_LMDB_ROOT")
    encoder_dir = require_env("BIGEARTHNET_ENCODER_DIR")
    satclip_checkpoint = (
        require_env("SATCLIP_CHECKPOINT_PATH")
        if args.condition == "loc_embed"
        else None
    )

    dataset = BENTxTDataset(
        lmdb_file=lmdb_file,
        metadata_file=metadata_file,
        bands="S1S2-10m20m",
        rgb_render_mode="copernicus",
        rgb_quantile=0.90,
        splits=[args.split],
    )
    indices = candidate_indices(dataset, args)
    if not indices:
        raise SystemExit(f"No samples found for split={args.split!r}.")

    profile_optimizer_steps = max(1, (args.warmup_steps + args.steps) * len(args.batch_sizes))

    module = Qwen3VLModule(
        model_name_or_path=MODEL_NAMES[args.size],
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=profile_optimizer_steps,
        max_new_tokens=0,
        num_validation_generation_batches=0,
        loc_mode=args.condition,
        location_text_template=(
            DEFAULT_LOCATION_TEXT_TEMPLATE if args.condition == "loc_text" else None
        ),
        location_embed_marker=(
            DEFAULT_LOCATION_EMBED_MARKER if args.condition == "loc_embed" else None
        ),
        non_rgb_conditioning="enabled",
        non_rgb_encoder_dir=encoder_dir,
        non_rgb_feature_mode="spatial_4x4",
        non_rgb_spatial_pool_size=4,
        num_non_rgb_tokens=16,
        non_rgb_projection_lr_multiplier=5.0,
        satclip_checkpoint=satclip_checkpoint,
        satclip_dim=256,
        num_location_tokens=8 if args.condition == "loc_embed" else 1,
        location_projection_lr_multiplier=5.0 if args.condition == "loc_embed" else 1.0,
    )

    module.setup("fit")
    module.to(device)
    module.train()
    collator = module._collator
    if collator is None:
        raise SystemExit("Qwen3VLModule did not initialize its collator.")

    optimizer_config = module.configure_optimizers()
    optimizer = optimizer_config["optimizer"]
    scheduler = optimizer_config["lr_scheduler"]["scheduler"]

    results: list[dict[str, Any]] = []
    print(
        "Profiling "
        f"size={args.size} condition={args.condition} split={args.split} "
        f"selection={args.selection}"
    )
    print(f"Candidate rows: {len(indices)}")

    for batch_size in args.batch_sizes:
        accumulation = accumulation_steps(batch_size, args.target_effective_batch)
        print(
            f"\nTesting batch_size={batch_size}, accumulation_steps={accumulation}, "
            f"effective_batch={batch_size * accumulation}"
        )
        try:
            result = profile_batch_size(
                module=module,
                optimizer=optimizer,
                scheduler=scheduler,
                dataset=dataset,
                collator=collator,
                indices=indices,
                batch_size=batch_size,
                accumulation=accumulation,
                args=args,
                device=device,
            )
            result["status"] = "ok"
            results.append(result)
            print(
                "OK "
                f"peak_allocated={result['peak_allocated_gb']:.2f}GB "
                f"peak_reserved={result['peak_reserved_gb']:.2f}GB "
                f"sec/step={result['seconds_per_optimizer_step']:.2f} "
                f"samples/sec={result['samples_per_second']:.2f} "
                f"max_seq={result['sequence_length_max']}"
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            result = {
                "batch_size": batch_size,
                "accumulation_steps": accumulation,
                "effective_batch": batch_size * accumulation,
                "status": "oom",
                "error": str(error).splitlines()[0],
            }
            results.append(result)
            print(f"OOM batch_size={batch_size}: {result['error']}")
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if not args.continue_after_oom:
                break

    summary = {
        "size": args.size,
        "model_name": MODEL_NAMES[args.size],
        "condition": args.condition,
        "split": args.split,
        "selection": args.selection,
        "target_effective_batch": args.target_effective_batch,
        "results": results,
    }

    json_output = args.json_output or default_json_output(args)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {json_output}")

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    return summary


def main_full(args: argparse.Namespace) -> None:
    load_env(ENV_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    for size in args.full_sizes:
        for condition in args.full_conditions:
            output_path = args.output_dir / f"profile_{size}_{condition}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--size",
                size,
                "--condition",
                condition,
                "--batch-sizes",
                *[str(batch_size) for batch_size in args.batch_sizes],
                "--split",
                args.split,
                "--selection",
                args.selection,
                "--candidate-count",
                str(args.candidate_count),
                "--seed",
                str(args.seed),
                "--steps",
                str(args.steps),
                "--warmup-steps",
                str(args.warmup_steps),
                "--max-seq-length",
                str(args.max_seq_length),
                "--lora-r",
                str(args.lora_r),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--json-output",
                str(output_path),
            ]
            if args.target_effective_batch is not None:
                command.extend(
                    ["--target-effective-batch", str(args.target_effective_batch)]
                )
            if args.continue_after_oom:
                command.append("--continue-after-oom")
            commands.append(command)

    summary: list[dict[str, Any]] = []
    for command in commands:
        print("\nRunning:")
        print(" ".join(shlex.quote(part) for part in command))
        started_at = time.perf_counter()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        elapsed_seconds = time.perf_counter() - started_at
        scenario = {
            "command": command,
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed_seconds,
        }
        summary.append(scenario)
        if completed.returncode != 0:
            summary_path = args.output_dir / "profile_full_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            raise SystemExit(
                f"Profiling command failed with return code {completed.returncode}. "
                f"Partial summary written to {summary_path}."
            )

    summary_path = args.output_dir / "profile_full_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull profiling summary written to {summary_path}")


def main() -> None:
    args = parse_args()
    if args.full:
        main_full(args)
    else:
        main_single(args)


if __name__ == "__main__":
    main()
