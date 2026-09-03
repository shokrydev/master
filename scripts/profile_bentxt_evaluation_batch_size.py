#!/usr/bin/env python3
"""Profile safe and efficient BigEarthNet.txt generation batch sizes.

For each production generation bucket, the capacity phase replicates its
longest tokenized bench instruction under the heaviest core condition
(`loc_embed`) at the selected model size and forces every sequence to its
configured token cap. The throughput phase measures natural-EOS generation on
an evenly spaced subset of the same bucket.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shlex
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import unsloth  # noqa: F401  # Import before transformers/Unsloth model modules.
from evaluation_batch_profile_logic import (
    evenly_spaced_indices,
    recommend_throughput_batch,
    recommend_worker_count,
    refinement_batch,
    safe_capacity_batches,
)
from torch.utils.data import DataLoader, Subset
from unsloth import FastVisionModel

from src.bentxt_generation import (
    DEFAULT_MAX_NEW_TOKENS_BY_BUCKET,
    GENERATION_BUCKETS,
    bucket_indices,
)
from src.bentxt_grounding import format_grounding_prompt, format_grounding_target

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
MODEL_NAMES = {
    "2B": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    "4B": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
    "8B": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
}
GIB = 1024**3


def load_env(path: Path) -> None:
    if not path.is_file():
        raise SystemExit("Missing .env. Copy .env.example to .env and fill in server-local paths.")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise SystemExit(f"Invalid .env line: {raw_line}")
        key, value = line.split("=", 1)
        parsed = shlex.split(value.strip(), comments=False, posix=True)
        raw_value = parsed[0] if parsed else ""
        os.environ[key.strip()] = os.path.expanduser(os.path.expandvars(raw_value))


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required env var in .env: {name}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile task-aware capacity and evaluation throughput."
    )
    parser.add_argument("--size", choices=tuple(MODEL_NAMES), default="2B")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 384, 512],
    )
    parser.add_argument("--throughput-samples", type=int, default=1024)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument(
        "--capacity-resolution",
        type=int,
        default=32,
        help="Narrow the safe/unsafe batch boundary to this many samples.",
    )
    parser.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=0.90,
        help="Maximum allowed peak reserved VRAM fraction for a recommendation.",
    )
    parser.add_argument(
        "--throughput-near-best-fraction",
        type=float,
        default=0.98,
        help="Choose the smallest safe batch reaching this fraction of peak throughput.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("--batch-sizes must contain positive integers")
    if args.batch_sizes != sorted(set(args.batch_sizes)):
        parser.error("--batch-sizes must be unique and strictly increasing")
    if args.throughput_samples <= 0:
        parser.error("--throughput-samples must be positive")
    if any(worker_count < 0 for worker_count in args.worker_counts):
        parser.error("--worker-counts must contain non-negative integers")
    if args.worker_counts != sorted(set(args.worker_counts)):
        parser.error("--worker-counts must be unique and strictly increasing")
    if args.capacity_resolution <= 0:
        parser.error("--capacity-resolution must be positive")
    if not 0.0 < args.memory_safety_fraction < 1.0:
        parser.error("--memory-safety-fraction must be between 0 and 1")
    if not 0.0 < args.throughput_near_best_fraction <= 1.0:
        parser.error("--throughput-near-best-fraction must be in (0, 1]")
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


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def longest_instruction_index(
    dataset: Any,
    tokenizer: Any,
    indices: Sequence[int],
) -> tuple[int, int]:
    """Find a bucket's longest model-facing instruction without loading imagery."""
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    best_index = -1
    best_length = -1
    for index in indices:
        row = dataset.text_data.iloc[index]
        prompt = format_grounding_prompt(
            str(row["input"]),
            grounding_format="qwen3_json",
            ref_token=("<|object_ref_start|>", "<|object_ref_end|>"),
            point_token=("", ""),
        )
        prompt = f"{prompt}\nScene coordinates:"
        token_count = len(raw_tokenizer.encode(prompt, add_special_tokens=False))
        if token_count > best_length:
            best_index = index
            best_length = token_count
    if best_index < 0:
        raise RuntimeError("The generation bucket is empty")
    return best_index, best_length


def longest_target(
    dataset: Any,
    tokenizer: Any,
    indices: Sequence[int],
) -> tuple[int, int]:
    """Return the row and token length of a bucket's longest model target."""
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    best_index = -1
    best_length = -1
    for index in indices:
        row = dataset.text_data.iloc[index]
        target = format_grounding_target(
            str(row["output"]),
            task_type=str(row["type"]),
            grounding_format="qwen3_json",
        )
        token_count = len(raw_tokenizer.encode(target, add_special_tokens=False))
        if token_count > best_length:
            best_index = index
            best_length = token_count
    if best_index < 0:
        raise RuntimeError("The generation bucket is empty")
    return best_index, best_length


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def generate_collated_batch(
    *,
    module: Any,
    batch: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
    force_full_length: bool,
    clear_before: bool = True,
) -> dict[str, Any]:
    if clear_before:
        clear_cuda(device)
        torch.cuda.reset_peak_memory_stats(device)
    generated_ids: torch.Tensor | None = None
    try:
        batch = move_to_device(batch, device)
        model_inputs, *_ = module._prepare_model_inputs(batch)
        batch_size = int(model_inputs["input_ids"].shape[0])
        input_length = int(model_inputs["input_ids"].shape[1])
        generation_args: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        if force_full_length:
            generation_args["min_new_tokens"] = max_new_tokens
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = module.model.generate(**model_inputs, **generation_args)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        generated_tokens = int(generated_ids.shape[1] - input_length)
        return {
            "status": "ok",
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "samples_per_second": batch_size / elapsed,
            "input_sequence_length": input_length,
            "generated_sequence_length_max": generated_tokens,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / GIB,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / GIB,
        }
    finally:
        module._reset_decoder_conditioning_state()
        del generated_ids
        del batch


def generate_sample_batch(
    *,
    module: Any,
    collator: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    force_full_length: bool,
) -> dict[str, Any]:
    return generate_collated_batch(
        module=module,
        batch=collator(samples),
        device=device,
        max_new_tokens=max_new_tokens,
        force_full_length=force_full_length,
    )


def generate_production_batch(
    *,
    module: Any,
    batch: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Measure the production generation and decoding path for one batch."""
    try:
        batch = move_to_device(batch, device)
        model_inputs, *_ = module._prepare_model_inputs(batch)
        batch_size = int(model_inputs["input_ids"].shape[0])
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        module._generate_for_batch(
            model_inputs,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        return {
            "status": "ok",
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "samples_per_second": batch_size / elapsed,
        }
    finally:
        module._reset_decoder_conditioning_state()
        del batch


def capacity_profile(
    *,
    module: Any,
    collator: Any,
    worst_sample: dict[str, Any],
    batch_sizes: Sequence[int],
    device: torch.device,
    max_new_tokens: int,
    total_memory_gb: float,
    safety_fraction: float,
    resolution: int,
    bucket: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    by_batch: dict[int, dict[str, Any]] = {}
    memory_limit_gb = total_memory_gb * safety_fraction

    def run_candidate(batch_size: int) -> dict[str, Any]:
        if batch_size in by_batch:
            return by_batch[batch_size]
        print(
            f"\nCapacity stress: bucket={bucket}, batch={batch_size}, "
            f"forced_new_tokens={max_new_tokens}"
        )
        samples = [copy.deepcopy(worst_sample) for _ in range(batch_size)]
        try:
            result = generate_sample_batch(
                module=module,
                collator=collator,
                samples=samples,
                device=device,
                max_new_tokens=max_new_tokens,
                force_full_length=True,
            )
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            result = {
                "status": "oom",
                "batch_size": batch_size,
                "error": str(error).splitlines()[0],
            }
            print(f"OOM batch={batch_size}: {result['error']}")
            samples = []
            clear_cuda(device)
            results.append(result)
            by_batch[batch_size] = result
            return result
        result["within_memory_safety_margin"] = float(result["peak_reserved_gb"]) <= memory_limit_gb
        results.append(result)
        by_batch[batch_size] = result
        print(
            f"OK batch={batch_size} peak_reserved={result['peak_reserved_gb']:.2f}GB "
            f"peak_allocated={result['peak_allocated_gb']:.2f}GB "
            f"samples/s={result['samples_per_second']:.3f} "
            f"within_{safety_fraction:.0%}_margin="
            f"{result['within_memory_safety_margin']}"
        )
        samples = []
        clear_cuda(device)
        return result

    lower_safe = 0
    upper_unsafe: int | None = None
    for batch_size in batch_sizes:
        result = run_candidate(batch_size)
        is_safe = result.get("status") == "ok" and bool(result.get("within_memory_safety_margin"))
        if is_safe:
            lower_safe = batch_size
            continue
        upper_unsafe = batch_size
        break

    while upper_unsafe is not None:
        candidate = refinement_batch(lower_safe, upper_unsafe, resolution)
        if candidate is None:
            break
        result = run_candidate(candidate)
        is_safe = result.get("status") == "ok" and bool(result.get("within_memory_safety_margin"))
        if is_safe:
            lower_safe = candidate
        else:
            upper_unsafe = candidate

    return results


def bucket_throughput_profile(
    *,
    module: Any,
    collator: Any,
    dataset: Any,
    indices: Sequence[int],
    batch_sizes: Sequence[int],
    device: torch.device,
    max_new_tokens: int,
    num_workers: int,
    bucket: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        usable_count = len(indices)
        if usable_count == 0:
            continue
        print(f"\nBucket throughput: bucket={bucket}, batch={batch_size}, samples={usable_count}")
        loader = DataLoader(
            Subset(dataset, list(indices)),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collator,
        )
        clear_cuda(device)
        torch.cuda.reset_peak_memory_stats(device)
        wall_started = time.perf_counter()
        generation_seconds = 0.0
        batch_count = 0
        try:
            for batch in loader:
                batch_result = generate_production_batch(
                    module=module,
                    batch=batch,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )
                generation_seconds += float(batch_result["elapsed_seconds"])
                batch_count += 1
            torch.cuda.synchronize(device)
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            result = {
                "status": "oom",
                "batch_size": batch_size,
                "num_workers": num_workers,
                "error": str(error).splitlines()[0],
            }
            results.append(result)
            loader = None
            clear_cuda(device)
            break
        wall_elapsed = time.perf_counter() - wall_started
        result = {
            "status": "ok",
            "batch_size": batch_size,
            "samples": usable_count,
            "batches": batch_count,
            "num_workers": num_workers,
            "generation_seconds": generation_seconds,
            "wall_seconds": wall_elapsed,
            "generation_samples_per_second": usable_count / generation_seconds,
            "samples_per_second": usable_count / wall_elapsed,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / GIB,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / GIB,
        }
        results.append(result)
        print(
            f"OK batch={batch_size} generation={generation_seconds:.2f}s "
            "generation_samples/s="
            f"{result['generation_samples_per_second']:.3f} "
            f"end_to_end_samples/s={result['samples_per_second']:.3f}"
        )
        loader = None
    return results


def close_parent_lmdb(dataset: Any) -> None:
    """Ensure DataLoader workers open independent read-only LMDB handles."""
    image_reader = getattr(dataset, "image_reader", None)
    parent_lmdb = getattr(image_reader, "env", None)
    if parent_lmdb is not None:
        parent_lmdb.close()
        image_reader.env = None


def profile_generation_bucket(
    *,
    bucket: str,
    dataset: Any,
    dataset_indices: Sequence[int],
    module: Any,
    collator: Any,
    device: torch.device,
    batch_sizes: Sequence[int],
    max_new_tokens: int,
    throughput_samples: int,
    worker_counts: Sequence[int],
    total_memory_gb: float,
    memory_safety_fraction: float,
    capacity_resolution: int,
    near_best_fraction: float,
) -> dict[str, Any]:
    worst_index, instruction_tokens = longest_instruction_index(
        dataset,
        module.tokenizer,
        dataset_indices,
    )
    longest_target_index, target_tokens = longest_target(
        dataset,
        module.tokenizer,
        dataset_indices,
    )
    if target_tokens + 1 > max_new_tokens:
        raise RuntimeError(
            f"Bucket {bucket!r} generation cap {max_new_tokens} cannot fit its "
            f"longest {target_tokens}-token target plus EOS"
        )
    worst_sample = dataset[worst_index]
    print(
        f"\nGeneration bucket: {bucket}\n"
        f"rows={len(dataset_indices)}\n"
        f"worst_index={worst_index}\n"
        f"worst_sample_id={worst_sample.get('sample_id')}\n"
        "instruction_tokens_without_fixed_chat_or_image_tokens="
        f"{instruction_tokens}\n"
        f"generation_cap={max_new_tokens}"
        f"\nlongest_target_tokens_without_eos={target_tokens}"
    )
    capacity_results = capacity_profile(
        module=module,
        collator=collator,
        worst_sample=worst_sample,
        batch_sizes=batch_sizes,
        device=device,
        max_new_tokens=max_new_tokens,
        total_memory_gb=total_memory_gb,
        safety_fraction=memory_safety_fraction,
        resolution=capacity_resolution,
        bucket=bucket,
    )
    safe_batches = safe_capacity_batches(
        capacity_results,
        total_memory_gb=total_memory_gb,
        safety_fraction=memory_safety_fraction,
    )

    close_parent_lmdb(dataset)
    relative_indices = evenly_spaced_indices(len(dataset_indices), throughput_samples)
    selected_indices = [dataset_indices[index] for index in relative_indices]
    batch_selection_workers = max(worker_counts)
    throughput_results = bucket_throughput_profile(
        module=module,
        collator=collator,
        dataset=dataset,
        indices=selected_indices,
        batch_sizes=safe_batches,
        device=device,
        max_new_tokens=max_new_tokens,
        num_workers=batch_selection_workers,
        bucket=bucket,
    )
    recommended_batch = recommend_throughput_batch(
        throughput_results,
        safe_batches=safe_batches,
        near_best_fraction=near_best_fraction,
    )

    worker_results: list[dict[str, Any]] = []
    if recommended_batch is not None:
        for worker_count in worker_counts:
            if worker_count == batch_selection_workers:
                existing = next(
                    result
                    for result in throughput_results
                    if result.get("status") == "ok"
                    and int(result["batch_size"]) == recommended_batch
                )
                worker_results.append(existing)
                continue
            worker_results.extend(
                bucket_throughput_profile(
                    module=module,
                    collator=collator,
                    dataset=dataset,
                    indices=selected_indices,
                    batch_sizes=[recommended_batch],
                    device=device,
                    max_new_tokens=max_new_tokens,
                    num_workers=worker_count,
                    bucket=bucket,
                )
            )
        worker_results.sort(key=lambda result: int(result["num_workers"]))
    recommended_workers = recommend_worker_count(
        worker_results,
        near_best_fraction=near_best_fraction,
    )
    close_parent_lmdb(dataset)
    return {
        "rows": len(dataset_indices),
        "max_new_tokens": max_new_tokens,
        "longest_target": {
            "dataset_index": longest_target_index,
            "target_tokens_without_eos": target_tokens,
            "required_tokens_with_eos": target_tokens + 1,
        },
        "worst_instruction": {
            "dataset_index": worst_index,
            "sample_id": str(worst_sample.get("sample_id")),
            "instruction_tokens_without_fixed_chat_or_image_tokens": instruction_tokens,
        },
        "capacity_results": capacity_results,
        "safe_capacity_batches": safe_batches,
        "throughput_samples": len(selected_indices),
        "throughput_selection": "evenly_spaced_within_bucket",
        "batch_selection_num_workers": batch_selection_workers,
        "throughput_results": throughput_results,
        "worker_results_at_recommended_batch": worker_results,
        "recommended_batch_size": recommended_batch,
        "recommended_num_workers": recommended_workers,
    }


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for evaluation batch profiling")
    if not args.adapter_dir.is_dir():
        raise SystemExit(f"Adapter directory is not a directory: {args.adapter_dir}")

    from src.data_modules.ben_txt_datamodule import BENTxTDataModule
    from src.lightning_modules.qwen3_vl_module import Qwen3VLModule

    device = torch.device("cuda")
    torch.manual_seed(42)
    datamodule = BENTxTDataModule(
        image_lmdb_file=require_env("BIGEARTHNET_V2_LMDB_ROOT"),
        metadata_file=require_env("BIGEARTHNET_TXT_PARQUET_PATH"),
        bands="S1S2-10m20m",
        rgb_render_mode="copernicus",
        rgb_quantile=0.90,
        point_token=("", ""),
        ref_token=("<|object_ref_start|>", "<|object_ref_end|>"),
        grounding_format="qwen3_json",
        test_splits=("bench",),
        batch_size=1,
        num_workers_dataloader=0,
    )
    datamodule.setup("test")
    dataset = datamodule.test_ds
    if dataset is None:
        raise SystemExit("Failed to initialize the bench dataset")

    model_name = MODEL_NAMES[args.size]
    module = Qwen3VLModule(
        model_name_or_path=model_name,
        adapter_dir=str(args.adapter_dir),
        max_seq_length=2048,
        max_new_tokens=max(DEFAULT_MAX_NEW_TOKENS_BY_BUCKET.values()),
        generation_max_new_tokens_by_bucket=DEFAULT_MAX_NEW_TOKENS_BY_BUCKET,
        prediction_export_path="profile-only.jsonl",
        loc_mode="loc_embed",
        location_embed_marker="Scene coordinates:",
        non_rgb_conditioning="enabled",
        non_rgb_encoder_dir=require_env("BIGEARTHNET_ENCODER_DIR"),
        non_rgb_feature_mode="spatial_4x4",
        non_rgb_spatial_pool_size=4,
        num_non_rgb_tokens=16,
        satclip_checkpoint=require_env("SATCLIP_L40_CHECKPOINT_PATH"),
        satclip_dim=256,
        num_location_tokens=8,
        location_projection_lr_multiplier=5.0,
        model_size=args.size,
    )
    module.setup("test")
    module.to(device)
    module.eval()
    FastVisionModel.for_inference(module.model)
    collator = module._test_collator
    if collator is None:
        raise SystemExit("Prompt-only evaluation collator was not initialized")

    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / GIB
    print(
        "Task-aware evaluation capacity and throughput profile\n"
        f"model={model_name}\n"
        f"adapter={args.adapter_dir}\n"
        f"bench_rows={len(dataset)}\n"
        f"gpu_total_memory_gb={total_memory_gb:.2f}\n"
        f"memory_safety_fraction={args.memory_safety_fraction:.2f}"
    )
    task_types = dataset.text_data["type"].astype(str).tolist()
    indices_by_bucket = bucket_indices(task_types)
    bucket_results = {
        bucket: profile_generation_bucket(
            bucket=bucket,
            dataset=dataset,
            dataset_indices=indices_by_bucket[bucket],
            module=module,
            collator=collator,
            device=device,
            batch_sizes=args.batch_sizes,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS_BY_BUCKET[bucket],
            throughput_samples=args.throughput_samples,
            worker_counts=args.worker_counts,
            total_memory_gb=total_memory_gb,
            memory_safety_fraction=args.memory_safety_fraction,
            capacity_resolution=args.capacity_resolution,
            near_best_fraction=args.throughput_near_best_fraction,
        )
        for bucket in GENERATION_BUCKETS
    }

    summary = {
        "model": model_name,
        "model_size": args.size,
        "adapter_dir": str(args.adapter_dir),
        "condition": "loc_embed",
        "split": "bench",
        "bench_rows": len(dataset),
        "generation_max_new_tokens_by_bucket": DEFAULT_MAX_NEW_TOKENS_BY_BUCKET,
        "gpu_total_memory_gb": total_memory_gb,
        "memory_safety_fraction": args.memory_safety_fraction,
        "memory_limit_gb": total_memory_gb * args.memory_safety_fraction,
        "capacity_resolution": args.capacity_resolution,
        "throughput_near_best_fraction": args.throughput_near_best_fraction,
        "buckets": bucket_results,
        "recommended_batch_sizes": {
            bucket: result["recommended_batch_size"] for bucket, result in bucket_results.items()
        },
        "recommended_num_workers": {
            bucket: result["recommended_num_workers"] for bucket, result in bucket_results.items()
        },
    }
    output_path = args.json_output or Path(
        f"outputs/batch_profiles/profile_evaluation_{args.size}_loc_embed.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
