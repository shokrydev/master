#!/usr/bin/env python3
"""Profile CLAIR judge batch sizes with one model load and real caption prompts."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

from scripts.score_bentxt_clair import DEFAULT_JUDGE, _generate_batch, _load_judge
from src.evaluation.bentxt_records import load_predictions_jsonl
from src.evaluation.clair import caption_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name-or-path", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 112, 128],
    )
    parser.add_argument("--profile-rows", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--vram-fraction", type=float, default=0.95)
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if not args.predictions.is_file():
        raise ValueError(f"predictions do not exist: {args.predictions}")
    if not args.batch_sizes or any(size <= 0 for size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if args.batch_sizes != sorted(set(args.batch_sizes)):
        raise ValueError("batch sizes must be unique and strictly increasing")
    if args.profile_rows < max(args.batch_sizes):
        raise ValueError("profile rows must be at least the largest batch size")
    if not 0 < args.vram_fraction <= 1:
        raise ValueError("VRAM fraction must be in (0, 1]")


def _clear_cuda(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _representative_records(records: list[Any], count: int) -> list[Any]:
    """Deterministically span the observed caption/prompt-length distribution."""
    ordered = sorted(
        records,
        key=lambda row: len(row.prediction) + sum(map(len, row.target_texts)),
    )
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[-1]]
    return [ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)]


def main() -> None:
    args = parse_args()
    _validate(args)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for CLAIR batch profiling")
    records = caption_records(load_predictions_jsonl(args.predictions))
    if len(records) < args.profile_rows:
        raise ValueError(f"requested {args.profile_rows} rows, found {len(records)} captions")
    profile_records = _representative_records(records, args.profile_rows)

    model, processor, text_tokenizer = _load_judge(
        args.model_name_or_path,
        args.max_sequence_length,
    )
    device = torch.device("cuda")
    total_vram = torch.cuda.get_device_properties(device).total_memory
    results: list[dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        _clear_cuda(torch)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        completed = 0
        batch_seconds: list[float] = []
        status = "ok"
        error = None
        try:
            for offset in range(0, len(profile_records), batch_size):
                current = profile_records[offset : offset + batch_size]
                rows = _generate_batch(
                    current,
                    model=model,
                    processor=processor,
                    text_tokenizer=text_tokenizer,
                    max_sequence_length=args.max_sequence_length,
                    max_new_tokens=args.max_new_tokens,
                )
                completed += len(rows)
                batch_seconds.append(rows[0]["timings"]["batch_seconds"])
        except torch.OutOfMemoryError as exc:
            status = "oom"
            error = str(exc)
        elapsed = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_allocated(device)
        peak_reserved_vram = torch.cuda.max_memory_reserved(device)
        result = {
            "batch_size": batch_size,
            "status": status,
            "completed_rows": completed,
            "elapsed_seconds": elapsed,
            "generation_seconds": sum(batch_seconds),
            "end_to_end_rows_per_second": completed / elapsed if completed else 0.0,
            "generation_rows_per_second": completed / sum(batch_seconds)
            if batch_seconds
            else 0.0,
            "peak_vram_bytes": peak_vram,
            "peak_vram_fraction": peak_vram / total_vram,
            "peak_reserved_vram_bytes": peak_reserved_vram,
            "peak_reserved_vram_fraction": peak_reserved_vram / total_vram,
            "error": error,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        if status == "oom":
            _clear_cuda(torch)
            break

    safe = [
        result
        for result in results
        if result["status"] == "ok"
        and result["peak_reserved_vram_fraction"] <= args.vram_fraction
        and result["completed_rows"] == args.profile_rows
    ]
    if not safe:
        raise RuntimeError("no candidate completed within the configured VRAM limit")
    best_throughput = max(item["end_to_end_rows_per_second"] for item in safe)
    near_best = [
        item for item in safe if item["end_to_end_rows_per_second"] >= 0.98 * best_throughput
    ]
    recommendation = min(near_best, key=lambda item: item["batch_size"])
    payload = {
        "predictions": str(args.predictions.resolve()),
        "model_name_or_path": args.model_name_or_path,
        "profile_rows": args.profile_rows,
        "selection_rule": (
            "smallest batch within 98% of best end-to-end throughput, among complete "
            f"runs using at most {args.vram_fraction:.0%} of VRAM"
        ),
        "recommended_batch_size": recommendation["batch_size"],
        "total_vram_bytes": total_vram,
        "candidate_median_throughput": statistics.median(
            item["end_to_end_rows_per_second"] for item in safe
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Recommended CLAIR batch size: {recommendation['batch_size']}", flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
