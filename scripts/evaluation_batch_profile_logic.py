"""Pure recommendation logic for the evaluation generation batch profiler."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def evenly_spaced_indices(population_size: int, sample_size: int) -> list[int]:
    if population_size <= 0 or sample_size <= 0:
        raise ValueError("population_size and sample_size must be positive")
    if sample_size >= population_size:
        return list(range(population_size))
    return [(index * population_size) // sample_size for index in range(sample_size)]


def refinement_batch(lower: int, upper: int, resolution: int) -> int | None:
    if lower < 0 or upper <= lower or resolution <= 0:
        raise ValueError("Require 0 <= lower < upper and positive resolution")
    if upper - lower <= resolution:
        return None
    midpoint = ((lower + upper) // 2 // resolution) * resolution
    if midpoint <= lower:
        midpoint = lower + resolution
    if midpoint >= upper:
        return None
    return midpoint


def safe_capacity_batches(
    results: Sequence[Mapping[str, Any]],
    *,
    total_memory_gb: float,
    safety_fraction: float,
) -> list[int]:
    limit_gb = total_memory_gb * safety_fraction
    return sorted(
        int(result["batch_size"])
        for result in results
        if result.get("status") == "ok" and float(result["peak_reserved_gb"]) <= limit_gb
    )


def recommend_throughput_batch(
    results: Sequence[Mapping[str, Any]],
    *,
    safe_batches: Sequence[int],
    near_best_fraction: float,
) -> int | None:
    safe = set(safe_batches)
    eligible = [
        result
        for result in results
        if result.get("status") == "ok" and int(result["batch_size"]) in safe
    ]
    if not eligible:
        return None
    best_rate = max(float(result["samples_per_second"]) for result in eligible)
    threshold = best_rate * near_best_fraction
    return min(
        int(result["batch_size"])
        for result in eligible
        if float(result["samples_per_second"]) >= threshold
    )


def recommend_worker_count(
    results: Sequence[Mapping[str, Any]],
    *,
    near_best_fraction: float,
) -> int | None:
    eligible = [result for result in results if result.get("status") == "ok"]
    if not eligible:
        return None
    best_rate = max(float(result["samples_per_second"]) for result in eligible)
    threshold = best_rate * near_best_fraction
    return min(
        int(result["num_workers"])
        for result in eligible
        if float(result["samples_per_second"]) >= threshold
    )
