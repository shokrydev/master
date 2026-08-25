"""Paired patch-cluster bootstrap analysis for BEN.txt predictions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.bentxt_records import BENTxTPrediction, load_predictions_jsonl
from src.evaluation.bentxt_scoring import SampleScore, score_predictions

_BLEU_SMALL = 1e-9
_BLEU_TINY = 1e-15


@dataclass(frozen=True)
class PredictionComparison:
    """Two aligned prediction exports compared as system A minus system B."""

    name: str
    system_a: str
    predictions_a: Path
    system_b: str
    predictions_b: Path
    kind: str


@dataclass(frozen=True)
class MetricSpec:
    """One benchmark metric evaluated on a task family or category."""

    name: str
    task_type: str
    score_field: str
    task_category: str | None = None


PRIMARY_METRICS = (
    MetricSpec("caption_bleu4", "captioning", "bleu4"),
    MetricSpec("binary_accuracy", "binary", "correct"),
    MetricSpec("mcq_accuracy", "mcq", "correct"),
    MetricSpec("bbox_miou", "bounding box", "iou"),
)

DIRECT_GEOGRAPHY_METRICS = (
    MetricSpec("mcq_climate_zone_accuracy", "mcq", "correct", "climate zone"),
    MetricSpec("mcq_country_accuracy", "mcq", "correct", "country"),
    MetricSpec("mcq_season_accuracy", "mcq", "correct", "season"),
)


def _records_by_id(
    records: Sequence[BENTxTPrediction], *, source: Path
) -> dict[str, BENTxTPrediction]:
    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"Duplicate sample_id values in {source}")
    return by_id


def _align_prediction_exports(
    comparison: PredictionComparison,
) -> tuple[list[BENTxTPrediction], list[BENTxTPrediction]]:
    records_a = load_predictions_jsonl(comparison.predictions_a)
    records_b = load_predictions_jsonl(comparison.predictions_b)
    by_id_a = _records_by_id(records_a, source=comparison.predictions_a)
    by_id_b = _records_by_id(records_b, source=comparison.predictions_b)

    ids_a = set(by_id_a)
    ids_b = set(by_id_b)
    if ids_a != ids_b:
        raise ValueError(
            f"Prediction populations differ for {comparison.name}: "
            f"{len(ids_a - ids_b)} rows only in A and {len(ids_b - ids_a)} rows only in B"
        )

    aligned_a: list[BENTxTPrediction] = []
    aligned_b: list[BENTxTPrediction] = []
    metadata_fields = ("patch_id", "task_type", "task_category", "split", "target_texts")
    for sample_id in sorted(ids_a):
        record_a = by_id_a[sample_id]
        record_b = by_id_b[sample_id]
        for field in metadata_fields:
            if getattr(record_a, field) != getattr(record_b, field):
                raise ValueError(
                    f"Mismatched {field} for sample_id {sample_id!r} in {comparison.name}"
                )
        aligned_a.append(record_a)
        aligned_b.append(record_b)

    return aligned_a, aligned_b


def _caption_bleu_statistics(record: BENTxTPrediction) -> np.ndarray:
    """Return the sufficient statistics used by pycocoevalcap corpus BLEU."""
    prediction_words = record.prediction.split()
    prediction_length = len(prediction_words)
    guesses = [max(0, prediction_length - order + 1) for order in range(1, 5)]

    prediction_counts: Counter[tuple[str, ...]] = Counter()
    for order in range(1, 5):
        prediction_counts.update(
            tuple(prediction_words[start : start + order])
            for start in range(prediction_length - order + 1)
        )

    reference_lengths: list[int] = []
    maximum_reference_counts: dict[tuple[str, ...], int] = {}
    for reference in record.target_texts:
        reference_words = reference.split()
        reference_lengths.append(len(reference_words))
        reference_counts: Counter[tuple[str, ...]] = Counter()
        for order in range(1, 5):
            reference_counts.update(
                tuple(reference_words[start : start + order])
                for start in range(len(reference_words) - order + 1)
            )
        for ngram, count in reference_counts.items():
            maximum_reference_counts[ngram] = max(
                maximum_reference_counts.get(ngram, 0),
                count,
            )

    correct = [0, 0, 0, 0]
    for ngram, count in prediction_counts.items():
        correct[len(ngram) - 1] += min(maximum_reference_counts.get(ngram, 0), count)

    reference_length = min(
        (abs(length - prediction_length), length) for length in reference_lengths
    )[1]
    return np.asarray(
        [prediction_length, reference_length, *guesses, *correct],
        dtype=np.float64,
    )


def _bleu4_from_totals(totals: np.ndarray) -> np.ndarray:
    totals = np.asarray(totals, dtype=np.float64)
    was_vector = totals.ndim == 1
    if was_vector:
        totals = totals[np.newaxis, :]

    test_length = totals[:, 0]
    reference_length = totals[:, 1]
    guesses = totals[:, 2:6]
    correct = totals[:, 6:10]
    bleu = np.prod((correct + _BLEU_TINY) / (guesses + _BLEU_SMALL), axis=1) ** 0.25
    ratio = (test_length + _BLEU_TINY) / (reference_length + _BLEU_SMALL)
    shorter = ratio < 1.0
    bleu[shorter] *= np.exp(1.0 - (1.0 / ratio[shorter]))
    return bleu[0] if was_vector else bleu


def _aggregate_rows_by_patch(
    row_values: np.ndarray,
    row_patch_indices: np.ndarray,
    n_patches: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(row_patch_indices, minlength=n_patches).astype(np.float64)
    totals = np.bincount(
        row_patch_indices,
        weights=row_values,
        minlength=n_patches,
    )
    return totals, counts


def _metric_rows(
    records: Sequence[BENTxTPrediction],
    scores: dict[str, SampleScore],
    metric: MetricSpec,
) -> tuple[list[int], np.ndarray]:
    row_indices: list[int] = []
    values: list[float] = []
    for index, record in enumerate(records):
        if record.task_type != metric.task_type:
            continue
        if metric.task_category is not None and record.task_category != metric.task_category:
            continue
        if metric.score_field == "bleu4":
            raise ValueError("BLEU-4 is handled as a corpus metric")

        score = scores[record.sample_id]
        value = getattr(score, metric.score_field)
        row_indices.append(index)
        values.append(float(value or 0.0))

    if not row_indices:
        raise ValueError(f"No rows found for metric {metric.name!r}")
    return row_indices, np.asarray(values, dtype=np.float64)


def _bootstrap_scalar_metric(
    *,
    records_a: Sequence[BENTxTPrediction],
    records_b: Sequence[BENTxTPrediction],
    scores_a: dict[str, SampleScore],
    scores_b: dict[str, SampleScore],
    metric: MetricSpec,
    patch_index: dict[str, int],
    bootstrap_weights: np.ndarray,
) -> tuple[float, float, np.ndarray, int, int]:
    row_indices, values_a = _metric_rows(records_a, scores_a, metric)
    row_indices_b, values_b = _metric_rows(records_b, scores_b, metric)
    if row_indices != row_indices_b:
        raise ValueError(f"Metric populations differ for {metric.name!r}")

    patch_indices = np.fromiter(
        (patch_index[records_a[index].patch_id] for index in row_indices),
        dtype=np.int64,
    )
    totals_a, counts = _aggregate_rows_by_patch(values_a, patch_indices, len(patch_index))
    totals_b, counts_b = _aggregate_rows_by_patch(values_b, patch_indices, len(patch_index))
    if not np.array_equal(counts, counts_b):
        raise ValueError(f"Metric cluster populations differ for {metric.name!r}")

    bootstrap_counts = bootstrap_weights @ counts
    if np.any(bootstrap_counts == 0):
        raise ValueError(f"A bootstrap replicate contained no rows for {metric.name!r}")
    bootstrap_a = (bootstrap_weights @ totals_a) / bootstrap_counts
    bootstrap_b = (bootstrap_weights @ totals_b) / bootstrap_counts
    return (
        float(values_a.mean()),
        float(values_b.mean()),
        bootstrap_a - bootstrap_b,
        len(row_indices),
        int(np.count_nonzero(counts)),
    )


def _bootstrap_caption_bleu4(
    *,
    records_a: Sequence[BENTxTPrediction],
    records_b: Sequence[BENTxTPrediction],
    patch_index: dict[str, int],
    bootstrap_weights: np.ndarray,
) -> tuple[float, float, np.ndarray, int, int]:
    caption_indices = [
        index for index, record in enumerate(records_a) if record.task_type == "captioning"
    ]
    if not caption_indices:
        raise ValueError("No captioning rows found")

    n_patches = len(patch_index)
    patch_statistics_a = np.zeros((n_patches, 10), dtype=np.float64)
    patch_statistics_b = np.zeros((n_patches, 10), dtype=np.float64)
    patches_with_captions: set[str] = set()
    for index in caption_indices:
        record_a = records_a[index]
        record_b = records_b[index]
        patch_position = patch_index[record_a.patch_id]
        patches_with_captions.add(record_a.patch_id)
        patch_statistics_a[patch_position] += _caption_bleu_statistics(record_a)
        patch_statistics_b[patch_position] += _caption_bleu_statistics(record_b)

    totals_a = patch_statistics_a.sum(axis=0)
    totals_b = patch_statistics_b.sum(axis=0)
    bootstrap_a = _bleu4_from_totals(bootstrap_weights @ patch_statistics_a)
    bootstrap_b = _bleu4_from_totals(bootstrap_weights @ patch_statistics_b)
    return (
        float(_bleu4_from_totals(totals_a)),
        float(_bleu4_from_totals(totals_b)),
        bootstrap_a - bootstrap_b,
        len(caption_indices),
        len(patches_with_captions),
    )


def paired_cluster_bootstrap(
    comparison: PredictionComparison,
    *,
    metrics: Sequence[MetricSpec] = PRIMARY_METRICS,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate paired A-minus-B differences by resampling BEN.txt patches."""
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    records_a, records_b = _align_prediction_exports(comparison)
    patches = sorted({record.patch_id for record in records_a})
    patch_index = {patch_id: index for index, patch_id in enumerate(patches)}

    rng = np.random.default_rng(seed)
    bootstrap_weights = rng.multinomial(
        len(patches),
        np.full(len(patches), 1.0 / len(patches)),
        size=n_resamples,
    )

    scores_a = {score.sample_id: score for score in score_predictions(records_a)}
    scores_b = {score.sample_id: score for score in score_predictions(records_b)}
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, object]] = []

    for metric in metrics:
        if metric.score_field == "bleu4":
            score_a, score_b, differences, n_rows, n_metric_patches = _bootstrap_caption_bleu4(
                records_a=records_a,
                records_b=records_b,
                patch_index=patch_index,
                bootstrap_weights=bootstrap_weights,
            )
        else:
            score_a, score_b, differences, n_rows, n_metric_patches = _bootstrap_scalar_metric(
                records_a=records_a,
                records_b=records_b,
                scores_a=scores_a,
                scores_b=scores_b,
                metric=metric,
                patch_index=patch_index,
                bootstrap_weights=bootstrap_weights,
            )

        interval_low, interval_high = np.quantile(differences, [alpha, 1.0 - alpha])
        rows.append(
            {
                "comparison": comparison.name,
                "kind": comparison.kind,
                "system_a": comparison.system_a,
                "system_b": comparison.system_b,
                "metric": metric.name,
                "score_a": score_a,
                "score_b": score_b,
                "difference_a_minus_b": score_a - score_b,
                "ci_low": float(interval_low),
                "ci_high": float(interval_high),
                "ci_excludes_zero": bool(interval_low > 0.0 or interval_high < 0.0),
                "n_rows": n_rows,
                "n_patches_total": len(patches),
                "n_patches_with_metric": n_metric_patches,
                "confidence_level": confidence_level,
                "n_resamples": n_resamples,
                "bootstrap_seed": seed,
            }
        )

    return pd.DataFrame(rows)


def analyze_comparisons(
    comparisons: Sequence[PredictionComparison],
    *,
    metrics: Sequence[MetricSpec] = PRIMARY_METRICS,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the same paired bootstrap protocol for several comparisons."""
    if not comparisons:
        raise ValueError("At least one prediction comparison is required")
    return pd.concat(
        [
            paired_cluster_bootstrap(
                comparison,
                metrics=metrics,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=seed,
            )
            for comparison in comparisons
        ],
        ignore_index=True,
    )
