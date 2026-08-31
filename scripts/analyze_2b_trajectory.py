#!/usr/bin/env python3
"""Build tables, uncertainty intervals, and plots for the 2B trajectory matrix."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from notebooks.utils.paired_bootstrap import (
    DIRECT_GEOGRAPHY_METRICS,
    PRIMARY_METRICS,
    PredictionComparison,
    analyze_comparisons,
)

STEP_ORDER = ("50", "100", "500", "1000", "5000", "final")
CONDITION_ORDER = ("no_loc", "loc_text", "loc_embed", "loc_additive_satclip")
LOCATION_CONDITIONS = CONDITION_ORDER[1:]
SEEDS = (42, 43)
EFFECTIVE_BATCH = 128


@dataclass(frozen=True)
class TrajectoryRun:
    evaluation_job: str
    fit_job: str
    condition: str
    seed: int
    step: str
    coordinate_setting: str
    adapter_dir: str
    run_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evaluation-root", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/analysis/trajectory_2b"),
    )
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--skip-bootstrap", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[TrajectoryRun]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            TrajectoryRun(**(row | {"seed": int(row["seed"])}))
            for row in csv.DictReader(handle)
        ]
    validate_manifest(rows)
    return rows


def validate_manifest(rows: list[TrajectoryRun]) -> None:
    expected = {
        (seed, condition, step, "correct")
        for seed in SEEDS
        for condition in CONDITION_ORDER
        for step in STEP_ORDER
    }
    expected |= {
        (seed, condition, step, "shuffled")
        for seed in SEEDS
        for condition in LOCATION_CONDITIONS
        for step in ("1000", "final")
    }
    observed = {(row.seed, row.condition, row.step, row.coordinate_setting) for row in rows}
    if len(observed) != len(rows):
        raise ValueError("trajectory manifest contains duplicate condition/seed/step/setting rows")
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise ValueError(
            f"trajectory manifest grid mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    jobs = [row.evaluation_job for row in rows]
    if len(set(jobs)) != len(jobs):
        raise ValueError("trajectory manifest contains duplicate evaluation job IDs")


def _summary_path(evaluation_root: Path, job: str) -> Path:
    return evaluation_root / job / "scored_predictions" / "summary.json"


def _prediction_path(evaluation_root: Path, job: str) -> Path:
    return evaluation_root / job / "predictions.jsonl"


def _task_row(summary: dict[str, Any], task_type: str) -> dict[str, Any]:
    return next(row for row in summary["by_task_type"] if row["task_type"] == task_type)


def _headline_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "caption_bleu4": float(summary["captioning"]["bleu4"]),
        "binary_accuracy": float(_task_row(summary, "binary")["accuracy"]),
        "mcq_accuracy": float(_task_row(summary, "mcq")["accuracy"]),
        "bbox_miou": float(_task_row(summary, "bounding box")["miou"]),
    }


def build_metric_table(rows: list[TrajectoryRun], evaluation_root: Path) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for row in rows:
        path = _summary_path(evaluation_root, row.evaluation_job)
        if not path.is_file():
            raise FileNotFoundError(f"missing trajectory summary for job {row.evaluation_job}: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        numeric_step = None if row.step == "final" else int(row.step)
        output.append(
            {
                "evaluation_job": row.evaluation_job,
                "fit_job": row.fit_job,
                "condition": row.condition,
                "seed": row.seed,
                "step": row.step,
                "optimizer_step": numeric_step,
                "examples_seen": None if numeric_step is None else numeric_step * EFFECTIVE_BATCH,
                "coordinate_setting": row.coordinate_setting,
                **_headline_metrics(summary),
            }
        )
    return pd.DataFrame(output)


def build_difference_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = [metric.name for metric in PRIMARY_METRICS]
    correct = metrics[metrics["coordinate_setting"] == "correct"].copy()
    baseline = correct[correct["condition"] == "no_loc"][
        ["seed", "step", *metric_names]
    ].rename(columns={metric: f"baseline_{metric}" for metric in metric_names})
    gains = correct[correct["condition"].isin(LOCATION_CONDITIONS)].merge(
        baseline,
        on=["seed", "step"],
        validate="many_to_one",
    )
    for metric in metric_names:
        gains[f"delta_{metric}"] = gains[metric] - gains[f"baseline_{metric}"]

    shuffled = metrics[metrics["coordinate_setting"] == "shuffled"].copy()
    reliance = shuffled.merge(
        correct,
        on=["seed", "condition", "step"],
        suffixes=("_shuffled", "_correct"),
        validate="one_to_one",
    )
    for metric in metric_names:
        reliance[f"delta_{metric}_shuffled_minus_correct"] = (
            reliance[f"{metric}_shuffled"] - reliance[f"{metric}_correct"]
        )
    return gains, reliance


def build_comparisons(rows: list[TrajectoryRun], evaluation_root: Path) -> list[PredictionComparison]:
    by_key = {
        (row.seed, row.condition, row.step, row.coordinate_setting): row for row in rows
    }
    comparisons: list[PredictionComparison] = []
    for seed in SEEDS:
        for step in STEP_ORDER:
            baseline = by_key[(seed, "no_loc", step, "correct")]
            for condition in LOCATION_CONDITIONS:
                location = by_key[(seed, condition, step, "correct")]
                comparisons.append(
                    PredictionComparison(
                        name=f"2B step {step} seed {seed}: {condition} minus no_loc",
                        system_a=f"{condition}-correct",
                        predictions_a=_prediction_path(
                            evaluation_root, location.evaluation_job
                        ),
                        system_b="no_loc",
                        predictions_b=_prediction_path(
                            evaluation_root, baseline.evaluation_job
                        ),
                        kind="trajectory_core",
                    )
                )
        for step in ("1000", "final"):
            for condition in LOCATION_CONDITIONS:
                correct = by_key[(seed, condition, step, "correct")]
                shuffled = by_key[(seed, condition, step, "shuffled")]
                comparisons.append(
                    PredictionComparison(
                        name=f"2B step {step} seed {seed}: {condition} shuffled minus correct",
                        system_a=f"{condition}-shuffled",
                        predictions_a=_prediction_path(
                            evaluation_root, shuffled.evaluation_job
                        ),
                        system_b=f"{condition}-correct",
                        predictions_b=_prediction_path(
                            evaluation_root, correct.evaluation_job
                        ),
                        kind="trajectory_counterfactual",
                    )
                )
    return comparisons


def plot_trajectory_gains(gains: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metric_labels = {
        "caption_bleu4": "BLEU-4 difference",
        "binary_accuracy": "Binary accuracy difference",
        "mcq_accuracy": "MCQ accuracy difference",
        "bbox_miou": "BBox mIoU difference",
    }
    colors = {
        "loc_text": "#D08770",
        "loc_embed": "#5E81AC",
        "loc_additive_satclip": "#A3BE8C",
    }
    labels = {
        "loc_text": "Coordinate text",
        "loc_embed": "Location tokens",
        "loc_additive_satclip": "Additive SatCLIP",
    }
    for seed in SEEDS:
        selected = gains[gains["seed"] == seed].copy()
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True)
        for ax, (metric, label) in zip(axes.flat, metric_labels.items(), strict=True):
            for condition in LOCATION_CONDITIONS:
                condition_rows = selected[selected["condition"] == condition].set_index("step")
                condition_rows = condition_rows.reindex(STEP_ORDER)
                ax.plot(
                    range(len(STEP_ORDER)),
                    condition_rows[f"delta_{metric}"],
                    marker="o",
                    color=colors[condition],
                    label=labels[condition],
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(label)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(range(len(STEP_ORDER)), STEP_ORDER)
        axes[0, 0].legend(frameon=False)
        fig.supxlabel("Saved optimizer-step milestone")
        fig.supylabel("Condition minus matched no_loc")
        fig.tight_layout()
        path = output_dir / f"trajectory_gains_seed{seed}.png"
        fig.savefig(path, dpi=240)
        plt.close(fig)
        print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_metric_table(rows, args.evaluation_root)
    gains, reliance = build_difference_tables(metrics)
    metrics.to_csv(args.output_dir / "trajectory_metrics.csv", index=False)
    gains.to_csv(args.output_dir / "trajectory_gains_over_no_loc.csv", index=False)
    reliance.to_csv(args.output_dir / "trajectory_coordinate_reliance.csv", index=False)
    plot_trajectory_gains(gains, args.output_dir)

    if not args.skip_bootstrap:
        intervals = analyze_comparisons(
            build_comparisons(rows, args.evaluation_root),
            metrics=PRIMARY_METRICS + DIRECT_GEOGRAPHY_METRICS,
            n_resamples=args.n_resamples,
            seed=args.bootstrap_seed,
        )
        intervals.to_csv(args.output_dir / "trajectory_paired_intervals.csv", index=False)
        print(f"Wrote {args.output_dir / 'trajectory_paired_intervals.csv'}")


if __name__ == "__main__":
    main()
