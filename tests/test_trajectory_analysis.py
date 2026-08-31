from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_2b_trajectory import (
    CONDITION_ORDER,
    LOCATION_CONDITIONS,
    SEEDS,
    STEP_ORDER,
    TrajectoryRun,
    build_comparisons,
    build_difference_tables,
    validate_manifest,
)


def _trajectory_rows() -> list[TrajectoryRun]:
    rows: list[TrajectoryRun] = []
    job = 20000
    for seed in SEEDS:
        for condition in CONDITION_ORDER:
            for step in STEP_ORDER:
                job += 1
                rows.append(
                    TrajectoryRun(
                        str(job),
                        "fit",
                        condition,
                        seed,
                        step,
                        "correct",
                        "/adapter",
                        "run",
                    )
                )
            if condition in LOCATION_CONDITIONS:
                for step in ("1000", "final"):
                    job += 1
                    rows.append(
                        TrajectoryRun(
                            str(job),
                            "fit",
                            condition,
                            seed,
                            step,
                            "shuffled",
                            "/adapter",
                            "run",
                        )
                    )
    return rows


def test_trajectory_grid_and_comparison_inventory():
    rows = _trajectory_rows()
    validate_manifest(rows)
    comparisons = build_comparisons(rows, Path("outputs/evaluation"))
    assert len(rows) == 60
    assert len(comparisons) == 48
    assert sum(comparison.kind == "trajectory_core" for comparison in comparisons) == 36
    assert sum(comparison.kind == "trajectory_counterfactual" for comparison in comparisons) == 12

    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(rows + [rows[0]])


def test_trajectory_differences_have_declared_direction():
    metrics = pd.DataFrame(
        [
            {
                "evaluation_job": "1",
                "fit_job": "fit",
                "condition": "no_loc",
                "seed": 42,
                "step": "1000",
                "coordinate_setting": "correct",
                "caption_bleu4": 0.4,
                "binary_accuracy": 0.7,
                "mcq_accuracy": 0.6,
                "bbox_miou": 0.5,
            },
            {
                "evaluation_job": "2",
                "fit_job": "fit",
                "condition": "loc_text",
                "seed": 42,
                "step": "1000",
                "coordinate_setting": "correct",
                "caption_bleu4": 0.5,
                "binary_accuracy": 0.72,
                "mcq_accuracy": 0.65,
                "bbox_miou": 0.55,
            },
            {
                "evaluation_job": "3",
                "fit_job": "fit",
                "condition": "loc_text",
                "seed": 42,
                "step": "1000",
                "coordinate_setting": "shuffled",
                "caption_bleu4": 0.45,
                "binary_accuracy": 0.71,
                "mcq_accuracy": 0.61,
                "bbox_miou": 0.51,
            },
        ]
    )
    gains, reliance = build_difference_tables(metrics)
    assert gains.iloc[0]["delta_caption_bleu4"] == pytest.approx(0.1)
    assert reliance.iloc[0]["delta_caption_bleu4_shuffled_minus_correct"] == pytest.approx(
        -0.05
    )
