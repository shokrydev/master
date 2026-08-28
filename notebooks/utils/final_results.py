"""Build the fixed tables and figures used by the thesis Results chapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "outputs" / "evaluation"
FINETUNING_ROOT = REPO_ROOT / "outputs" / "finetuning"
ANALYSIS_ROOT = REPO_ROOT / "notebooks" / "analysis"
TABLES_ROOT = ANALYSIS_ROOT / "tables"
FIGURES_ROOT = ANALYSIS_ROOT / "figures"

METRIC_LABELS = {
    "bleu4": "BLEU-4",
    "binary_accuracy": "Binary accuracy",
    "mcq_accuracy": "MCQ accuracy",
    "bbox_miou": "BBox mIoU",
}

INTERVAL_METRIC_LABELS = {
    "caption_bleu4": "BLEU-4",
    "binary_accuracy": "Binary accuracy",
    "mcq_accuracy": "MCQ accuracy",
    "bbox_miou": "BBox mIoU",
}


@dataclass(frozen=True)
class CoreRun:
    model_size: str
    seed: int
    condition: str
    correct_job: str
    shuffled_job: str | None = None


CORE_RUNS = (
    CoreRun("2B", 42, "no_loc", "11437"),
    CoreRun("2B", 42, "loc_text", "11441", "11445"),
    CoreRun("2B", 42, "loc_embed", "11438", "11446"),
    CoreRun("2B", 43, "no_loc", "11622"),
    CoreRun("2B", 43, "loc_text", "11624", "11625"),
    CoreRun("2B", 43, "loc_embed", "11627", "11628"),
    CoreRun("4B", 42, "no_loc", "11442"),
    CoreRun("4B", 42, "loc_text", "11443", "11447"),
    CoreRun("4B", 42, "loc_embed", "11444", "11448"),
    CoreRun("8B", 42, "no_loc", "11680"),
    CoreRun("8B", 42, "loc_text", "11682", "11683"),
    CoreRun("8B", 42, "loc_embed", "11685", "11686"),
)

DEVELOPMENT_RUNS = (
    ("no_loc", "11401"),
    ("loc_text, two decimals", "11402"),
    ("loc_text, integer", "11409"),
    ("loc_embed, L10, 8 tokens, 5x LR", "11403"),
    ("loc_embed, L10, 4 tokens, 5x LR", "11404"),
    ("loc_embed, L10, 8 tokens, 2x LR", "11405"),
    ("loc_embed, L10, 4 tokens, 2x LR", "11406"),
    ("loc_embed, L40, 8 tokens, 5x LR", "11410"),
    ("loc_embed, L40, 4 tokens, 2x LR", "11411"),
)

ABLATION_RUNS = (
    ("no_loc", "1000 steps", 42, "11401"),
    ("loc_embed", "1000 steps", 42, "11410"),
    ("loc_encoding, all visual", "1000 steps", 42, "11486"),
    ("loc_encoding, S1/S2", "1000 steps", 42, "11495"),
    ("projected direct, S1/S2", "1000 steps", 42, "11555"),
    ("additive SatCLIP, S1/S2", "1000 steps", 42, "11558"),
    ("loc_embed, geolocation marker", "1000 steps", 42, "11561"),
    ("loc_embed, compact projector", "1000 steps", 42, "11593"),
    ("RGB only", "1000 steps", 42, "11492"),
    ("no_loc", "1000 steps", 43, "11596"),
    ("loc_embed", "1000 steps", 43, "11601"),
    ("additive SatCLIP, S1/S2", "1000 steps", 43, "11607"),
    ("loc_embed, compact projector", "1000 steps", 43, "11604"),
    ("no_loc", "full epoch", 42, "11437"),
    ("loc_embed", "full epoch", 42, "11438"),
    ("additive SatCLIP, S1/S2", "full epoch", 42, "11619"),
)

LIMITED_BUDGET_RUNS = {
    "11383": "no_loc",
    "11400": "loc_text",
    "11398": "loc_embed",
}

CONDITION_LABELS = {
    "no_loc": "No location",
    "loc_text": "Coordinate text",
    "loc_embed": "Location tokens",
}


def _summary(job_id: str) -> dict[str, object]:
    path = EVALUATION_ROOT / job_id / "scored_predictions" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing scored summary for evaluation job {job_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _task_row(summary: dict[str, object], task_type: str) -> dict[str, object]:
    rows = summary["by_task_type"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["task_type"] == task_type)


def _category_accuracy(summary: dict[str, object], category: str) -> float:
    rows = summary["by_task_category"]
    assert isinstance(rows, list)
    row = next(
        row
        for row in rows
        if row["task_type"] == "mcq" and row["task_category"] == category
    )
    return float(row["accuracy"])


def _headline_metrics(summary: dict[str, object]) -> dict[str, float]:
    captioning = summary["captioning"]
    assert isinstance(captioning, dict)
    return {
        "bleu4": float(captioning["bleu4"]),
        "binary_accuracy": float(_task_row(summary, "binary")["accuracy"]),
        "mcq_accuracy": float(_task_row(summary, "mcq")["accuracy"]),
        "bbox_miou": float(_task_row(summary, "bounding box")["miou"]),
    }


def build_core_headline_table() -> pd.DataFrame:
    rows = []
    for run in CORE_RUNS:
        if run.seed != 42:
            continue
        rows.append(
            {
                "model_size": run.model_size,
                "condition": run.condition,
                "evaluation_job": run.correct_job,
                **_headline_metrics(_summary(run.correct_job)),
            }
        )
    return pd.DataFrame(rows)


def build_seed_replication_table() -> pd.DataFrame:
    rows = []
    for run in CORE_RUNS:
        if run.model_size != "2B":
            continue
        rows.append(
            {
                "seed": run.seed,
                "condition": run.condition,
                "evaluation_job": run.correct_job,
                **_headline_metrics(_summary(run.correct_job)),
            }
        )
    return pd.DataFrame(rows)


def build_direct_geography_table() -> pd.DataFrame:
    rows = []
    for run in CORE_RUNS:
        if run.seed != 42:
            continue
        summary = _summary(run.correct_job)
        rows.append(
            {
                "model_size": run.model_size,
                "condition": run.condition,
                "country_accuracy": _category_accuracy(summary, "country"),
                "climate_zone_accuracy": _category_accuracy(summary, "climate zone"),
                "season_accuracy": _category_accuracy(summary, "season"),
            }
        )
    return pd.DataFrame(rows)


def build_development_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "configuration": label,
                "evaluation_job": job,
                **_headline_metrics(_summary(job)),
            }
            for label, job in DEVELOPMENT_RUNS
        ]
    )


def build_ablation_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "configuration": label,
                "budget": budget,
                "seed": seed,
                "evaluation_job": job,
                **_headline_metrics(_summary(job)),
            }
            for label, budget, seed, job in ABLATION_RUNS
        ]
    )


def build_budget_table() -> pd.DataFrame:
    development = {
        "no_loc": _headline_metrics(_summary("11401")),
        "loc_embed": _headline_metrics(_summary("11410")),
    }
    full = {
        "no_loc": _headline_metrics(_summary("11437")),
        "loc_embed": _headline_metrics(_summary("11438")),
    }
    rows = []
    for budget, metrics in (("1000 steps", development), ("full epoch", full)):
        rows.append(
            {
                "budget": budget,
                **{
                    metric: metrics["loc_embed"][metric] - metrics["no_loc"][metric]
                    for metric in METRIC_LABELS
                },
            }
        )
    return pd.DataFrame(rows)


def build_interval_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ANALYSIS_ROOT / "paired_cluster_bootstrap_intervals.csv"
    intervals = pd.read_csv(path)
    required = {
        "comparison",
        "kind",
        "metric",
        "difference_a_minus_b",
        "ci_low",
        "ci_high",
    }
    missing = required.difference(intervals.columns)
    if missing:
        raise ValueError(f"Missing interval columns in {path}: {sorted(missing)}")
    return (
        intervals[intervals["kind"] == "core"].copy(),
        intervals[intervals["kind"] == "counterfactual"].copy(),
    )


def _write_table(frame: pd.DataFrame, name: str) -> None:
    path = TABLES_ROOT / name
    frame.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


def plot_cross_size_effects(core_intervals: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    selected = core_intervals[
        core_intervals["metric"].isin(INTERVAL_METRIC_LABELS)
        & core_intervals["comparison"].str.contains("seed 42")
    ].copy()
    selected["model_size"] = selected["comparison"].str.extract(r"^(2B|4B|8B)")
    selected["condition"] = selected["comparison"].str.extract(r": (loc_text|loc_embed)")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
    sizes = ["2B", "4B", "8B"]
    colors = {"loc_text": "#D08770", "loc_embed": "#5E81AC"}
    offsets = {"loc_text": -0.07, "loc_embed": 0.07}
    for ax, (metric, label) in zip(
        axes.flat,
        INTERVAL_METRIC_LABELS.items(),
        strict=True,
    ):
        metric_rows = selected[selected["metric"] == metric]
        for condition in ("loc_text", "loc_embed"):
            rows = metric_rows[metric_rows["condition"] == condition].set_index("model_size")
            rows = rows.reindex(sizes)
            x = [index + offsets[condition] for index in range(len(sizes))]
            y = rows["difference_a_minus_b"].to_numpy()
            yerr = [
                y - rows["ci_low"].to_numpy(),
                rows["ci_high"].to_numpy() - y,
            ]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3,
                linewidth=1.5,
                color=colors[condition],
                label={"loc_text": "Coordinate text", "loc_embed": "Location tokens"}[
                    condition
                ],
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(label)
        ax.set_xticks(range(len(sizes)), sizes)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("Model size")
    fig.supylabel("Difference from no_loc")
    fig.tight_layout()
    path = FIGURES_ROOT / "core_effects_by_model_size.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


def plot_limited_budget_optimization() -> None:
    import matplotlib.pyplot as plt

    from notebooks.utils.extract_training_curves import extract_training_curves

    rows = extract_training_curves(
        runs_root=FINETUNING_ROOT,
        output=TABLES_ROOT / "limited_budget_optimization_curves.csv",
        jobs=set(LIMITED_BUDGET_RUNS),
        tags={"train/loss_step", "val/loss"},
    )
    curves = pd.DataFrame(rows)
    if curves.empty:
        raise ValueError("No repaired 1000-step optimization curves were found")

    colors = {"no_loc": "#4C566A", "loc_text": "#D08770", "loc_embed": "#5E81AC"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for job_id, condition in LIMITED_BUDGET_RUNS.items():
        run = curves[curves["job_id"].astype(str) == job_id]
        train = run[run["tag"] == "train/loss_step"].sort_values("step")
        validation = run[run["tag"] == "val/loss"].sort_values("step")

        axes[0].plot(train["step"], train["value"], color=colors[condition], alpha=0.18)
        axes[0].plot(
            train["step"],
            train["value"].rolling(5, min_periods=1).mean(),
            color=colors[condition],
            linewidth=1.6,
            label=CONDITION_LABELS[condition],
        )
        axes[1].plot(
            validation["step"],
            validation["value"],
            color=colors[condition],
            marker="o",
            markersize=3,
            linewidth=1.6,
        )

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = FIGURES_ROOT / "limited_budget_optimization_diagnostics.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    core_intervals, counterfactual_intervals = build_interval_tables()
    _write_table(build_development_table(), "development_selection.csv")
    _write_table(build_budget_table(), "training_budget_effects.csv")
    _write_table(build_core_headline_table(), "core_headline_seed42.csv")
    _write_table(build_seed_replication_table(), "core_2b_seed_replication.csv")
    _write_table(build_direct_geography_table(), "direct_geography_seed42.csv")
    _write_table(core_intervals, "core_paired_intervals.csv")
    _write_table(counterfactual_intervals, "counterfactual_paired_intervals.csv")
    _write_table(build_ablation_table(), "post_core_ablations.csv")
    plot_cross_size_effects(core_intervals)
    plot_limited_budget_optimization()


if __name__ == "__main__":
    main()
