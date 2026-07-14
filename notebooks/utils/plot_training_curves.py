"""Plot scalar training curves extracted from TensorBoard events."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CURVES = Path("notebooks/analysis/training_curves.csv")
DEFAULT_OUTPUT = Path("notebooks/analysis/figures/training_curve.png")

CONDITION_ORDER = {"no_loc": 0, "loc_text": 1, "loc_embed": 2}
CONDITION_COLORS = {
    "no_loc": "#4C566A",
    "loc_text": "#D08770",
    "loc_embed": "#5E81AC",
}


def load_curves(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"job_id", "run_label", "condition", "tag", "step", "value"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {path}: {missing_text}")
    return df


def _sorted_groups(df: pd.DataFrame) -> list[tuple[tuple[str, str, str], pd.DataFrame]]:
    groups = list(df.groupby(["condition", "job_id", "run_label"], dropna=False))
    return sorted(
        groups,
        key=lambda item: (
            CONDITION_ORDER.get(str(item[0][0]), 99),
            str(item[0][2]),
            str(item[0][1]),
        ),
    )


def plot_scalar(
    df: pd.DataFrame,
    tag: str,
    output: Path,
    jobs: set[str] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    max_step: int | None = None,
    min_step: int | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting training curves.") from exc

    plot_df = df[df["tag"] == tag].copy()
    if jobs is not None:
        plot_df = plot_df[plot_df["job_id"].astype(str).isin(jobs)]
    if max_step is not None:
        plot_df = plot_df[plot_df["step"] <= max_step]
    if min_step is not None:
        plot_df = plot_df[plot_df["step"] >= min_step]
    if plot_df.empty:
        raise ValueError(f"No rows found for tag '{tag}'.")

    plot_df = plot_df.sort_values(["job_id", "tag", "step", "wall_time"])
    plot_df = plot_df.drop_duplicates(["job_id", "tag", "step"], keep="last")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (condition, job_id, run_label), group in _sorted_groups(plot_df):
        label = str(condition) if condition else (run_label or str(job_id))
        color = CONDITION_COLORS.get(str(condition), None)
        ax.plot(group["step"], group["value"], marker="o", markersize=3, linewidth=1.8, label=label, color=color)

    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(ylabel or tag)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--tag", default="val/loss")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", nargs="*", help="Optional job ids to include.")
    parser.add_argument("--title")
    parser.add_argument("--ylabel")
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = set(args.jobs) if args.jobs else None
    df = load_curves(args.curves)
    plot_scalar(
        df=df,
        tag=args.tag,
        output=args.output,
        jobs=jobs,
        title=args.title,
        ylabel=args.ylabel,
        min_step=args.min_step,
        max_step=args.max_step,
    )


if __name__ == "__main__":
    main()
