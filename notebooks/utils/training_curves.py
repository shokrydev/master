"""Build a training-curve figure directly from synced finetuning runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .extract_training_curves import DEFAULT_OUTPUT as DEFAULT_CURVES_OUTPUT
from .extract_training_curves import DEFAULT_RUNS_ROOT, extract_training_curves
from .plot_training_curves import DEFAULT_OUTPUT as DEFAULT_FIGURE_OUTPUT
from .plot_training_curves import load_curves, plot_scalar


def build_training_curve_figure(
    *,
    jobs: set[str] | None = None,
    tag: str = "val/loss",
    output: Path = DEFAULT_FIGURE_OUTPUT,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    curves_output: Path = DEFAULT_CURVES_OUTPUT,
    title: str | None = None,
    ylabel: str | None = None,
    min_step: int | None = None,
    max_step: int | None = None,
) -> None:
    """Extract synced scalar events and write one selected scalar figure."""
    extract_training_curves(
        runs_root=runs_root,
        output=curves_output,
        jobs=jobs,
    )
    plot_scalar(
        df=load_curves(curves_output),
        tag=tag,
        output=output,
        jobs=jobs,
        title=title,
        ylabel=ylabel,
        min_step=min_step,
        max_step=max_step,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--curves-output", type=Path, default=DEFAULT_CURVES_OUTPUT)
    parser.add_argument("--tag", default="val/loss")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE_OUTPUT)
    parser.add_argument("--jobs", nargs="*", help="Optional finetuning job ids to include.")
    parser.add_argument("--title")
    parser.add_argument("--ylabel")
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_training_curve_figure(
        jobs=set(args.jobs) if args.jobs else None,
        tag=args.tag,
        output=args.output,
        runs_root=args.runs_root,
        curves_output=args.curves_output,
        title=args.title,
        ylabel=args.ylabel,
        min_step=args.min_step,
        max_step=args.max_step,
    )


if __name__ == "__main__":
    main()
