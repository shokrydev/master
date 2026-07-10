"""Extract TensorBoard scalar curves from synced server runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_RUNS_ROOT = Path("outputs/finetuning")
DEFAULT_OUTPUT = Path("notebooks/analysis/training_curves.csv")


def _job_id_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.isdigit():
            return parent.name
        match = re.fullmatch(r"version_(\d+)", parent.name)
        if match:
            return match.group(1)
    return ""


def _run_label_from_logs(run_dir: Path, job_id: str) -> str:
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        return job_id
    for log_path in sorted(log_dir.glob(f"*_{job_id}.out")):
        return log_path.name.removesuffix(f"_{job_id}.out")
    return job_id


def _condition_from_label(label: str) -> str:
    for condition in ("no_loc", "loc_text", "loc_embed"):
        if label.startswith(condition):
            return condition
    return ""


def _model_size_from_label(label: str) -> str:
    match = re.search(r"(?:^|-)(2B|4B|8B)(?:-|$)", label)
    return match.group(1) if match else ""


def _event_files_for_run(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "lightning_logs").glob("version_*/events.out.tfevents.*"))


def discover_event_files(
    runs_root: Path,
    jobs: set[str] | None,
) -> list[tuple[Path, Path]]:
    event_files: list[tuple[Path, Path]] = []

    if runs_root.exists():
        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            if jobs is not None and run_dir.name not in jobs:
                continue
            event_files.extend((run_dir, event_path) for event_path in _event_files_for_run(run_dir))

    return event_files


def extract_event_file(
    run_dir: Path,
    event_path: Path,
    tags: set[str] | None,
) -> list[dict[str, str | int | float]]:
    job_id = _job_id_from_path(event_path)
    run_label = _run_label_from_logs(run_dir, job_id)
    condition = _condition_from_label(run_label)
    model_size = _model_size_from_label(run_label)

    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()

    rows: list[dict[str, str | int | float]] = []
    scalar_tags = sorted(accumulator.Tags().get("scalars", []))
    for tag in scalar_tags:
        if tags is not None and tag not in tags:
            continue
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "job_id": job_id,
                    "run_label": run_label,
                    "condition": condition,
                    "model_size": model_size,
                    "tag": tag,
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "value": event.value,
                    "event_file": str(event_path),
                }
            )
    return rows


def write_rows(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "run_label",
        "condition",
        "model_size",
        "tag",
        "step",
        "wall_time",
        "value",
        "event_file",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_training_curves(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    output: Path = DEFAULT_OUTPUT,
    jobs: set[str] | None = None,
    tags: set[str] | None = None,
) -> list[dict[str, str | int | float]]:
    """Extract scalar rows from synced server runs and write them to CSV."""
    event_files = discover_event_files(runs_root, jobs)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {runs_root}")

    rows: list[dict[str, str | int | float]] = []
    seen: set[tuple[str, str, int, float, str]] = set()
    for run_dir, event_path in event_files:
        for row in extract_event_file(run_dir, event_path, tags):
            key = (
                str(row["job_id"]),
                str(row["tag"]),
                int(row["step"]),
                float(row["value"]),
                str(row["event_file"]),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    rows.sort(key=lambda row: (str(row["job_id"]), str(row["tag"]), int(row["step"])))
    write_rows(rows, output)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", nargs="*", help="Optional job ids to include.")
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Optional scalar tags to include, for example train/loss val/loss.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = set(args.jobs) if args.jobs else None
    tags = set(args.tags) if args.tags else None

    rows = extract_training_curves(
        runs_root=args.runs_root,
        output=args.output,
        jobs=jobs,
        tags=tags,
    )
    print(f"Wrote {len(rows)} scalar rows to {args.output}")


if __name__ == "__main__":
    main()
