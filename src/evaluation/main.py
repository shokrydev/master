"""Offline evaluation commands for prediction JSONL exports.

Model inference is run through the repository root `main.py test` entrypoint.
This module parses and scores the exported predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.evaluation.bentxt_records import load_predictions_jsonl
from src.evaluation.bentxt_scoring import (
    evaluate_predictions,
    score_predictions,
    scores_as_dicts,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _score_predictions_command(args: argparse.Namespace) -> None:
    records = load_predictions_jsonl(args.predictions)
    scores = score_predictions(records)
    summary = evaluate_predictions(records)

    output_dir = args.output_dir
    _write_jsonl(output_dir / "sample_scores.jsonl", scores_as_dicts(scores))
    _write_json(output_dir / "summary.json", summary)

    for key, rows in summary.items():
        if key == "captioning":
            continue
        _write_csv(output_dir / f"{key}.csv", rows)

    print(f"Scored {len(records)} predictions")
    print(f"Wrote scores to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.main",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser(
        "score",
        help="Score a BigEarthNet.txt prediction JSONL export.",
    )
    score_parser.add_argument(
        "predictions",
        type=Path,
        help="Prediction JSONL exported by the BigEarthNet.txt test job.",
    )
    score_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for sample_scores.jsonl, summary.json, and summary CSV tables.",
    )
    score_parser.set_defaults(func=_score_predictions_command)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
