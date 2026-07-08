"""Prediction-record loading for BigEarthNet.txt evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

TaskType = Literal["binary", "mcq", "captioning", "bounding box"]

REQUIRED_PREDICTION_FIELDS = {
    "prediction",
    "target_texts",
    "sample_id",
    "patch_id",
    "task_type",
    "task_category",
    "split",
}


@dataclass(frozen=True)
class BENTxTPrediction:
    """One generated prediction row exported by the Lightning test loop."""

    prediction: str
    target_texts: tuple[str, ...]
    sample_id: str
    patch_id: str
    task_type: TaskType
    task_category: str
    split: str
    input_text: str | None = None
    country: str | None = None
    season: str | None = None
    climate_zone: str | None = None
    lat: float | None = None
    lon: float | None = None
    location_condition: str | None = None
    model_size: str | None = None
    run_label: str | None = None
    adapter_dir: str | None = None
    model_name_or_path: str | None = None


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    return None if value is None else str(value)


def _optional_float(raw: dict[str, Any], key: str) -> float | None:
    value = raw.get(key)
    return None if value is None else float(value)


def prediction_from_json(raw: dict[str, Any], *, line_number: int | None = None) -> BENTxTPrediction:
    """Validate and normalize one exported prediction JSON object."""
    missing = REQUIRED_PREDICTION_FIELDS - raw.keys()
    if missing:
        prefix = f"line {line_number}: " if line_number is not None else ""
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{prefix}prediction record missing required fields: {missing_text}")

    task_type_text = str(raw["task_type"])
    if task_type_text not in {"binary", "mcq", "captioning", "bounding box"}:
        prefix = f"line {line_number}: " if line_number is not None else ""
        raise ValueError(f"{prefix}unsupported task_type: {task_type_text!r}")
    task_type = cast(TaskType, task_type_text)

    target_texts = raw["target_texts"]
    if isinstance(target_texts, str):
        target_texts_tuple = (target_texts,)
    elif isinstance(target_texts, (list, tuple)):
        target_texts_tuple = tuple(str(target) for target in target_texts)
    else:
        prefix = f"line {line_number}: " if line_number is not None else ""
        raise ValueError(f"{prefix}target_texts must be a string or list of strings")
    if not target_texts_tuple:
        prefix = f"line {line_number}: " if line_number is not None else ""
        raise ValueError(f"{prefix}target_texts must not be empty")

    return BENTxTPrediction(
        prediction=str(raw["prediction"]),
        target_texts=target_texts_tuple,
        sample_id=str(raw["sample_id"]),
        patch_id=str(raw["patch_id"]),
        task_type=task_type,
        task_category=str(raw["task_category"]),
        split=str(raw["split"]),
        input_text=_optional_str(raw, "input_text"),
        country=_optional_str(raw, "country"),
        season=_optional_str(raw, "season"),
        climate_zone=_optional_str(raw, "climate_zone"),
        lat=_optional_float(raw, "lat"),
        lon=_optional_float(raw, "lon"),
        location_condition=_optional_str(raw, "location_condition"),
        model_size=_optional_str(raw, "model_size"),
        run_label=_optional_str(raw, "run_label"),
        adapter_dir=_optional_str(raw, "adapter_dir"),
        model_name_or_path=_optional_str(raw, "model_name_or_path"),
    )


def load_predictions_jsonl(path: str | Path) -> list[BENTxTPrediction]:
    """Load a prediction JSONL file produced by the benchmark export job."""
    predictions: list[BENTxTPrediction] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            predictions.append(prediction_from_json(raw, line_number=line_number))
    return predictions
