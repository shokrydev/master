"""BigEarthNet.txt/Qwen3-VL grounding-format conversion helpers."""

from __future__ import annotations

import json
import math
import re
from typing import Literal

GroundingFormat = Literal["bentxt", "qwen3_json"]
BBox = tuple[float, float, float, float]

QWEN_OBJECT_REF_TOKENS = (
    "<|object_ref_start|>",
    "<|object_ref_end|>",
)
QWEN_BOX_TOKENS = ("<|box_start|>", "<|box_end|>")
QWEN_COORDINATE_SCALE = 1000

_BEN_FLOAT_PATTERN = r"(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)"
_BEN_BBOX_RE = re.compile(
    rf"^\[\s*({_BEN_FLOAT_PATTERN})\s+({_BEN_FLOAT_PATTERN})\s*,\s*"
    rf"({_BEN_FLOAT_PATTERN})\s+({_BEN_FLOAT_PATTERN})\s*\]$"
)
_BEN_POINT_TAG_RE = re.compile(
    rf"<point>\s*\(\s*({_BEN_FLOAT_PATTERN})\s*,\s*"
    rf"({_BEN_FLOAT_PATTERN})\s*\)\s*</point>"
)


def _valid_bbox(values: tuple[float, float, float, float], *, upper: float) -> bool:
    x1, y1, x2, y2 = values
    return (
        all(math.isfinite(value) and 0.0 <= value <= upper for value in values)
        and x1 <= x2
        and y1 <= y2
    )


def parse_bentxt_bbox(text: str) -> BBox | None:
    """Parse BEN.txt's normalized ``[x1 y1, x2 y2]`` target format."""
    match = _BEN_BBOX_RE.fullmatch(str(text).strip().lower())
    if not match:
        return None
    values = tuple(float(part) for part in match.groups())
    if len(values) != 4 or not _valid_bbox(values, upper=1.0):
        return None
    return values


def _qwen_coordinate(value: float) -> int:
    return int(round(value * QWEN_COORDINATE_SCALE))


def bentxt_bbox_to_qwen3_json(text: str) -> str:
    """Convert one BEN.txt box to Qwen3-VL's 0-1000 JSON representation."""
    bbox = parse_bentxt_bbox(text)
    if bbox is None:
        raise ValueError(f"Invalid BigEarthNet.txt bounding-box target: {text!r}")
    payload = [{"bbox_2d": [_qwen_coordinate(value) for value in bbox]}]
    return json.dumps(payload, separators=(",", ":"))


def bentxt_bbox_to_qwen3_tokens(text: str) -> str:
    """Convert one BEN.txt box to Qwen's token-delimited 0-1000 format."""
    bbox = parse_bentxt_bbox(text)
    if bbox is None:
        raise ValueError(f"Invalid BigEarthNet.txt bounding-box target: {text!r}")
    x1, y1, x2, y2 = (_qwen_coordinate(value) for value in bbox)
    return (
        f"{QWEN_BOX_TOKENS[0]}({x1},{y1}),({x2},{y2})"
        f"{QWEN_BOX_TOKENS[1]}"
    )


def format_grounding_target(
    text: str,
    *,
    task_type: str,
    grounding_format: GroundingFormat,
) -> str:
    """Return the model-facing target while preserving non-grounding answers."""
    if task_type != "bounding box" or grounding_format == "bentxt":
        return str(text)
    if grounding_format == "qwen3_json":
        return bentxt_bbox_to_qwen3_json(text)
    raise ValueError(f"Unsupported grounding format: {grounding_format}")


def format_grounding_prompt(
    text: str,
    *,
    grounding_format: GroundingFormat,
    ref_token: tuple[str, str] | list[str],
    point_token: tuple[str, str] | list[str],
) -> str:
    """Adapt BEN.txt reference/point markup to the configured model contract."""
    formatted = str(text)
    if grounding_format == "qwen3_json":
        # Both datasets use image coordinates with a top-left origin. Qwen3-VL
        # changes only the scale: normalized BEN.txt values become integers on
        # a 1000 x 1000 reference grid.
        def replace_point(match: re.Match[str]) -> str:
            x = _qwen_coordinate(float(match.group(1)))
            y = _qwen_coordinate(float(match.group(2)))
            point = json.dumps({"point_2d": [x, y]}, separators=(",", ":"))
            return f"{point_token[0]}{point}{point_token[1]}"

        formatted = _BEN_POINT_TAG_RE.sub(replace_point, formatted)
    else:
        formatted = formatted.replace("<point>", point_token[0]).replace(
            "</point>", point_token[1]
        )

    formatted = formatted.replace("<ref>", ref_token[0]).replace(
        "</ref>", ref_token[1]
    )
    return formatted


def _strip_qwen_box_tokens(text: str) -> str:
    return str(text).replace(QWEN_BOX_TOKENS[0], "").replace(
        QWEN_BOX_TOKENS[1], ""
    )


def _json_bbox_candidate(value: object) -> object:
    if isinstance(value, dict):
        if "bbox_2d" in value:
            return value["bbox_2d"]
        if "bbox" in value:
            return value["bbox"]
    if isinstance(value, list):
        if len(value) == 1:
            return _json_bbox_candidate(value[0])
    return None


def parse_qwen3_bbox(text: str) -> BBox | None:
    """Parse Qwen JSON or token-delimited 0-1000 boxes into BEN coordinates."""
    stripped = _strip_qwen_box_tokens(text).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.lower().startswith("json\n"):
                stripped = stripped[5:].strip()
    candidate: object = None
    try:
        candidate = _json_bbox_candidate(json.loads(stripped))
    except json.JSONDecodeError:
        # Retained Qwen grounding tokens are also used with ``(x1,y1),(x2,y2)``.
        match = re.fullmatch(
            r"\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*,\s*"
            r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*",
            stripped,
        )
        if match:
            candidate = list(match.groups())

    if not isinstance(candidate, list) or len(candidate) != 4:
        return None
    try:
        qwen_bbox = tuple(float(value) for value in candidate)
    except (TypeError, ValueError):
        return None
    if len(qwen_bbox) != 4 or not _valid_bbox(
        qwen_bbox, upper=float(QWEN_COORDINATE_SCALE)
    ):
        return None
    if any(not value.is_integer() for value in qwen_bbox):
        return None
    return tuple(value / QWEN_COORDINATE_SCALE for value in qwen_bbox)
