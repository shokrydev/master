"""Strict generated-answer parsers for BigEarthNet.txt."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

BinaryAnswer = Literal["yes", "no"]
MCQAnswer = Literal["a", "b", "c", "d"]
BBox = tuple[float, float, float, float]

_FLOAT_PATTERN = r"(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)"
_BBOX_RE = re.compile(
    rf"^\[\s*({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s*,\s*"
    rf"({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s*\]$"
)


@dataclass(frozen=True)
class ParsedAnswer:
    """Parser output with explicit extraction success."""

    value: str | BBox | None
    extracted: bool


def first_answer_span(text: str) -> str:
    """Return the first non-empty generated line, stripped and lowercased."""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.lower()
    return ""


def parse_binary_answer(text: str) -> ParsedAnswer:
    """Parse a strict BigEarthNet.txt binary answer."""
    span = first_answer_span(text)
    if span in {"yes", "no"}:
        return ParsedAnswer(span, True)
    return ParsedAnswer(None, False)


def parse_mcq_answer(text: str) -> ParsedAnswer:
    """Parse a strict BigEarthNet.txt multiple-choice answer."""
    span = first_answer_span(text)
    if span in {"a", "b", "c", "d"}:
        return ParsedAnswer(span, True)
    return ParsedAnswer(None, False)


def parse_bbox_answer(text: str) -> ParsedAnswer:
    """Parse one normalized box in the official `[x1 y1, x2 y2]` style."""
    span = first_answer_span(text)
    match = _BBOX_RE.fullmatch(span)
    if not match:
        return ParsedAnswer(None, False)

    x1, y1, x2, y2 = (float(part) for part in match.groups())
    values = (x1, y1, x2, y2)
    if any(not math.isfinite(value) for value in values):
        return ParsedAnswer(None, False)
    if not (x1 <= x2 and y1 <= y2):
        return ParsedAnswer(None, False)
    return ParsedAnswer(values, True)


def bbox_iou(prediction: BBox, target: BBox) -> float:
    """Compute IoU for normalized `[x1, y1, x2, y2]` boxes."""
    px1, py1, px2, py2 = prediction
    tx1, ty1, tx2, ty2 = target

    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)

    intersection_width = max(0.0, ix2 - ix1)
    intersection_height = max(0.0, iy2 - iy1)
    intersection = intersection_width * intersection_height

    prediction_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = prediction_area + target_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
