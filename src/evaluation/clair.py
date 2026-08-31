"""CLAIR prompt construction, response parsing, and aggregation.

This module contains no model runtime.  It keeps the published metric contract
testable and lets the GPU runner preserve model outputs before parsing them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.evaluation.bentxt_records import BENTxTPrediction

CLAIR_PROMPT = """\
You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.

Candidate set:
{candidate_statements}
Reference set:
{target_statements}
On a precise scale from 0 to 100, how likely is it that the candidate set is describing the same image as the reference set? (JSON format, with a key \"score\", value between 0 and 100, and a key \"reason\" with a string value.)
"""


@dataclass(frozen=True)
class CLAIRParseResult:
    score: float | None
    reason: str | None
    parse_method: str | None
    error: str | None


def caption_records(records: list[BENTxTPrediction]) -> list[BENTxTPrediction]:
    """Return only captioning rows, retaining their input order."""
    return [record for record in records if record.task_type == "captioning"]


def format_clair_prompt(candidate: str, references: tuple[str, ...]) -> str:
    """Render the prompt from the official CLAIR implementation."""
    candidate_statements = f"- {candidate}\n"
    target_statements = "".join(f"- {reference}\n" for reference in references)
    return CLAIR_PROMPT.format(
        candidate_statements=candidate_statements,
        target_statements=target_statements,
    )


def _first_score_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "score" in value:
            return value
    return None


def parse_clair_response(text: str) -> CLAIRParseResult:
    """Parse a CLAIR response while distinguishing JSON and numeric fallback."""
    parsed = _first_score_json_object(text)
    if parsed is not None:
        try:
            score = float(parsed["score"])
        except (TypeError, ValueError):
            return CLAIRParseResult(None, None, None, "JSON score is not numeric")
        if not 0.0 <= score <= 100.0:
            return CLAIRParseResult(None, None, None, "JSON score is outside [0, 100]")
        reason_value = parsed.get("reason")
        reason = None if reason_value is None else str(reason_value)
        return CLAIRParseResult(score, reason, "json", None)

    # This matches the official implementation's documented fallback, but does
    # not silently turn a missing score into zero.
    numeric = re.search(r"(?<![\w.])(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)(?!\w)", text)
    if numeric is None:
        return CLAIRParseResult(None, None, None, "no score could be parsed")
    score = float(numeric.group(1))
    reason_match = re.search(r"(?is)\breason\b\s*[:=-]?\s*(.+)", text)
    reason = reason_match.group(1).strip() if reason_match else None
    return CLAIRParseResult(score, reason, "numeric_fallback", None)


def summarize_clair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate sample-level outputs without hiding parse failures."""
    valid_scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    json_count = sum(row.get("parse_method") == "json" for row in rows)
    fallback_count = sum(row.get("parse_method") == "numeric_fallback" for row in rows)
    return {
        "num_caption_rows": len(rows),
        "num_scored": len(valid_scores),
        "num_parse_failures": len(rows) - len(valid_scores),
        "json_parse_count": json_count,
        "numeric_fallback_count": fallback_count,
        "parse_success_rate": len(valid_scores) / len(rows) if rows else None,
        "mean_clair_0_100": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "mean_clair_0_1": sum(valid_scores) / (100.0 * len(valid_scores)) if valid_scores else None,
    }
