"""BigEarthNet.txt prediction scoring."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score

from src.evaluation.bentxt_parsing import (
    bbox_iou,
    parse_bbox_answer,
    parse_binary_answer,
    parse_mcq_answer,
)
from src.evaluation.bentxt_records import BENTxTPrediction

IOU_THRESHOLDS = (0.25, 0.50, 0.75, 0.90)
STRATIFICATION_FIELDS = (
    "task_type",
    "task_category",
    "split",
    "country",
    "season",
    "climate_zone",
    "location_condition",
    "model_size",
    "run_label",
)


@dataclass(frozen=True)
class SampleScore:
    """Per-row parsed score used for stratified BEN.txt summaries."""

    sample_id: str
    patch_id: str
    task_type: str
    task_category: str
    split: str
    country: str | None
    season: str | None
    climate_zone: str | None
    location_condition: str | None
    model_size: str | None
    run_label: str | None
    extracted: bool
    correct: bool | None = None
    iou: float | None = None


def _ensure_meteor_resources() -> None:
    try:
        wordnet.ensure_loaded()
    except LookupError as exc:
        raise RuntimeError(
            "METEOR requires NLTK WordNet data. Install it once with: "
            "uv run python -m nltk.downloader wordnet omw-1.4"
        ) from exc


def _compute_rouge(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    if not predictions:
        return totals

    for prediction, refs in zip(predictions, references, strict=True):
        best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for reference in refs:
            scores = scorer.score(reference, prediction)
            for key in best:
                best[key] = max(best[key], scores[key].fmeasure)
        totals["rouge_1"] += best["rouge1"]
        totals["rouge_2"] += best["rouge2"]
        totals["rouge_l"] += best["rougeL"]

    return {key: value / len(predictions) for key, value in totals.items()}


def compute_caption_metrics(records: Sequence[BENTxTPrediction]) -> dict[str, float]:
    """Compute corpus-level caption metrics for BEN.txt captioning rows."""
    caption_records = [record for record in records if record.task_type == "captioning"]
    metric_names = [
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "meteor",
        "cider",
    ]
    if not caption_records:
        return dict.fromkeys(metric_names, 0.0)

    _ensure_meteor_resources()

    predictions = [record.prediction for record in caption_records]
    references = [list(record.target_texts) for record in caption_records]

    gts: dict[int, list[str]] = {}
    res: dict[int, list[str]] = {}
    for index, (prediction, refs) in enumerate(zip(predictions, references, strict=True)):
        res[index] = [prediction]
        gts[index] = [str(reference) for reference in refs]

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    scores: dict[str, float] = {}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    for name, value in zip(["bleu1", "bleu2", "bleu3", "bleu4"], bleu_scores, strict=True):
        scores[name] = float(value)

    meteor_total = 0.0
    for prediction, refs in zip(predictions, references, strict=True):
        meteor_total += meteor_score(
            [reference.split() for reference in refs],
            prediction.split(),
        )
    scores["meteor"] = meteor_total / len(predictions)

    cider_score, _ = Cider().compute_score(gts, res)
    scores["cider"] = float(cider_score)
    scores.update(_compute_rouge(predictions, references))
    return scores


def score_prediction(record: BENTxTPrediction) -> SampleScore:
    """Parse and score one exported BEN.txt prediction row."""
    base = {
        "sample_id": record.sample_id,
        "patch_id": record.patch_id,
        "task_type": record.task_type,
        "task_category": record.task_category,
        "split": record.split,
        "country": record.country,
        "season": record.season,
        "climate_zone": record.climate_zone,
        "location_condition": record.location_condition,
        "model_size": record.model_size,
        "run_label": record.run_label,
    }

    target = record.target_texts[0].strip().lower()
    if record.task_type == "captioning":
        return SampleScore(**base, extracted=True)

    if record.task_type == "binary":
        parsed = parse_binary_answer(record.prediction)
        return SampleScore(
            **base,
            extracted=parsed.extracted,
            correct=bool(parsed.extracted and parsed.value == target),
        )

    if record.task_type == "mcq":
        parsed = parse_mcq_answer(record.prediction)
        return SampleScore(
            **base,
            extracted=parsed.extracted,
            correct=bool(parsed.extracted and parsed.value == target),
        )

    if record.task_type == "bounding box":
        parsed_prediction = parse_bbox_answer(record.prediction)
        parsed_target = parse_bbox_answer(target)
        if not parsed_prediction.extracted or not parsed_target.extracted:
            return SampleScore(**base, extracted=False, iou=0.0)
        if not isinstance(parsed_prediction.value, tuple) or not isinstance(
            parsed_target.value, tuple
        ):
            return SampleScore(**base, extracted=False, iou=0.0)
        iou = bbox_iou(parsed_prediction.value, parsed_target.value)
        return SampleScore(**base, extracted=True, iou=iou)

    raise ValueError(f"Unsupported task_type: {record.task_type}")


def score_predictions(records: Iterable[BENTxTPrediction]) -> list[SampleScore]:
    """Parse and score exported BEN.txt prediction rows."""
    return [score_prediction(record) for record in records]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_group(scores: Sequence[SampleScore]) -> dict[str, float | int]:
    total = len(scores)
    extracted = sum(score.extracted for score in scores)
    summary: dict[str, float | int] = {
        "n": total,
        "extraction_success": extracted / total if total else 0.0,
    }

    task_types = {score.task_type for score in scores}
    if task_types <= {"binary", "mcq"}:
        correct = sum(bool(score.correct) for score in scores)
        summary["accuracy"] = correct / total if total else 0.0
        summary["correct"] = correct
    elif task_types == {"bounding box"}:
        ious = [float(score.iou or 0.0) for score in scores]
        summary["miou"] = _mean(ious)
        for threshold in IOU_THRESHOLDS:
            summary[f"acc@{int(threshold * 100)}"] = (
                sum(iou >= threshold for iou in ious) / total if total else 0.0
            )
    elif task_types == {"captioning"}:
        summary["extraction_success"] = 1.0 if total else 0.0

    return summary


def summarize_scores(
    scores: Sequence[SampleScore],
    *,
    group_by: Sequence[str] = ("task_type",),
) -> list[dict[str, Any]]:
    """Summarize sample scores, optionally stratified by metadata fields."""
    unknown = [field for field in group_by if field not in STRATIFICATION_FIELDS]
    if unknown:
        raise ValueError(f"Unsupported stratification field(s): {unknown}")
    if "task_type" not in group_by:
        raise ValueError("BEN.txt score summaries must include 'task_type'")

    grouped: dict[tuple[str, ...], list[SampleScore]] = defaultdict(list)
    for score in scores:
        key = tuple(str(getattr(score, field) or "<missing>") for field in group_by)
        grouped[key].append(score)

    rows: list[dict[str, Any]] = []
    for key, group_scores in sorted(grouped.items()):
        row = dict(zip(group_by, key, strict=True))
        row.update(_summarize_group(group_scores))
        rows.append(row)
    return rows


def evaluate_predictions(records: Sequence[BENTxTPrediction]) -> dict[str, Any]:
    """Return the standard BEN.txt offline evaluation summary."""
    scores = score_predictions(records)
    return {
        "captioning": compute_caption_metrics(records),
        "by_task_type": summarize_scores(scores, group_by=("task_type",)),
        "by_task_category": summarize_scores(scores, group_by=("task_type", "task_category")),
        "by_country": summarize_scores(scores, group_by=("task_type", "country")),
        "by_season": summarize_scores(scores, group_by=("task_type", "season")),
        "by_climate_zone": summarize_scores(scores, group_by=("task_type", "climate_zone")),
    }


def scores_as_dicts(scores: Iterable[SampleScore]) -> list[dict[str, Any]]:
    """Serialize per-sample scores for JSONL/CSV output."""
    return [asdict(score) for score in scores]
