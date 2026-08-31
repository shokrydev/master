"""Shared BigEarthNet.txt task-aware generation contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

SHORT_ANSWER_BUCKET = "short_answer"
BOUNDING_BOX_BUCKET = "bounding_box"
CAPTION_BUCKET = "captioning"
GENERATION_BUCKETS = (
    SHORT_ANSWER_BUCKET,
    BOUNDING_BOX_BUCKET,
    CAPTION_BUCKET,
)

DEFAULT_MAX_NEW_TOKENS_BY_BUCKET = {
    SHORT_ANSWER_BUCKET: 32,
    BOUNDING_BOX_BUCKET: 64,
    CAPTION_BUCKET: 512,
}

_TASK_TO_BUCKET = {
    "binary": SHORT_ANSWER_BUCKET,
    "mcq": SHORT_ANSWER_BUCKET,
    "bounding box": BOUNDING_BOX_BUCKET,
    "captioning": CAPTION_BUCKET,
}


def generation_bucket_for_task(task_type: str) -> str:
    """Map a BigEarthNet.txt task type to its generation-length bucket."""
    try:
        return _TASK_TO_BUCKET[str(task_type)]
    except KeyError as error:
        raise ValueError(f"Unsupported BigEarthNet.txt generation task: {task_type!r}") from error


def validate_bucket_values(
    values: Mapping[str, int] | None,
    *,
    label: str,
    minimum: int = 1,
) -> dict[str, int] | None:
    """Validate a complete integer value for every generation bucket."""
    if values is None:
        return None
    normalized = {str(key): int(value) for key, value in values.items()}
    missing = sorted(set(GENERATION_BUCKETS) - set(normalized))
    extra = sorted(set(normalized) - set(GENERATION_BUCKETS))
    if missing or extra:
        raise ValueError(
            f"{label} must contain exactly {list(GENERATION_BUCKETS)}; "
            f"missing={missing}, extra={extra}"
        )
    below_minimum = {key: value for key, value in normalized.items() if value < minimum}
    if below_minimum:
        raise ValueError(f"{label} values must be at least {minimum}: {below_minimum}")
    return normalized


def bucket_indices(task_types: Sequence[str]) -> dict[str, list[int]]:
    """Return stable dataset indices for each generation bucket."""
    indices = {bucket: [] for bucket in GENERATION_BUCKETS}
    for index, task_type in enumerate(task_types):
        indices[generation_bucket_for_task(task_type)].append(index)
    return indices
