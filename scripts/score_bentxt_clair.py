#!/usr/bin/env python3
"""Score BigEarthNet.txt captions with a local Unsloth CLAIR judge."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from src.evaluation.bentxt_records import BENTxTPrediction, load_predictions_jsonl
from src.evaluation.clair import (
    caption_records,
    format_clair_prompt,
    parse_clair_response,
    summarize_clair_rows,
)

DEFAULT_JUDGE = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="append",
        nargs=2,
        required=True,
        metavar=("PREDICTIONS", "OUTPUT_DIR"),
        help="Prediction export and its CLAIR output directory; may be repeated.",
    )
    parser.add_argument("--model-name-or-path", default=DEFAULT_JUDGE)
    parser.add_argument("--judge-label", default=DEFAULT_JUDGE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Pilot-only per-export limit.")
    return parser.parse_args()


def batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_judge_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_judge(model_name_or_path: str, max_sequence_length: int):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        device_map="auto",
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _generate_batch(
    records: Sequence[BENTxTPrediction],
    *,
    model: Any,
    tokenizer: Any,
    max_sequence_length: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    prompts = [format_clair_prompt(record.prediction, record.target_texts) for record in records]
    rendered = [
        tokenizer.apply_chat_template(
            build_judge_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sequence_length - max_new_tokens,
    )
    input_device = next(model.parameters()).device
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    generated = sequences[:, input_width:]
    responses = tokenizer.batch_decode(generated, skip_special_tokens=True)

    rows = []
    for index, (record, prompt, raw_response) in enumerate(
        zip(records, prompts, responses, strict=True)
    ):
        parsed = parse_clair_response(raw_response)
        prompt_tokens = int(inputs["attention_mask"][index].sum().item())
        completion_tokens = int(generated[index].ne(tokenizer.pad_token_id).sum().item())
        rows.append(
            {
                "sample_id": record.sample_id,
                "patch_id": record.patch_id,
                "candidate": record.prediction,
                "references": list(record.target_texts),
                "prompt": prompt,
                "raw_response": raw_response,
                "raw_reasoning_content": None,
                "score": parsed.score,
                "reason": parsed.reason,
                "parse_method": parsed.parse_method,
                "parse_error": parsed.error,
                "finish_reason": None,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "timings": {"batch_seconds": elapsed, "batch_size": len(records)},
            }
        )
    return rows


def _score_export(
    predictions: Path,
    output_dir: Path,
    *,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    provenance: dict[str, Any],
) -> int:
    records = caption_records(load_predictions_jsonl(predictions))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"no captioning rows found in {predictions}")

    output_rows: list[dict[str, Any]] = []
    for batch_index, record_batch in enumerate(batches(records, args.batch_size), start=1):
        output_rows.extend(
            _generate_batch(
                record_batch,
                model=model,
                tokenizer=tokenizer,
                max_sequence_length=args.max_sequence_length,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print(
            f"{predictions}: scored {len(output_rows)}/{len(records)} captions "
            f"(batch {batch_index})",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "clair_sample_scores.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = provenance | {
        "predictions": str(predictions.resolve()),
        "batch_size": args.batch_size,
        "max_sequence_length": args.max_sequence_length,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "decoding": {"do_sample": False, "enable_thinking": False},
    }
    _write_json(output_dir / "clair_config.json", config)
    _write_json(output_dir / "clair_summary.json", summarize_clair_rows(output_rows) | config)
    print(f"Wrote {len(output_rows)} CLAIR scores to {output_dir}", flush=True)
    return len(output_rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_sequence_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch size and token limits must be positive")
    if args.max_new_tokens >= args.max_sequence_length:
        raise ValueError("max new tokens must be smaller than max sequence length")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    score_specs = [(Path(predictions), Path(output)) for predictions, output in args.score]
    for predictions, _ in score_specs:
        if not predictions.is_file():
            raise ValueError(f"predictions do not exist: {predictions}")

    print(f"Loading CLAIR judge once for {len(score_specs)} prediction export(s)", flush=True)
    model, tokenizer = _load_judge(args.model_name_or_path, args.max_sequence_length)
    model_config = getattr(model, "config", None)
    provenance = {
        "backend": "transformers-bitsandbytes",
        "judge_label": args.judge_label,
        "model_name_or_path": args.model_name_or_path,
        "model_commit_hash": getattr(model_config, "_commit_hash", None),
        "quantization": "bitsandbytes NF4",
        "package_versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "unsloth": _package_version("unsloth"),
            "unsloth_zoo": _package_version("unsloth-zoo"),
            "bitsandbytes": _package_version("bitsandbytes"),
        },
    }
    for predictions, output_dir in score_specs:
        _score_export(
            predictions,
            output_dir,
            model=model,
            tokenizer=tokenizer,
            args=args,
            provenance=provenance,
        )


if __name__ == "__main__":
    main()
