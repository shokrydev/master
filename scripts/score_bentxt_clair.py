#!/usr/bin/env python3
"""Score BigEarthNet.txt captions through a local llama.cpp CLAIR judge."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from src.evaluation.bentxt_records import BENTxTPrediction, load_predictions_jsonl
from src.evaluation.clair import (
    caption_records,
    format_clair_prompt,
    parse_clair_response,
    summarize_clair_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--judge-label", default="unsloth/Qwen3.8-27B-GGUF:UD-Q6_K")
    parser.add_argument("--llama-version", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=None, help="Pilot-only caption-row limit.")
    return parser.parse_args()


def build_request_payload(
    prompt: str,
    *,
    max_new_tokens: int,
    judge_label: str,
) -> dict[str, Any]:
    """Build deterministic CLAIR chat-completion parameters."""
    return {
        "model": judge_label,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": max_new_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
    }


def _score_one(
    record: BENTxTPrediction,
    *,
    base_url: str,
    max_new_tokens: int,
    judge_label: str,
    timeout: float,
) -> dict[str, Any]:
    prompt = format_clair_prompt(record.prediction, record.target_texts)
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=build_request_payload(
            prompt,
            max_new_tokens=max_new_tokens,
            judge_label=judge_label,
        ),
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    message = choice["message"]
    raw_response = message.get("content") or ""
    parsed = parse_clair_response(raw_response)
    return {
        "sample_id": record.sample_id,
        "patch_id": record.patch_id,
        "candidate": record.prediction,
        "references": list(record.target_texts),
        "prompt": prompt,
        "raw_response": raw_response,
        "raw_reasoning_content": message.get("reasoning_content"),
        "score": parsed.score,
        "reason": parsed.reason,
        "parse_method": parsed.parse_method,
        "parse_error": parsed.error,
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "timings": body.get("timings"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0 or args.max_new_tokens <= 0 or args.request_timeout <= 0:
        raise ValueError("concurrency, max new tokens, and timeout must be positive")
    if not args.model_path.is_file():
        raise ValueError(f"GGUF model does not exist: {args.model_path}")

    records = caption_records(load_predictions_jsonl(args.predictions))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        records = records[: args.limit]
    if not records:
        raise ValueError(f"no captioning rows found in {args.predictions}")

    def score(record: BENTxTPrediction) -> dict[str, Any]:
        return _score_one(
            record,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            judge_label=args.judge_label,
            timeout=args.request_timeout,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        output_rows = list(executor.map(score, records))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "clair_sample_scores.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "predictions": str(args.predictions.resolve()),
        "judge_label": args.judge_label,
        "gguf_model_path": str(args.model_path.resolve()),
        "gguf_size_bytes": args.model_path.stat().st_size,
        "llama_version": args.llama_version,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "request_timeout": args.request_timeout,
        "limit": args.limit,
        "quantization": "UD-Q6_K",
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "enable_thinking": False,
        },
    }
    _write_json(args.output_dir / "clair_config.json", config)
    _write_json(args.output_dir / "clair_summary.json", summarize_clair_rows(output_rows) | config)
    print(f"Scored {len(output_rows)} caption rows")
    print(f"Wrote CLAIR outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
