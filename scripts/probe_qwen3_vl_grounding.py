#!/usr/bin/env python3
"""Probe Qwen3-VL grounding formats and the real Unsloth supervision mask."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from src.bentxt_grounding import (
    QWEN_OBJECT_REF_TOKENS,
    bentxt_bbox_to_qwen3_json,
    format_grounding_prompt,
    parse_qwen3_bbox,
)
from src.data_modules.ben_txt_datamodule import BENTxTDataset
from src.data_modules.geo_aware_collator import GeoAwareCollator
from src.evaluation.bentxt_parsing import bbox_iou

SYSTEM_PROMPT = "You are a remote sensing image analysis assistant."
BOX_TOKEN_OUTPUT_INSTRUCTION = (
    "Return only one box in the form "
    "<|box_start|>(x1,y1),(x2,y2)<|box_end|> using integer coordinates "
    "from 0 to 1000."
)
JSON_OUTPUT_INSTRUCTION = (
    'Return only a JSON list in the form [{"bbox_2d":[x1,y1,x2,y2]}] '
    "using integer coordinates from 0 to 1000."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--image-lmdb-file", required=True)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    for index in range(len(sequence) - len(subsequence) + 1):
        if sequence[index : index + len(subsequence)] == subsequence:
            return index
    return None


def stop_token_ids(model: Any, processor: Any) -> set[int]:
    ids: set[int] = set()
    generation_config = getattr(model, "generation_config", None)
    eos_ids = getattr(generation_config, "eos_token_id", None)
    if isinstance(eos_ids, int):
        ids.add(eos_ids)
    elif eos_ids is not None:
        ids.update(int(token_id) for token_id in eos_ids)
    tokenizer = getattr(processor, "tokenizer", processor)
    for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if token_id is not None:
            ids.add(int(token_id))
    return ids


def decode_response(
    generated_ids: torch.Tensor,
    *,
    input_length: int,
    model: Any,
    processor: Any,
) -> str:
    ids = generated_ids[input_length:].tolist()
    terminal_ids = stop_token_ids(model, processor)
    while ids and ids[-1] in terminal_ids:
        ids.pop()
    return processor.decode(ids, skip_special_tokens=False)


def select_examples(dataset: BENTxTDataset, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("num_examples must be positive")
    frame = dataset.text_data
    point_rows = frame["input"].str.contains("<point>", regex=False)
    ref_rows = frame["input"].str.contains("<ref>", regex=False)
    groups = [frame.index[point_rows], frame.index[ref_rows]]
    selected: list[int] = []
    per_group = max(count // 2, 1)
    for group_index, indices in enumerate(groups):
        if len(indices) == 0:
            continue
        sampled = indices.to_series().sample(
            n=min(per_group, len(indices)),
            random_state=42 + group_index,
        )
        selected.extend(int(index) for index in sampled.tolist())
    if len(selected) < count:
        remaining = frame.index.difference(selected)
        sampled = remaining.to_series().sample(
            n=min(count - len(selected), len(remaining)),
            random_state=44,
        )
        selected.extend(int(index) for index in sampled.tolist())
    return selected[:count]


def prompt_variants(raw_prompt: str) -> list[tuple[str, str, str]]:
    official = format_grounding_prompt(
        raw_prompt,
        grounding_format="qwen3_json",
        ref_token=QWEN_OBJECT_REF_TOKENS,
        point_token=("", ""),
    )
    plain = format_grounding_prompt(
        raw_prompt,
        grounding_format="qwen3_json",
        ref_token=("", ""),
        point_token=("", ""),
    )
    schema_instructed = f"{official.rstrip()}\n{JSON_OUTPUT_INSTRUCTION}"
    token_output = f"{official.rstrip()}\n{BOX_TOKEN_OUTPUT_INSTRUCTION}"
    return [
        ("qwen_json_image_first", official, "image_then_text"),
        ("qwen_json_text_first", official, "text_then_image"),
        ("qwen_json_plain_reference", plain, "image_then_text"),
        ("qwen_json_schema_instruction", schema_instructed, "image_then_text"),
        ("qwen_box_tokens", token_output, "image_then_text"),
        ("bentxt_original", raw_prompt, "image_then_text"),
    ]


def build_messages(image: Any, prompt: str, content_order: str) -> list[dict[str, Any]]:
    image_content = {"type": "image", "image": image}
    text_content = {"type": "text", "text": prompt}
    if content_order == "image_then_text":
        content = [image_content, text_content]
    elif content_order == "text_then_image":
        content = [text_content, image_content]
    else:
        raise ValueError(f"Unsupported content order: {content_order}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def validate_supervision_mask(
    *,
    collator: UnslothVisionDataCollator,
    processor: Any,
    sample: dict[str, Any],
) -> dict[str, Any]:
    wrapped = GeoAwareCollator(collator, system_prompt=SYSTEM_PROMPT)
    batch = wrapped(
        [
            {
                "image": sample["image"],
                "input_text": sample["input_text"],
                "target_texts": sample["target_texts"],
                "model_target_texts": sample["model_target_texts"],
                "lat": sample["lat"],
                "lon": sample["lon"],
            }
        ]
    )
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    tokenizer = getattr(processor, "tokenizer", processor)
    marker_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    marker_index = find_subsequence(input_ids.tolist(), marker_ids)
    if marker_index is None:
        raise RuntimeError("Assistant marker was not found after real Unsloth collation")
    response_start = marker_index + len(marker_ids)
    if labels[:response_start].ne(-100).any():
        raise RuntimeError("System/user/vision prompt positions unexpectedly receive loss")
    if not labels[response_start:].ne(-100).any():
        raise RuntimeError("Assistant answer has no supervised tokens")

    vision_token_strings = (
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    )
    vision_token_ids = {
        int(tokenizer.convert_tokens_to_ids(token)) for token in vision_token_strings
    }
    vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in vision_token_ids:
        vision_mask |= input_ids.eq(token_id)
    if labels[vision_mask].ne(-100).any():
        raise RuntimeError("Native Qwen vision tokens unexpectedly receive loss")

    supervised_ids = labels[labels.ne(-100)].tolist()
    return {
        "assistant_marker_index": marker_index,
        "ignored_tokens": int(labels.eq(-100).sum()),
        "supervised_tokens": len(supervised_ids),
        "supervised_text": processor.decode(
            supervised_ids,
            skip_special_tokens=False,
        ),
        "vision_tokens": int(vision_mask.sum()),
    }


def model_identity(model_source: str, model: Any, processor: Any) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    special_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        *QWEN_OBJECT_REF_TOKENS,
        "<|box_start|>",
        "<|box_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    return {
        "model_source": model_source,
        "model_name_or_path": getattr(model.config, "name_or_path", None),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "processor_name_or_path": getattr(processor, "name_or_path", None),
        "processor_commit_hash": getattr(processor, "_commit_hash", None),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "special_token_ids": {
            token: tokenizer.convert_tokens_to_ids(token) for token in special_tokens
        },
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "unsloth", "unsloth_zoo")
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_source = args.adapter_dir or args.model
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_source,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    model_device = next(model.parameters()).device
    collator = UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    generation_wrapper = GeoAwareCollator(
        collator,
        system_prompt=SYSTEM_PROMPT,
        generation_prompt=True,
    )
    dataset = BENTxTDataset(
        args.image_lmdb_file,
        args.metadata_file,
        bands="RGB",
        types=("bounding box",),
        splits=("validation",),
        grounding_format="qwen3_json",
        ref_token=QWEN_OBJECT_REF_TOKENS,
        point_token=("", ""),
    )
    selected_indices = select_examples(dataset, args.num_examples)
    first_sample = dataset[selected_indices[0]]
    supervision = validate_supervision_mask(
        collator=collator,
        processor=processor,
        sample=first_sample,
    )

    records: list[dict[str, Any]] = []
    for index in selected_indices:
        row = dataset.text_data.iloc[index]
        sample = dataset[index]
        target_bbox = parse_qwen3_bbox(bentxt_bbox_to_qwen3_json(str(row.output)))
        if target_bbox is None:
            raise RuntimeError(f"Could not parse target box for row {row.ID}")
        for variant, prompt, content_order in prompt_variants(str(row.input)):
            messages = build_messages(sample["image"], prompt, content_order)
            batch = generation_wrapper._collate_generation_prompts(
                [{"messages": messages}]
            )
            batch = {key: value.to(model_device) for key, value in batch.items()}
            with torch.no_grad():
                generated_ids = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            prediction = decode_response(
                generated_ids[0],
                input_length=batch["input_ids"].shape[-1],
                model=model,
                processor=processor,
            )
            predicted_bbox = parse_qwen3_bbox(prediction)
            records.append(
                {
                    "sample_id": str(row.ID),
                    "patch_id": str(row.patch_id),
                    "task_category": str(row.category),
                    "variant": variant,
                    "content_order": content_order,
                    "prompt": prompt,
                    "prediction": prediction,
                    "target_text": str(row.output),
                    "extracted": predicted_bbox is not None,
                    "iou": (
                        bbox_iou(predicted_bbox, target_bbox)
                        if predicted_bbox is not None
                        else 0.0
                    ),
                }
            )

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["variant"]].append(record)
    variant_summary = {
        variant: {
            "n": len(group),
            "extraction_success": sum(row["extracted"] for row in group) / len(group),
            "mean_iou": sum(float(row["iou"]) for row in group) / len(group),
        }
        for variant, group in sorted(grouped.items())
    }
    summary = {
        "model": model_identity(model_source, model, processor),
        "assistant_only_supervision": supervision,
        "num_examples": len(selected_indices),
        "variants": variant_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
