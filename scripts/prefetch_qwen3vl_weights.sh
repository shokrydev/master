#!/usr/bin/env bash
# Prefetch exact Unsloth Qwen3-VL 4-bit model repos into the Hugging Face cache.
#
# Run this once before the first finetuning job to avoid slow downloads on the
# compute node.
#
# Usage:
#   bash scripts/prefetch_qwen3vl_weights.sh
#   bash scripts/prefetch_qwen3vl_weights.sh 2B 4B 8B

set -e

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

SIZES=("${@:-2B}")

declare -A MODEL_REPO
MODEL_REPO[2B]="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
MODEL_REPO[4B]="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
MODEL_REPO[8B]="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"

for size in "${SIZES[@]}"; do
    if [ -z "${MODEL_REPO[$size]:-}" ]; then
        echo "Invalid size '$size'. Use 2B, 4B or 8B."
        exit 1
    fi
    echo "=== Downloading ${size} ==="
    uv run hf download "${MODEL_REPO[$size]}"
done

echo "Done. All requested Qwen3-VL weights are cached under $HF_HOME."
