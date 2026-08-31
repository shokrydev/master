#!/bin/bash
# Submit only the prespecified evaluations for the already-running 2B fits
# 11807--11814. The all-in-one fit+evaluation helper remains unchanged.

set -euo pipefail

SHORT_BATCH=""
BBOX_BATCH=""
CAPTION_BATCH=""
SHORT_WORKERS=""
BBOX_WORKERS=""
CAPTION_WORKERS=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --short-batch) SHORT_BATCH="${2:-}"; shift 2 ;;
        --bbox-batch) BBOX_BATCH="${2:-}"; shift 2 ;;
        --caption-batch) CAPTION_BATCH="${2:-}"; shift 2 ;;
        --short-workers) SHORT_WORKERS="${2:-}"; shift 2 ;;
        --bbox-workers) BBOX_WORKERS="${2:-}"; shift 2 ;;
        --caption-workers) CAPTION_WORKERS="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

values=("$SHORT_BATCH" "$BBOX_BATCH" "$CAPTION_BATCH" "$SHORT_WORKERS" "$BBOX_WORKERS" "$CAPTION_WORKERS")
for value in "${values[@]}"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "All three batch sizes and worker counts must be positive integers from profiler 11836."
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
if [ ! -f .env ]; then
    echo "Missing .env."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
if [ -z "${FINETUNING_OUTPUT_ROOT:-}" ]; then
    echo "FINETUNING_OUTPUT_ROOT is required."
    exit 1
fi

CONDITIONS=(no_loc loc_text loc_embed loc_additive_satclip)
SEEDS=(42 43)
CORRECT_STEPS=(50 100 500 1000 5000 final)
SHUFFLED_STEPS=(1000 final)
declare -A FIT_IDS=(
    [42:no_loc]=11807
    [42:loc_text]=11808
    [42:loc_embed]=11809
    [42:loc_additive_satclip]=11810
    [43:no_loc]=11811
    [43:loc_text]=11812
    [43:loc_embed]=11813
    [43:loc_additive_satclip]=11814
)

adapter_dir_for_step() {
    local fit_id="$1"
    local step="$2"
    local root="${FINETUNING_OUTPUT_ROOT%/}/bigearthnet_${fit_id}"
    if [ "$step" = final ]; then
        printf '%s/qlora_adapter' "$root"
    else
        printf '%s/qlora_adapter_steps/step_%06d' "$root" "$step"
    fi
}

COMMON_OVERRIDES=(
    --data.init_args.evaluation_batch_sizes.short_answer "$SHORT_BATCH"
    --data.init_args.evaluation_batch_sizes.bounding_box "$BBOX_BATCH"
    --data.init_args.evaluation_batch_sizes.captioning "$CAPTION_BATCH"
    --data.init_args.evaluation_num_workers_by_bucket.short_answer "$SHORT_WORKERS"
    --data.init_args.evaluation_num_workers_by_bucket.bounding_box "$BBOX_WORKERS"
    --data.init_args.evaluation_num_workers_by_bucket.captioning "$CAPTION_WORKERS"
)
if [ "$DRY_RUN" = true ]; then
    COMMON_OVERRIDES+=(--dry-run)
fi

for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        fit_id="${FIT_IDS["${seed}:${condition}"]}"
        prefix="qwen3-json-${condition}-2B-full-seed${seed}"
        for step in "${CORRECT_STEPS[@]}"; do
            "$SCRIPT_DIR/submit_evaluation_job.sh" \
                --condition "$condition" --size 2B \
                --adapter-dir "$(adapter_dir_for_step "$fit_id" "$step")" \
                --name "eval-${fit_id}-${condition}-${step}" \
                --run-label "${prefix}-step${step}-j${fit_id}" \
                --dependency "afterok:${fit_id}" \
                "${COMMON_OVERRIDES[@]}"
        done
        if [ "$condition" = no_loc ]; then
            continue
        fi
        for step in "${SHUFFLED_STEPS[@]}"; do
            "$SCRIPT_DIR/submit_evaluation_job.sh" \
                --condition "$condition" --size 2B \
                --adapter-dir "$(adapter_dir_for_step "$fit_id" "$step")" \
                --name "eval-${fit_id}-${condition}-${step}-shuf" \
                --run-label "${prefix}-step${step}-shuffled-j${fit_id}" \
                --coordinate-perturbation shuffled \
                --dependency "afterok:${fit_id}" \
                "${COMMON_OVERRIDES[@]}"
        done
    done
done

echo "Submitted 60 evaluation-only trajectory jobs for fits 11807--11814."
