#!/bin/bash
# ============================================================================
# Finetuning Job Submission Helper
# ============================================================================
# Usage:
#   ./scripts/submit_finetuning_job.sh
#   ./scripts/submit_finetuning_job.sh --condition loc_text --size 4B
#   ./scripts/submit_finetuning_job.sh --dry-run
# ============================================================================

set -e

SIZE="2B"
CONDITION="baseline"
JOB_NAME=""
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            SIZE="$2"
            shift 2
            ;;
        --condition)
            CONDITION="$2"
            shift 2
            ;;
        --name)
            JOB_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$SIZE" in
    2B)
        MODEL_NAME="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
        ;;
    4B)
        MODEL_NAME="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
        ;;
    8B)
        MODEL_NAME="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
        ;;
    *)
        echo "Invalid --size '$SIZE'. Use 2B, 4B or 8B."
        exit 1
        ;;
esac

case "$CONDITION" in
    baseline)
        OVERRIDE_CONFIG=""
        ;;
    loc_text)
        OVERRIDE_CONFIG="configs/finetuning/loc_text.yaml"
        ;;
    loc_embed)
        OVERRIDE_CONFIG="configs/finetuning/loc_embed.yaml"
        ;;
    *)
        echo "Invalid --condition '$CONDITION'. Use baseline, loc_text or loc_embed."
        exit 1
        ;;
esac

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="finetune-${CONDITION}-${SIZE}"
fi

mkdir -p logs

echo "=============================================="
echo "Finetuning Job Submission"
echo "=============================================="
echo "Override config: ${OVERRIDE_CONFIG:-<none>}"
echo "Condition: $CONDITION"
echo "Size: $SIZE"
echo "Model: $MODEL_NAME"
echo "Job name: $JOB_NAME"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "=============================================="

SCRIPT="scripts/finetune_job.sbatch"
FULL_CMD=(
    sbatch
    "--job-name=$JOB_NAME"
)
FULL_CMD+=("--export=ALL,OVERRIDE_CONFIG=$OVERRIDE_CONFIG" "$SCRIPT")
FULL_CMD+=("--model.init_args.model_name_or_path" "$MODEL_NAME")
FULL_CMD+=("${EXTRA_ARGS[@]}")

printf 'Command:'
printf ' %q' "${FULL_CMD[@]}"
printf '\n\n'

if [ "$DRY_RUN" = true ]; then
    echo "[Dry run - not submitting]"
else
    echo "Submitting job..."
    "${FULL_CMD[@]}"
fi
