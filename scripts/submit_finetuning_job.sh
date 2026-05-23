#!/bin/bash
# ============================================================================
# Finetuning Job Submission Helper
# ============================================================================
# Usage:
#   ./scripts/submit_finetuning_job.sh
#   ./scripts/submit_finetuning_job.sh --condition loc_text --size 4B
#   ./scripts/submit_finetuning_job.sh --condition no_loc --full
#   ./scripts/submit_finetuning_job.sh --dry-run
# ============================================================================

set -e

SIZE="2B"
CONDITION="loc_embed"
SMOKE=true
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
        --smoke)
            SMOKE=true
            shift
            ;;
        --full)
            SMOKE=false
            shift
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
    no_loc)
        CONDITION_CONFIG=""
        ;;
    loc_text)
        CONDITION_CONFIG="configs/finetuning/loc_text.yaml"
        ;;
    loc_embed)
        CONDITION_CONFIG="configs/finetuning/loc_embed.yaml"
        ;;
    *)
        echo "Invalid --condition '$CONDITION'. Use no_loc, loc_text or loc_embed."
        exit 1
        ;;
esac

SMOKE_CONFIG=""
RUN_KIND="full"
if [ "$SMOKE" = true ]; then
    SMOKE_CONFIG="configs/finetuning/bigearthnet_txt_smoke.yaml"
    RUN_KIND="smoke"
fi

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="bentxt-${CONDITION}-${SIZE}-${RUN_KIND}"
fi

mkdir -p logs

echo "=============================================="
echo "BigEarthNet.txt Finetuning Job Submission"
echo "=============================================="
echo "Base config: configs/finetuning/bigearthnet_txt_shared.yaml"
echo "Condition config: ${CONDITION_CONFIG:-<none>}"
echo "Smoke config: ${SMOKE_CONFIG:-<none>}"
echo "Condition: $CONDITION"
echo "Run kind: $RUN_KIND"
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
FULL_CMD+=("--export=ALL,CONDITION_CONFIG=$CONDITION_CONFIG,SMOKE_CONFIG=$SMOKE_CONFIG" "$SCRIPT")
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
