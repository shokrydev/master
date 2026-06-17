#!/bin/bash
# ============================================================================
# Full Finetuning Job Submission Helper
# ============================================================================
# Usage:
#   ./scripts/submit_finetuning_job.sh
#   ./scripts/submit_finetuning_job.sh --condition loc_text --size 4B
#   ./scripts/submit_finetuning_job.sh --caption-target location_redacted_caption
#   ./scripts/submit_finetuning_job.sh --size 8B --partition big_job --time 7-00:00:00 --mem 128G --cpus 14
#   ./scripts/submit_finetuning_job.sh --dry-run
# ============================================================================

set -e

if [ ! -f .env ]; then
    echo "Missing .env. Copy .env.example to .env and fill in the server-local paths."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

SIZE="2B"
CONDITION="loc_embed"
CAPTION_TARGET="caption"
JOB_NAME=""
PARTITION="${SLURM_DEFAULT_PARTITION:-}"
TIME_LIMIT=""
MEMORY=""
CPUS=""
DRY_RUN=false
EXTRA_ARGS=()

require_arg() {
    if [ -z "${2:-}" ]; then
        echo "Missing value for $1"
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            require_arg "$1" "${2:-}"
            SIZE="$2"
            shift 2
            ;;
        --condition)
            require_arg "$1" "${2:-}"
            CONDITION="$2"
            shift 2
            ;;
        --caption-target)
            require_arg "$1" "${2:-}"
            CAPTION_TARGET="$2"
            shift 2
            ;;
        --name)
            require_arg "$1" "${2:-}"
            JOB_NAME="$2"
            shift 2
            ;;
        --partition)
            require_arg "$1" "${2:-}"
            PARTITION="$2"
            shift 2
            ;;
        --time)
            require_arg "$1" "${2:-}"
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem)
            require_arg "$1" "${2:-}"
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            require_arg "$1" "${2:-}"
            CPUS="$2"
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

case "$CAPTION_TARGET" in
    caption)
        CAPTION_TARGET_CONFIG=""
        ;;
    location_redacted_caption)
        CAPTION_TARGET_CONFIG="configs/finetuning/location_redacted_captions.yaml"
        ;;
    *)
        echo "Invalid --caption-target '$CAPTION_TARGET'. Use caption or location_redacted_caption."
        exit 1
        ;;
esac

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

SMOKE_CONFIG="${SMOKE_CONFIG:-}"
RUN_KIND="full"
if [ -n "$SMOKE_CONFIG" ]; then
    RUN_KIND="smoke"
fi
VALIDATION_CONFIG=""
if [ "$SIZE" = "2B" ] && [ "$RUN_KIND" = "full" ]; then
    VALIDATION_CONFIG="configs/finetuning/bigearthnet_txt_bounded_validation_monitor.yaml"
fi

REQUIRED_ENV_VARS=(
    BIGEARTHNET_V2_LMDB_ROOT
    BIGEARTHNET_TXT_PARQUET_PATH
    BIGEARTHNET_ENCODER_DIR
    FINETUNING_OUTPUT_ROOT
    HF_HOME
)
if [ "$CONDITION" = "loc_embed" ]; then
    REQUIRED_ENV_VARS+=(SATCLIP_CHECKPOINT_PATH)
fi
if [ "$CAPTION_TARGET" = "location_redacted_caption" ]; then
    REQUIRED_ENV_VARS+=(BENTXT_LOCATION_REDACTED_CAPTION_FILE)
fi

MISSING_ENV_VARS=()
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    if [ -z "${!VAR_NAME:-}" ]; then
        MISSING_ENV_VARS+=("$VAR_NAME")
    fi
done

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="${CONDITION}-${SIZE}-${RUN_KIND}"
fi

if [ -z "$PARTITION" ]; then
    echo "Missing Slurm partition. Set SLURM_DEFAULT_PARTITION in .env or pass --partition."
    exit 1
fi
mkdir -p logs

echo "=============================================="
echo "BigEarthNet.txt Finetuning Job Submission"
echo "=============================================="
echo "Base config: configs/finetuning/bigearthnet_txt_shared.yaml"
echo "Validation config: ${VALIDATION_CONFIG:-<none>}"
echo "Condition config: ${CONDITION_CONFIG:-<none>}"
echo "Caption target config: ${CAPTION_TARGET_CONFIG:-<none>}"
echo "Smoke config: ${SMOKE_CONFIG:-<none>}"
echo "Condition: $CONDITION"
echo "Caption target: $CAPTION_TARGET"
echo "Run kind: $RUN_KIND"
echo "Size: $SIZE"
echo "Model: $MODEL_NAME"
echo "Job name: $JOB_NAME"
echo "Slurm partition: $PARTITION"
echo "Slurm time limit: ${TIME_LIMIT:-<partition default>}"
echo "Slurm memory: ${MEMORY:-<sbatch default>}"
echo "Slurm CPUs per task: ${CPUS:-<sbatch default>}"
echo "Required paths:"
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    VALUE="${!VAR_NAME:-<missing>}"
    echo "  $VAR_NAME=$VALUE"
done
if [ "$CAPTION_TARGET" = "caption" ]; then
    echo "Optional paths:"
    echo "  BENTXT_LOCATION_REDACTED_CAPTION_FILE=${BENTXT_LOCATION_REDACTED_CAPTION_FILE:-<not set>}"
fi
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "=============================================="

SCRIPT="scripts/finetune_job.sbatch"
FULL_CMD=(
    sbatch
    "--job-name=$JOB_NAME"
    "--partition=$PARTITION"
)
if [ -n "$TIME_LIMIT" ]; then
    FULL_CMD+=("--time=$TIME_LIMIT")
fi
if [ -n "$MEMORY" ]; then
    FULL_CMD+=("--mem=$MEMORY")
fi
if [ -n "$CPUS" ]; then
    FULL_CMD+=("--cpus-per-task=$CPUS")
fi
FULL_CMD+=("--export=ALL,CONDITION_CONFIG=$CONDITION_CONFIG,VALIDATION_CONFIG=$VALIDATION_CONFIG,SMOKE_CONFIG=$SMOKE_CONFIG,CAPTION_TARGET_CONFIG=$CAPTION_TARGET_CONFIG" "$SCRIPT")
FULL_CMD+=("--model.init_args.model_name_or_path" "$MODEL_NAME")
FULL_CMD+=("${EXTRA_ARGS[@]}")

printf 'Command:'
printf ' %q' "${FULL_CMD[@]}"
printf '\n\n'

if [ "$DRY_RUN" = true ]; then
    if [ "${#MISSING_ENV_VARS[@]}" -gt 0 ]; then
        echo "Missing required env vars for real submission: ${MISSING_ENV_VARS[*]}"
        echo "[Dry run - not submitting]"
        exit 0
    fi
    echo "[Dry run - not submitting]"
else
    if [ "${#MISSING_ENV_VARS[@]}" -gt 0 ]; then
        echo "Missing required env vars: ${MISSING_ENV_VARS[*]}"
        echo "Set them before submitting so Slurm can inherit them via --export=ALL."
        exit 1
    fi
    echo "Submitting job..."
    "${FULL_CMD[@]}"
fi
