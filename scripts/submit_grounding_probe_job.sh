#!/bin/bash
# Submit the frozen Qwen3-VL grounding/interface probe before corrected runs.

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
ADAPTER_DIR=""
NUM_EXAMPLES=8
PARTITION="${SLURM_DEFAULT_PARTITION:-}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE="$2"; shift 2 ;;
        --adapter-dir) ADAPTER_DIR="$2"; shift 2 ;;
        --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "$SIZE" in
    2B) MODEL="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit" ;;
    4B) MODEL="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit" ;;
    8B) MODEL="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit" ;;
    *) echo "Invalid --size '$SIZE'. Use 2B, 4B or 8B."; exit 1 ;;
esac

if [ -z "$PARTITION" ]; then
    echo "Missing Slurm partition. Set SLURM_DEFAULT_PARTITION or pass --partition."
    exit 1
fi

REQUIRED_ENV_VARS=(
    BIGEARTHNET_V2_LMDB_ROOT
    BIGEARTHNET_TXT_PARQUET_PATH
    FINETUNING_OUTPUT_ROOT
    HF_HOME
)
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    if [ -z "${!VAR_NAME:-}" ] && [ "$DRY_RUN" = false ]; then
        echo "Missing required env var: $VAR_NAME"
        exit 1
    fi
done

mkdir -p logs
CMD=(
    sbatch
    --parsable
    "--partition=$PARTITION"
    "--job-name=qwen3-grounding-${SIZE}"
    "--export=ALL,GROUNDING_PROBE_MODEL=$MODEL,GROUNDING_PROBE_NUM_EXAMPLES=$NUM_EXAMPLES,GROUNDING_PROBE_ADAPTER_DIR=$ADAPTER_DIR"
    scripts/probe_qwen3_vl_grounding.sbatch
)
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "$DRY_RUN" = true ]; then
    echo "[Dry run - not submitting]"
else
    "${CMD[@]}"
fi
