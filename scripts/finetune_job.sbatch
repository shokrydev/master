#!/bin/bash
#SBATCH --job-name=vlm-finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
# Single-GPU BigEarthNet.txt finetuning job.

set -e

# Load modules (adjust for your cluster)
# module purge
# module load cuda/12.1
# module load python/3.12

# Activate virtual environment (adjust path)
# source /path/to/venv/bin/activate
# Or conda:
# source ~/.bashrc && conda activate vlm

export UNSLOTH_DISABLE_STATISTICS=1  # skip silent telemetry downloads that stall for minutes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDITION_CONFIG="${CONDITION_CONFIG:-}"
FIT_VALIDATION_CONFIG="${FIT_VALIDATION_CONFIG:-}"
SMOKE_CONFIG="${SMOKE_CONFIG:-}"
CAPTION_TARGET_CONFIG="${CAPTION_TARGET_CONFIG:-}"

cd "$SLURM_SUBMIT_DIR"

if [ ! -f .env ]; then
    echo "Missing .env. Copy .env.example to .env and fill in the server-local paths."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

FINETUNING_OUTPUT_ROOT="${FINETUNING_OUTPUT_ROOT%/}"
OUTPUT_DIR="${FINETUNING_OUTPUT_ROOT}/bigearthnet_${SLURM_JOB_ID}"
export FINETUNING_OUTPUT_DIR="$OUTPUT_DIR"
export FINETUNING_ADAPTER_DIR="${OUTPUT_DIR}/qlora_adapter"
REQUIRED_ENV_VARS=(
    BIGEARTHNET_V2_LMDB_ROOT
    BIGEARTHNET_TXT_PARQUET_PATH
    BIGEARTHNET_ENCODER_DIR
    FINETUNING_OUTPUT_ROOT
    HF_HOME
)
USES_SATCLIP_L40=false
for CLI_ARG in "$@"; do
    if [ "$CLI_ARG" = "configs/finetuning/ablations/loc_embed_satclip_l40.yaml" ]; then
        USES_SATCLIP_L40=true
        break
    fi
done
if [ "$CONDITION_CONFIG" = "configs/finetuning/loc_embed.yaml" ]; then
    if [ "$USES_SATCLIP_L40" = true ]; then
        REQUIRED_ENV_VARS+=(SATCLIP_L40_CHECKPOINT_PATH)
    else
        REQUIRED_ENV_VARS+=(SATCLIP_CHECKPOINT_PATH)
    fi
fi
if [ "$CONDITION_CONFIG" = "configs/finetuning/loc_additive_satclip.yaml" ]; then
    REQUIRED_ENV_VARS+=(SATCLIP_L40_CHECKPOINT_PATH)
fi
if [ "$CAPTION_TARGET_CONFIG" = "configs/finetuning/location_redacted_captions.yaml" ]; then
    REQUIRED_ENV_VARS+=(BENTXT_LOCATION_REDACTED_CAPTION_FILE)
elif [ -n "$CAPTION_TARGET_CONFIG" ]; then
    echo "Unsupported CAPTION_TARGET_CONFIG: $CAPTION_TARGET_CONFIG"
    exit 1
fi

MISSING_ENV_VARS=()
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    if [ -z "${!VAR_NAME:-}" ]; then
        MISSING_ENV_VARS+=("$VAR_NAME")
    fi
done
if [ "${#MISSING_ENV_VARS[@]}" -gt 0 ]; then
    echo "Missing required env vars: ${MISSING_ENV_VARS[*]}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CONFIG_ARGS=(--config configs/finetuning/bigearthnet_txt_shared.yaml)
if [ -n "$FIT_VALIDATION_CONFIG" ]; then
    CONFIG_ARGS+=(--config "$FIT_VALIDATION_CONFIG")
fi
if [ -n "$CONDITION_CONFIG" ]; then
    CONFIG_ARGS+=(--config "$CONDITION_CONFIG")
fi
if [ -n "$SMOKE_CONFIG" ]; then
    CONFIG_ARGS+=(--config "$SMOKE_CONFIG")
fi
if [ -n "$CAPTION_TARGET_CONFIG" ]; then
    CONFIG_ARGS+=(--config "$CAPTION_TARGET_CONFIG")
fi

printf 'Config args:'
printf ' %q' "${CONFIG_ARGS[@]}"
printf '\n'
echo "Fit validation config: ${FIT_VALIDATION_CONFIG:-<none>}"
echo "Caption target config: ${CAPTION_TARGET_CONFIG:-<none>}"
echo "Output dir: $OUTPUT_DIR"
echo "Adapter dir: $FINETUNING_ADAPTER_DIR"
echo "Required paths:"
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    echo "  $VAR_NAME=${!VAR_NAME}"
done
if [ -z "$CAPTION_TARGET_CONFIG" ]; then
    echo "Optional paths:"
    echo "  BENTXT_LOCATION_REDACTED_CAPTION_FILE=${BENTXT_LOCATION_REDACTED_CAPTION_FILE:-<not set>}"
fi

uv run python main.py fit \
    "${CONFIG_ARGS[@]}" \
    --trainer.default_root_dir "$OUTPUT_DIR" \
    --trainer.devices 1 \
    --trainer.num_nodes 1 \
    "$@"
