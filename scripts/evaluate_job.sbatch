#!/bin/bash
#SBATCH --job-name=vlm-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
# Single-GPU BigEarthNet.txt prediction export job.

set -e

export UNSLOTH_DISABLE_STATISTICS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDITION_CONFIG="${CONDITION_CONFIG:-}"
RUN_LABEL="${RUN_LABEL:-}"
MODEL_SIZE="${MODEL_SIZE:-}"
SUBMITTED_EVAL_ADAPTER_DIR="${EVAL_ADAPTER_DIR:-}"

cd "$SLURM_SUBMIT_DIR"

if [ ! -f .env ]; then
    echo "Missing .env. Copy .env.example to .env and fill in the server-local paths."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
if [ -n "$SUBMITTED_EVAL_ADAPTER_DIR" ]; then
    export EVAL_ADAPTER_DIR="$SUBMITTED_EVAL_ADAPTER_DIR"
fi
if [ -z "${EVALUATION_OUTPUT_ROOT:-}" ] && [ -z "${FINETUNING_OUTPUT_ROOT:-}" ]; then
    echo "Set EVALUATION_OUTPUT_ROOT or FINETUNING_OUTPUT_ROOT in .env."
    exit 1
fi

EVALUATION_OUTPUT_ROOT="${EVALUATION_OUTPUT_ROOT:-${FINETUNING_OUTPUT_ROOT%/}/evaluation_outputs}"
EVALUATION_OUTPUT_ROOT="${EVALUATION_OUTPUT_ROOT%/}"
OUTPUT_DIR="${EVALUATION_OUTPUT_ROOT}/bigearthnet_${SLURM_JOB_ID}"
export EVAL_OUTPUT_DIR="$OUTPUT_DIR"

REQUIRED_ENV_VARS=(
    BIGEARTHNET_V2_LMDB_ROOT
    BIGEARTHNET_TXT_PARQUET_PATH
    BIGEARTHNET_ENCODER_DIR
    EVAL_ADAPTER_DIR
    EVAL_OUTPUT_DIR
    HF_HOME
)
if [ "$CONDITION_CONFIG" = "configs/finetuning/loc_embed.yaml" ]; then
    REQUIRED_ENV_VARS+=(SATCLIP_CHECKPOINT_PATH)
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
if [ ! -d "$EVAL_ADAPTER_DIR" ]; then
    echo "EVAL_ADAPTER_DIR is not a directory: $EVAL_ADAPTER_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CONFIG_ARGS=(--config configs/evaluation/bigearthnet_txt.yaml)
if [ -n "$CONDITION_CONFIG" ]; then
    CONFIG_ARGS+=(--config "$CONDITION_CONFIG")
fi

printf 'Config args:'
printf ' %q' "${CONFIG_ARGS[@]}"
printf '\n'
echo "Condition config: ${CONDITION_CONFIG:-<none>}"
echo "Output dir: $OUTPUT_DIR"
echo "Adapter dir: $EVAL_ADAPTER_DIR"
echo "Prediction export: ${OUTPUT_DIR}/predictions.jsonl"
echo "Run label: ${RUN_LABEL:-<none>}"
echo "Model size: ${MODEL_SIZE:-<none>}"
echo "Required paths:"
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    echo "  $VAR_NAME=${!VAR_NAME}"
done

uv run python main.py test \
    "${CONFIG_ARGS[@]}" \
    --trainer.default_root_dir "$OUTPUT_DIR" \
    --trainer.devices 1 \
    --trainer.num_nodes 1 \
    --model.init_args.run_label "$RUN_LABEL" \
    --model.init_args.model_size "$MODEL_SIZE" \
    "$@"
