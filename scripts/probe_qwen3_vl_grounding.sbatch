#!/bin/bash
#SBATCH --job-name=qwen3-grounding-probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -e

export UNSLOTH_DISABLE_STATISTICS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

if [ ! -f .env ]; then
    echo "Missing .env. Copy .env.example to .env and fill in the server-local paths."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

OUTPUT_DIR="${FINETUNING_OUTPUT_ROOT%/}/grounding_probe_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

PROBE_ARGS=(
    --model "$GROUNDING_PROBE_MODEL"
    --image-lmdb-file "$BIGEARTHNET_V2_LMDB_ROOT"
    --metadata-file "$BIGEARTHNET_TXT_PARQUET_PATH"
    --output-dir "$OUTPUT_DIR"
    --num-examples "${GROUNDING_PROBE_NUM_EXAMPLES:-8}"
)
if [ -n "${GROUNDING_PROBE_ADAPTER_DIR:-}" ]; then
    PROBE_ARGS+=(--adapter-dir "$GROUNDING_PROBE_ADAPTER_DIR")
fi

uv run python scripts/probe_qwen3_vl_grounding.py "${PROBE_ARGS[@]}"
