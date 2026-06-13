#!/bin/bash
#SBATCH --job-name=bentxt-profile
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# BigEarthNet.txt batch-size profiling job.

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

REQUIRED_ENV_VARS=(
    BIGEARTHNET_V2_LMDB_ROOT
    BIGEARTHNET_TXT_PARQUET_PATH
    BIGEARTHNET_ENCODER_DIR
    HF_HOME
)

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

mkdir -p logs outputs/batch_profiles

echo "BigEarthNet.txt batch-size profiling"
echo "Required paths:"
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    echo "  $VAR_NAME=${!VAR_NAME}"
done
echo "Optional paths:"
echo "  SATCLIP_CHECKPOINT_PATH=${SATCLIP_CHECKPOINT_PATH:-<not set>}"
echo "Profiler args: $*"

uv run python scripts/profile_bentxt_batch_size.py "$@"
