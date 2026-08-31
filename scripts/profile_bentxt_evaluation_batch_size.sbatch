#!/bin/bash
#SBATCH --job-name=bentxt-eval-profile
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G

set -e

export UNSLOTH_DISABLE_STATISTICS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

if [ ! -f .env ]; then
    echo "Missing .env. Copy .env.example to .env and fill in server paths."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

for var_name in \
    BIGEARTHNET_V2_LMDB_ROOT \
    BIGEARTHNET_TXT_PARQUET_PATH \
    BIGEARTHNET_ENCODER_DIR \
    EVALUATION_OUTPUT_ROOT \
    HF_HOME \
    SATCLIP_L40_CHECKPOINT_PATH; do
    if [ -z "${!var_name:-}" ]; then
        echo "Missing required env var: $var_name"
        exit 1
    fi
done

output_dir="${EVALUATION_OUTPUT_ROOT%/}/batch_profile_${SLURM_JOB_ID}"
mkdir -p "$output_dir"

echo "Evaluation batch profile output: $output_dir"
echo "Profiler args: $*"

uv run python scripts/profile_bentxt_evaluation_batch_size.py \
    --json-output "$output_dir/summary.json" \
    "$@"
