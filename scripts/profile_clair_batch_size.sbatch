#!/bin/bash
#SBATCH --job-name=profile-clair-batches
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a
# shellcheck disable=SC1091
source .env
set +a
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${EVALUATION_OUTPUT_ROOT:-}" ]; then
    echo "EVALUATION_OUTPUT_ROOT is required."
    exit 1
fi
if [ -z "${CLAIR_PYTHON:-}" ] || [ ! -x "$CLAIR_PYTHON" ]; then
    echo "CLAIR_PYTHON must identify the executable CLAIR environment Python."
    exit 1
fi

output_dir="${EVALUATION_OUTPUT_ROOT%/}/clair_batch_profile_${SLURM_JOB_ID}"
mkdir -p "$output_dir"
"$CLAIR_PYTHON" scripts/profile_clair_batch_size.py \
    --output "$output_dir/profile.json" \
    "$@"
