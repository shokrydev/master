#!/bin/bash
#SBATCH --job-name=vlm-finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
# Single-GPU finetuning job.

set -e

# Load modules (adjust for your cluster)
# module purge
# module load cuda/12.1
# module load python/3.12

# Activate virtual environment (adjust path)
# source /path/to/venv/bin/activate
# Or conda:
# source ~/.bashrc && conda activate vlm

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export UNSLOTH_DISABLE_STATISTICS=1  # skip silent telemetry downloads that stall for minutes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OVERRIDE_CONFIG="${OVERRIDE_CONFIG:-}"
OUTPUT_DIR="${FINETUNING_OUTPUT_DIR:-outputs/run_${SLURM_JOB_ID}}"

mkdir -p "$OUTPUT_DIR"

cd "$SLURM_SUBMIT_DIR"

CONFIG_ARGS=(--config configs/finetuning/gaia_finetuning_shared.yaml)
if [ -n "$OVERRIDE_CONFIG" ]; then
    CONFIG_ARGS+=(--config "$OVERRIDE_CONFIG")
fi

python main.py fit \
    "${CONFIG_ARGS[@]}" \
    --trainer.default_root_dir "$OUTPUT_DIR" \
    --trainer.devices 1 \
    --trainer.num_nodes 1 \
    "$@"
