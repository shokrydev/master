#!/bin/bash
#SBATCH --job-name=bentxt-clair
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ ! -f .env ]; then
    echo "Missing .env."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

CLAIR_MODEL_NAME_OR_PATH="${CLAIR_MODEL_NAME_OR_PATH:-unsloth/Qwen3.8-27B-unsloth-bnb-4bit}"
if [ -z "${EVALUATION_OUTPUT_ROOT:-}" ]; then
    echo "EVALUATION_OUTPUT_ROOT is required."
    exit 1
fi

uv run python -c \
    'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

scorer_args=(
    --model-name-or-path "$CLAIR_MODEL_NAME_OR_PATH"
    "$@"
)

if [ -n "${CLAIR_TRAJECTORY_MANIFEST:-}" ] || [ -n "${CLAIR_FIT_JOB:-}" ]; then
    if [ -z "${CLAIR_TRAJECTORY_MANIFEST:-}" ] || [ -z "${CLAIR_FIT_JOB:-}" ]; then
        echo "CLAIR_TRAJECTORY_MANIFEST and CLAIR_FIT_JOB must be set together."
        exit 1
    fi
    if [ ! -f "$CLAIR_TRAJECTORY_MANIFEST" ]; then
        echo "Trajectory manifest is unavailable: $CLAIR_TRAJECTORY_MANIFEST"
        exit 1
    fi
    run_output="${EVALUATION_OUTPUT_ROOT%/}/clair_fit_${CLAIR_FIT_JOB}_${SLURM_JOB_ID}"
    mkdir -p "$run_output"
    scored_count=0
    while IFS=$'\t' read -r evaluation_job fit_job condition seed step coordinate_setting adapter_dir run_label; do
        if [ "$evaluation_job" = evaluation_job ] || [ "$fit_job" != "$CLAIR_FIT_JOB" ]; then
            continue
        fi
        predictions="${EVALUATION_OUTPUT_ROOT%/}/bigearthnet_${evaluation_job}/predictions.jsonl"
        clair_output="${EVALUATION_OUTPUT_ROOT%/}/bigearthnet_${evaluation_job}/clair_qwen38_bnb4"
        if [ ! -f "$predictions" ]; then
            echo "Missing predictions for evaluation job $evaluation_job: $predictions"
            exit 1
        fi
        echo "Queued export $evaluation_job: condition=$condition seed=$seed step=$step setting=$coordinate_setting"
        scorer_args+=(--score "$predictions" "$clair_output")
        scored_count=$((scored_count + 1))
    done < "$CLAIR_TRAJECTORY_MANIFEST"

    if [ "$scored_count" -ne 6 ] && [ "$scored_count" -ne 8 ]; then
        echo "Expected 6 or 8 trajectory exports for fit $CLAIR_FIT_JOB, found $scored_count."
        exit 1
    fi
    printf '%s\n' "$scored_count" > "$run_output/scored_export_count.txt"
    echo "CLAIR trajectory: fit $CLAIR_FIT_JOB, $scored_count exports"
else
    if [ "$#" -lt 1 ]; then
        echo "Single-export mode requires a predictions.jsonl argument."
        exit 1
    fi
    predictions="$1"
    shift
    if [ ! -f "$predictions" ]; then
        echo "Predictions are unavailable: $predictions"
        exit 1
    fi
    run_output="${EVALUATION_OUTPUT_ROOT%/}/clair_${SLURM_JOB_ID}"
    mkdir -p "$run_output"
    scorer_args=(
        --model-name-or-path "$CLAIR_MODEL_NAME_OR_PATH"
        "$@"
        --score "$predictions" "$run_output"
    )
    echo "CLAIR single export: $predictions"
fi

echo "Judge model: $CLAIR_MODEL_NAME_OR_PATH"
printf 'Scorer command: uv run python scripts/score_bentxt_clair.py'
printf ' %q' "${scorer_args[@]}"
printf '\n'
uv run python scripts/score_bentxt_clair.py "${scorer_args[@]}"
