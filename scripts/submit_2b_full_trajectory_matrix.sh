#!/bin/bash
# Submit the corrected 2B core-four full-run matrix at seeds 42 and 43, then
# submit the prespecified dependent trajectory evaluations.

set -euo pipefail

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "Usage: $0 [--dry-run]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [ ! -f .env ]; then
    echo "Missing .env. Copy .env.example to .env and fill in server paths."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
if [ -z "${FINETUNING_OUTPUT_ROOT:-}" ]; then
    echo "FINETUNING_OUTPUT_ROOT is required."
    exit 1
fi

CONDITIONS=(no_loc loc_text loc_embed loc_additive_satclip)
SEEDS=(42 43)
CORRECT_STEPS=(50 100 500 1000 5000 final)
SHUFFLED_STEPS=(1000 final)
declare -A FIT_IDS=()

adapter_dir_for_step() {
    local fit_id="$1"
    local step="$2"
    local run_root="${FINETUNING_OUTPUT_ROOT%/}/bigearthnet_${fit_id}"
    if [ "$step" = "final" ]; then
        printf '%s/qlora_adapter' "$run_root"
    else
        printf '%s/qlora_adapter_steps/step_%06d' "$run_root" "$step"
    fi
}

echo "Submitting all eight fits before their evaluations."
for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        fit_name="q3j-${condition}-2B-full-s${seed}"
        fit_args=(
            --condition "$condition"
            --size 2B
            --name "$fit_name"
            --seed_everything "$seed"
        )
        if [ "$DRY_RUN" = true ]; then
            "$SCRIPT_DIR/submit_finetuning_job.sh" "${fit_args[@]}" --dry-run
            continue
        fi

        fit_output="$("$SCRIPT_DIR/submit_finetuning_job.sh" "${fit_args[@]}")"
        printf '%s\n' "$fit_output"
        fit_id="$(
            printf '%s\n' "$fit_output" \
                | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' \
                | tail -n 1
        )"
        if [ -z "$fit_id" ]; then
            echo "Could not extract the fit job ID for $condition seed $seed."
            exit 1
        fi
        FIT_IDS["${seed}:${condition}"]="$fit_id"
    done
done

if [ "$DRY_RUN" = true ]; then
    echo
    echo "[Dry run] A real submission would add, for each of eight fits:"
    echo "  correct evaluations: ${CORRECT_STEPS[*]}"
    echo "  shuffled evaluations for location conditions: ${SHUFFLED_STEPS[*]}"
    echo "This is 8 fits and 60 dependent evaluations."
    exit 0
fi

echo "All fits submitted. Submitting dependent evaluations."
for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        fit_id="${FIT_IDS["${seed}:${condition}"]}"
        dependency="afterok:${fit_id}"
        run_prefix="qwen3-json-${condition}-2B-full-seed${seed}"

        for step in "${CORRECT_STEPS[@]}"; do
            adapter_dir="$(adapter_dir_for_step "$fit_id" "$step")"
            "$SCRIPT_DIR/submit_evaluation_job.sh" \
                --condition "$condition" \
                --size 2B \
                --adapter-dir "$adapter_dir" \
                --name "eval-${fit_id}-${condition}-${step}" \
                --run-label "${run_prefix}-step${step}-j${fit_id}" \
                --dependency "$dependency"
        done

        if [ "$condition" = "no_loc" ]; then
            continue
        fi
        for step in "${SHUFFLED_STEPS[@]}"; do
            adapter_dir="$(adapter_dir_for_step "$fit_id" "$step")"
            "$SCRIPT_DIR/submit_evaluation_job.sh" \
                --condition "$condition" \
                --size 2B \
                --adapter-dir "$adapter_dir" \
                --name "eval-${fit_id}-${condition}-${step}-shuf" \
                --run-label "${run_prefix}-step${step}-shuffled-j${fit_id}" \
                --coordinate-perturbation shuffled \
                --dependency "$dependency"
        done
    done
done

echo "Submitted corrected 2B full trajectory matrix."
echo "Fit IDs:"
for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        echo "  seed=${seed} condition=${condition} job=${FIT_IDS["${seed}:${condition}"]}"
    done
done
