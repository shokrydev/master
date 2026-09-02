#!/bin/bash
# Submit two full 2B loc_embed placement-ablation fits and their complete
# correct/shuffled trajectory evaluations plus one CLAIR job per fit.

set -euo pipefail

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

CONFIG="configs/finetuning/ablations/loc_embed_after_marker.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "Missing ablation config: $CONFIG"
    exit 1
fi
if [ ! -f .env ]; then
    echo "Missing .env."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

SEEDS=(42 43)
declare -A FIT_JOBS=()
timestamp="$(date +%Y%m%d_%H%M%S)"
FIT_MANIFEST="outputs/submission_manifests/2b_loc_embed_after_marker_fits_${timestamp}.tsv"

submit_fit() {
    local seed="$1"
    local args=(
        --condition loc_embed
        --size 2B
        --name "q3j-loc_embed-after-marker-2B-full-s${seed}"
        --config "$CONFIG"
        --seed_everything "$seed"
        --data.init_args.num_workers_dataloader 8
    )
    if [ "$DRY_RUN" = true ]; then
        args+=(--dry-run)
    fi

    local output
    output="$("$SCRIPT_DIR/submit_finetuning_job.sh" "${args[@]}")"
    printf '%s\n' "$output"
    if [ "$DRY_RUN" = true ]; then
        FIT_JOBS["$seed"]="90${seed}"
        return
    fi
    local job_id
    job_id="$(
        printf '%s\n' "$output" \
            | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' \
            | tail -n 1
    )"
    if [ -z "$job_id" ]; then
        echo "Could not extract fit job ID for seed $seed."
        exit 1
    fi
    FIT_JOBS["$seed"]="$job_id"
}

# Submit both fits before any dependent evaluation jobs enter the queue.
for seed in "${SEEDS[@]}"; do
    submit_fit "$seed"
done

if [ "$DRY_RUN" = false ]; then
    mkdir -p "$(dirname "$FIT_MANIFEST")"
    printf 'fit_job\tcondition\tseed\tvariant\tconfig\n' > "$FIT_MANIFEST"
    for seed in "${SEEDS[@]}"; do
        printf '%s\tloc_embed\t%s\tafter_location_marker\t%s\n' \
            "${FIT_JOBS[$seed]}" "$seed" "$CONFIG" >> "$FIT_MANIFEST"
    done
    echo "Wrote fit manifest: $FIT_MANIFEST"
else
    echo "[Dry run] Synthetic fit IDs below are placeholders used only to print dependent commands."
fi

for seed in "${SEEDS[@]}"; do
    fit_job="${FIT_JOBS[$seed]}"
    eval_args=(
        --fit-job "$fit_job"
        --condition loc_embed
        --seed "$seed"
        --dependency "afterok:${fit_job}"
        --config "$CONFIG"
        --run-prefix "qwen3-json-loc_embed-after-marker-2B-full-seed${seed}"
        --submit-clair
    )
    if [ "$DRY_RUN" = true ]; then
        eval_args+=(--dry-run)
    fi
    "$SCRIPT_DIR/submit_2b_trajectory_evaluations.sh" "${eval_args[@]}"
done

if [ "$DRY_RUN" = true ]; then
    echo "[Dry run] Real submission creates 2 fits, 16 evaluations and 2 fit-level CLAIR jobs."
else
    echo "Submitted 2 fits, 16 dependent evaluations and 2 dependent fit-level CLAIR jobs."
fi
