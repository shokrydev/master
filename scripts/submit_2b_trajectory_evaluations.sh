#!/bin/bash
# Submit only the prespecified evaluations for the already-running 2B fits
# 11807--11814. The all-in-one fit+evaluation helper remains unchanged.

set -euo pipefail

SHORT_BATCH=""
BBOX_BATCH=""
CAPTION_BATCH=""
SHORT_WORKERS=""
BBOX_WORKERS=""
CAPTION_WORKERS=""
MANIFEST=""
SUBMIT_CLAIR=false
CLAIR_MODEL_PATH="${CLAIR_MODEL_PATH:-}"
CLAIR_CONCURRENCY=8
CLAIR_MAX_NEW_TOKENS=512
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --short-batch) SHORT_BATCH="${2:-}"; shift 2 ;;
        --bbox-batch) BBOX_BATCH="${2:-}"; shift 2 ;;
        --caption-batch) CAPTION_BATCH="${2:-}"; shift 2 ;;
        --short-workers) SHORT_WORKERS="${2:-}"; shift 2 ;;
        --bbox-workers) BBOX_WORKERS="${2:-}"; shift 2 ;;
        --caption-workers) CAPTION_WORKERS="${2:-}"; shift 2 ;;
        --manifest) MANIFEST="${2:-}"; shift 2 ;;
        --submit-clair) SUBMIT_CLAIR=true; shift ;;
        --clair-model-path) CLAIR_MODEL_PATH="${2:-}"; shift 2 ;;
        --clair-concurrency) CLAIR_CONCURRENCY="${2:-}"; shift 2 ;;
        --clair-max-new-tokens) CLAIR_MAX_NEW_TOKENS="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

values=("$SHORT_BATCH" "$BBOX_BATCH" "$CAPTION_BATCH" "$SHORT_WORKERS" "$BBOX_WORKERS" "$CAPTION_WORKERS")
for value in "${values[@]}"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "All three batch sizes and worker counts must be positive integers from profiler 11836."
        exit 1
    fi
done
if ! [[ "$CLAIR_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || ! [[ "$CLAIR_MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CLAIR concurrency and max-new-tokens must be positive integers."
    exit 1
fi
if [ "$SUBMIT_CLAIR" = true ] && [ -z "$CLAIR_MODEL_PATH" ]; then
    echo "--submit-clair requires --clair-model-path or CLAIR_MODEL_PATH."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
if [ ! -f .env ]; then
    echo "Missing .env."
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
if [ "$SUBMIT_CLAIR" = true ] && [ ! -f "$CLAIR_MODEL_PATH" ]; then
    echo "CLAIR GGUF does not exist: $CLAIR_MODEL_PATH"
    exit 1
fi
if [ -z "$MANIFEST" ]; then
    MANIFEST="outputs/submission_manifests/2b_trajectory_evaluations_$(date +%Y%m%d_%H%M%S).tsv"
fi

CONDITIONS=(no_loc loc_text loc_embed loc_additive_satclip)
SEEDS=(42 43)
CORRECT_STEPS=(50 100 500 1000 5000 final)
SHUFFLED_STEPS=(1000 final)
declare -A FIT_IDS=(
    [42:no_loc]=11807
    [42:loc_text]=11808
    [42:loc_embed]=11809
    [42:loc_additive_satclip]=11810
    [43:no_loc]=11811
    [43:loc_text]=11812
    [43:loc_embed]=11813
    [43:loc_additive_satclip]=11814
)

adapter_dir_for_step() {
    local fit_id="$1"
    local step="$2"
    local root="${FINETUNING_OUTPUT_ROOT%/}/bigearthnet_${fit_id}"
    if [ "$step" = final ]; then
        printf '%s/qlora_adapter' "$root"
    else
        printf '%s/qlora_adapter_steps/step_%06d' "$root" "$step"
    fi
}

COMMON_OVERRIDES=(
    --data.init_args.evaluation_batch_sizes.short_answer "$SHORT_BATCH"
    --data.init_args.evaluation_batch_sizes.bounding_box "$BBOX_BATCH"
    --data.init_args.evaluation_batch_sizes.captioning "$CAPTION_BATCH"
    --data.init_args.evaluation_num_workers_by_bucket.short_answer "$SHORT_WORKERS"
    --data.init_args.evaluation_num_workers_by_bucket.bounding_box "$BBOX_WORKERS"
    --data.init_args.evaluation_num_workers_by_bucket.captioning "$CAPTION_WORKERS"
)
if [ "$DRY_RUN" = true ]; then
    COMMON_OVERRIDES+=(--dry-run)
else
    mkdir -p "$(dirname "$MANIFEST")"
    printf 'evaluation_job\tfit_job\tcondition\tseed\tstep\tcoordinate_setting\tadapter_dir\trun_label\n' > "$MANIFEST"
fi

submit_evaluation() {
    local fit_id="$1"
    local condition="$2"
    local seed="$3"
    local step="$4"
    local coordinate_setting="$5"
    local adapter_dir="$6"
    local run_label="$7"
    shift 7

    local output
    output="$(
        "$SCRIPT_DIR/submit_evaluation_job.sh" \
            --condition "$condition" --size 2B \
            --adapter-dir "$adapter_dir" \
            --run-label "$run_label" \
            --dependency "afterok:${fit_id}" \
            "$@" \
            "${COMMON_OVERRIDES[@]}"
    )"
    printf '%s\n' "$output"
    if [ "$DRY_RUN" = true ]; then
        return
    fi

    local evaluation_job
    evaluation_job="$(
        printf '%s\n' "$output" \
            | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' \
            | tail -n 1
    )"
    if [ -z "$evaluation_job" ]; then
        echo "Could not extract evaluation job ID for fit=$fit_id condition=$condition step=$step setting=$coordinate_setting."
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$evaluation_job" "$fit_id" "$condition" "$seed" "$step" \
        "$coordinate_setting" "$adapter_dir" "$run_label" >> "$MANIFEST"
}

for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        fit_id="${FIT_IDS["${seed}:${condition}"]}"
        prefix="qwen3-json-${condition}-2B-full-seed${seed}"
        for step in "${CORRECT_STEPS[@]}"; do
            adapter_dir="$(adapter_dir_for_step "$fit_id" "$step")"
            run_label="${prefix}-step${step}-j${fit_id}"
            submit_evaluation \
                "$fit_id" "$condition" "$seed" "$step" correct \
                "$adapter_dir" "$run_label" \
                --name "eval-${fit_id}-${condition}-${step}"
        done
        if [ "$condition" = no_loc ]; then
            continue
        fi
        for step in "${SHUFFLED_STEPS[@]}"; do
            adapter_dir="$(adapter_dir_for_step "$fit_id" "$step")"
            run_label="${prefix}-step${step}-shuffled-j${fit_id}"
            submit_evaluation \
                "$fit_id" "$condition" "$seed" "$step" shuffled \
                "$adapter_dir" "$run_label" \
                --name "eval-${fit_id}-${condition}-${step}-shuf" \
                --coordinate-perturbation shuffled
        done
    done
done

echo "Submitted 60 evaluation-only trajectory jobs for fits 11807--11814."
if [ "$DRY_RUN" = false ]; then
    echo "Wrote submission manifest: $MANIFEST"
fi

if [ "$SUBMIT_CLAIR" = false ]; then
    exit 0
fi
if [ "$DRY_RUN" = true ]; then
    echo "[Dry run] A real submission would add eight fit-level CLAIR jobs."
    echo "Each would depend on its fit's 6 or 8 evaluation jobs and load the Q6 judge once."
    exit 0
fi

if [ -z "${SLURM_DEFAULT_PARTITION:-}" ]; then
    echo "SLURM_DEFAULT_PARTITION is required for fit-level CLAIR submissions."
    exit 1
fi
manifest_absolute="$(realpath "$MANIFEST")"
clair_manifest="${MANIFEST%.tsv}_clair_jobs.tsv"
printf 'clair_job\tfit_job\tcondition\tseed\tevaluation_jobs\ttrajectory_manifest\n' > "$clair_manifest"

for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        fit_id="${FIT_IDS["${seed}:${condition}"]}"
        mapfile -t evaluation_ids < <(
            awk -F '\t' -v fit="$fit_id" 'NR > 1 && $2 == fit { print $1 }' "$MANIFEST"
        )
        expected_count=8
        if [ "$condition" = no_loc ]; then
            expected_count=6
        fi
        if [ "${#evaluation_ids[@]}" -ne "$expected_count" ]; then
            echo "Expected $expected_count evaluation jobs for fit $fit_id, found ${#evaluation_ids[@]}."
            exit 1
        fi
        dependency="afterok"
        evaluation_list=""
        for evaluation_id in "${evaluation_ids[@]}"; do
            dependency="${dependency}:${evaluation_id}"
            if [ -n "$evaluation_list" ]; then
                evaluation_list="${evaluation_list},"
            fi
            evaluation_list="${evaluation_list}${evaluation_id}"
        done

        clair_output="$(
            sbatch \
                "--job-name=clair-fit-${fit_id}" \
                "--partition=${SLURM_DEFAULT_PARTITION}" \
                "--dependency=${dependency}" \
                "--export=ALL,CLAIR_MODEL_PATH=${CLAIR_MODEL_PATH},CLAIR_TRAJECTORY_MANIFEST=${manifest_absolute},CLAIR_FIT_JOB=${fit_id}" \
                scripts/score_clair_job.sbatch \
                --concurrency "$CLAIR_CONCURRENCY" \
                --max-new-tokens "$CLAIR_MAX_NEW_TOKENS"
        )"
        printf '%s\n' "$clair_output"
        clair_job="$(
            printf '%s\n' "$clair_output" \
                | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' \
                | tail -n 1
        )"
        if [ -z "$clair_job" ]; then
            echo "Could not extract CLAIR job ID for fit $fit_id."
            exit 1
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$clair_job" "$fit_id" "$condition" "$seed" \
            "$evaluation_list" "$manifest_absolute" >> "$clair_manifest"
    done
done

echo "Submitted eight fit-level CLAIR jobs."
echo "Wrote CLAIR submission manifest: $clair_manifest"
