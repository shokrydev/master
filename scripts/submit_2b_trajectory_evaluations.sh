#!/bin/bash
# Submit only the prespecified evaluations for the corrected 2B trajectory
# fits. The all-in-one fit+evaluation helper remains unchanged.

set -euo pipefail

# Locked by task-aware profiler job 11836. Command-line flags remain available
# for explicit diagnostic overrides.
SHORT_BATCH="256"
BBOX_BATCH="512"
CAPTION_BATCH="384"
SHORT_WORKERS="8"
BBOX_WORKERS="8"
CAPTION_WORKERS="8"
FIT_JOB=""
MANIFEST=""
SUBMIT_CLAIR=false
CLAIR_MODEL_NAME_OR_PATH="${CLAIR_MODEL_NAME_OR_PATH:-unsloth/Qwen3.8-27B-unsloth-bnb-4bit}"
CLAIR_BATCH_SIZE=64
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
        --fit-job) FIT_JOB="${2:-}"; shift 2 ;;
        --manifest) MANIFEST="${2:-}"; shift 2 ;;
        --submit-clair) SUBMIT_CLAIR=true; shift ;;
        --clair-model) CLAIR_MODEL_NAME_OR_PATH="${2:-}"; shift 2 ;;
        --clair-batch-size) CLAIR_BATCH_SIZE="${2:-}"; shift 2 ;;
        --clair-max-new-tokens) CLAIR_MAX_NEW_TOKENS="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if ! [[ "$FIT_JOB" =~ ^[0-9]+$ ]]; then
    echo "--fit-job must identify one corrected 2B fit job."
    exit 1
fi

values=("$SHORT_BATCH" "$BBOX_BATCH" "$CAPTION_BATCH" "$SHORT_WORKERS" "$BBOX_WORKERS" "$CAPTION_WORKERS")
for value in "${values[@]}"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "All three batch sizes and worker counts must be positive integers from profiler 11836."
        exit 1
    fi
done
if ! [[ "$CLAIR_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || ! [[ "$CLAIR_MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CLAIR batch size and max-new-tokens must be positive integers."
    exit 1
fi
if [ "$SUBMIT_CLAIR" = true ] && [ -z "$CLAIR_MODEL_NAME_OR_PATH" ]; then
    echo "--submit-clair requires a CLAIR model name or path."
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
CONDITIONS=(no_loc loc_text loc_embed loc_additive_satclip)
SEEDS=(42 43)
CORRECT_STEPS=(50 100 500 1000 5000 final)
SHUFFLED_STEPS=(1000 final)
declare -A FIT_IDS=(
    [42:no_loc]=11881
    [42:loc_text]=11882
    [42:loc_embed]=11809
    [42:loc_additive_satclip]=11810
    [43:no_loc]=11811
    [43:loc_text]=11812
    [43:loc_embed]=11813
    [43:loc_additive_satclip]=11814
)

SELECTED_SEED=""
SELECTED_CONDITION=""
for seed in "${SEEDS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        if [ "${FIT_IDS["${seed}:${condition}"]}" = "$FIT_JOB" ]; then
            SELECTED_SEED="$seed"
            SELECTED_CONDITION="$condition"
        fi
    done
done
if [ -z "$SELECTED_SEED" ]; then
    echo "Unknown corrected 2B fit job: $FIT_JOB"
    echo "Expected one of: 11809 11810 11811 11812 11813 11814 11881 11882"
    exit 1
fi
if [ -z "$MANIFEST" ]; then
    MANIFEST="outputs/submission_manifests/2b_trajectory_fit_${FIT_JOB}_$(date +%Y%m%d_%H%M%S).tsv"
fi

FIT_DEPENDENCY="afterok:${FIT_JOB}"
if [ "$DRY_RUN" = false ]; then
    if ! command -v squeue >/dev/null 2>&1 || ! command -v sacct >/dev/null 2>&1; then
        echo "squeue and sacct are required to verify fit state before submission."
        exit 1
    fi
    active_state="$(squeue -h -j "$FIT_JOB" -o '%T' | awk 'NF { print $1; exit }')"
    if [ -n "$active_state" ]; then
        echo "Fit $FIT_JOB is $active_state; evaluations will depend on successful completion."
    else
        accounting_state="$(
            sacct -X -n -P -j "$FIT_JOB" --format=JobIDRaw,State \
                | awk -F '|' -v job="$FIT_JOB" '$1 == job { sub(/\+$/, "", $2); print $2; exit }'
        )"
        if [ "$accounting_state" = "COMPLETED" ]; then
            FIT_DEPENDENCY=""
            echo "Fit $FIT_JOB is COMPLETED; submitting evaluations without a stale dependency."
        elif [ -z "$accounting_state" ]; then
            echo "Could not verify fit $FIT_JOB in squeue or sacct; refusing submission."
            exit 1
        else
            echo "Fit $FIT_JOB is $accounting_state, not COMPLETED; refusing submission."
            exit 1
        fi
    fi
fi

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
    local dependency_args=()
    if [ -n "$FIT_DEPENDENCY" ]; then
        dependency_args=(--dependency "$FIT_DEPENDENCY")
    fi
    output="$(
        "$SCRIPT_DIR/submit_evaluation_job.sh" \
            --condition "$condition" --size 2B \
            --adapter-dir "$adapter_dir" \
            --run-label "$run_label" \
            "${dependency_args[@]}" \
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

fit_id="$FIT_JOB"
seed="$SELECTED_SEED"
condition="$SELECTED_CONDITION"
prefix="qwen3-json-${condition}-2B-full-seed${seed}"
for step in "${CORRECT_STEPS[@]}"; do
    adapter_dir="$(adapter_dir_for_step "$fit_id" "$step")"
    run_label="${prefix}-step${step}-j${fit_id}"
    submit_evaluation \
        "$fit_id" "$condition" "$seed" "$step" correct \
        "$adapter_dir" "$run_label" \
        --name "eval-${fit_id}-${condition}-${step}"
done
if [ "$condition" != no_loc ]; then
    for step in "${SHUFFLED_STEPS[@]}"; do
        adapter_dir="$(adapter_dir_for_step "$fit_id" "$step")"
        run_label="${prefix}-step${step}-shuffled-j${fit_id}"
        submit_evaluation \
            "$fit_id" "$condition" "$seed" "$step" shuffled \
            "$adapter_dir" "$run_label" \
            --name "eval-${fit_id}-${condition}-${step}-shuf" \
            --coordinate-perturbation shuffled
    done
fi

expected_count=8
if [ "$condition" = no_loc ]; then
    expected_count=6
fi
echo "Submitted $expected_count trajectory evaluation jobs for fit $fit_id ($condition, seed $seed)."
if [ "$DRY_RUN" = false ]; then
    echo "Wrote submission manifest: $MANIFEST"
fi

if [ "$SUBMIT_CLAIR" = false ]; then
    exit 0
fi
if [ "$DRY_RUN" = true ]; then
    echo "[Dry run] A real submission would add one CLAIR job for fit $fit_id."
    echo "It would depend on this fit's $expected_count evaluation jobs and load the judge once."
    exit 0
fi

if [ -z "${SLURM_DEFAULT_PARTITION:-}" ]; then
    echo "SLURM_DEFAULT_PARTITION is required for fit-level CLAIR submissions."
    exit 1
fi
manifest_absolute="$(realpath "$MANIFEST")"
clair_manifest="${MANIFEST%.tsv}_clair_jobs.tsv"
printf 'clair_job\tfit_job\tcondition\tseed\tevaluation_jobs\ttrajectory_manifest\n' > "$clair_manifest"

mapfile -t evaluation_ids < <(
    awk -F '\t' -v fit="$fit_id" 'NR > 1 && $2 == fit { print $1 }' "$MANIFEST"
)
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
        "--export=ALL,CLAIR_MODEL_NAME_OR_PATH=${CLAIR_MODEL_NAME_OR_PATH},CLAIR_TRAJECTORY_MANIFEST=${manifest_absolute},CLAIR_FIT_JOB=${fit_id}" \
        scripts/score_clair_job.sbatch \
        --batch-size "$CLAIR_BATCH_SIZE" \
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

echo "Submitted one CLAIR job for fit $fit_id."
echo "Wrote CLAIR submission manifest: $clair_manifest"
