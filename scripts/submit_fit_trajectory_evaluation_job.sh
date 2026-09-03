#!/bin/bash
# Submit one resumable Slurm evaluation job for several adapters from one fit.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/submit_fit_trajectory_evaluation_job.sh \
    --fit-job 11809 --condition loc_embed --size 2B --seed 42 \
    --correct-step 250 --correct-step 2500 \
    --correct-step 10000 --correct-step 20000

  ./scripts/submit_fit_trajectory_evaluation_job.sh \
    --fit-job 12345 --condition loc_text --size 4B --seed 42 \
    --dependency afterok:12345 \
    --correct-step 50 --correct-step 100 --correct-step 500 \
    --correct-step 1000 --correct-step 5000 --correct-step final \
    --shuffled-step 1000 --shuffled-step final --submit-clair

Resume an interrupted packed evaluation in its original output directory:
  ./scripts/submit_fit_trajectory_evaluation_job.sh \
    --fit-job 11809 --condition loc_embed --size 2B --seed 42 \
    --resume-job TRAJECTORY_JOB --manifest PATH_TO_ORIGINAL_PLAN

Repeat any original --config flags on resume. The helper verifies them against
the plan's argument sidecar before submitting.

Options:
  --fit-job ID
  --condition NAME
  --size 2B|4B|8B
  --seed INTEGER
  --correct-step STEP       Repeat for each factual checkpoint
  --shuffled-step STEP      Repeat for each shuffled-coordinate checkpoint
  --config PATH             Repeat for narrow evaluation-compatible overrides
  --short-batch N           Required for 4B/8B; defaults to profiled 2B value
  --bbox-batch N            Required for 4B/8B; defaults to profiled 2B value
  --caption-batch N         Required for 4B/8B; defaults to profiled 2B value
  --short-workers N         Default: 8
  --bbox-workers N          Default: 8
  --caption-workers N       Default: 8
  --manifest PATH           Plan path; required with --resume-job
  --dependency SPEC         Slurm dependency, e.g. afterok:12345
  --resume-job ID           Reuse the output directory and completion markers
  --submit-clair            Submit one dependent fit-level CLAIR job
  --dry-run                 Print the plan and commands without writing/submitting
EOF
}

FIT_JOB=""
CONDITION=""
SIZE=""
SEED=""
MANIFEST=""
DEPENDENCY=""
RESUME_JOB=""
PARTITION=""
TIME_LIMIT=""
MEMORY=""
CPUS=""
RUN_PREFIX=""
SHORT_BATCH=""
BBOX_BATCH=""
CAPTION_BATCH=""
SHORT_WORKERS=8
BBOX_WORKERS=8
CAPTION_WORKERS=8
SUBMIT_CLAIR=false
DRY_RUN=false
CORRECT_STEPS=()
SHUFFLED_STEPS=()
EVALUATION_CONFIGS=()
CLAIR_MODEL_NAME_OR_PATH="${CLAIR_MODEL_NAME_OR_PATH:-unsloth/Qwen3.8-27B-unsloth-bnb-4bit}"
CLAIR_BATCH_SIZE=64
CLAIR_MAX_NEW_TOKENS=512

require_arg() {
    if [ -z "${2:-}" ]; then
        echo "Missing value for $1"
        exit 1
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --fit-job) require_arg "$1" "${2:-}"; FIT_JOB="$2"; shift 2 ;;
        --condition) require_arg "$1" "${2:-}"; CONDITION="$2"; shift 2 ;;
        --size) require_arg "$1" "${2:-}"; SIZE="$2"; shift 2 ;;
        --seed) require_arg "$1" "${2:-}"; SEED="$2"; shift 2 ;;
        --correct-step) require_arg "$1" "${2:-}"; CORRECT_STEPS+=("$2"); shift 2 ;;
        --shuffled-step) require_arg "$1" "${2:-}"; SHUFFLED_STEPS+=("$2"); shift 2 ;;
        --config) require_arg "$1" "${2:-}"; EVALUATION_CONFIGS+=("$2"); shift 2 ;;
        --short-batch) require_arg "$1" "${2:-}"; SHORT_BATCH="$2"; shift 2 ;;
        --bbox-batch) require_arg "$1" "${2:-}"; BBOX_BATCH="$2"; shift 2 ;;
        --caption-batch) require_arg "$1" "${2:-}"; CAPTION_BATCH="$2"; shift 2 ;;
        --short-workers) require_arg "$1" "${2:-}"; SHORT_WORKERS="$2"; shift 2 ;;
        --bbox-workers) require_arg "$1" "${2:-}"; BBOX_WORKERS="$2"; shift 2 ;;
        --caption-workers) require_arg "$1" "${2:-}"; CAPTION_WORKERS="$2"; shift 2 ;;
        --manifest) require_arg "$1" "${2:-}"; MANIFEST="$2"; shift 2 ;;
        --dependency) require_arg "$1" "${2:-}"; DEPENDENCY="$2"; shift 2 ;;
        --resume-job) require_arg "$1" "${2:-}"; RESUME_JOB="$2"; shift 2 ;;
        --partition) require_arg "$1" "${2:-}"; PARTITION="$2"; shift 2 ;;
        --time) require_arg "$1" "${2:-}"; TIME_LIMIT="$2"; shift 2 ;;
        --mem) require_arg "$1" "${2:-}"; MEMORY="$2"; shift 2 ;;
        --cpus) require_arg "$1" "${2:-}"; CPUS="$2"; shift 2 ;;
        --run-prefix) require_arg "$1" "${2:-}"; RUN_PREFIX="$2"; shift 2 ;;
        --submit-clair) SUBMIT_CLAIR=true; shift ;;
        --clair-model) require_arg "$1" "${2:-}"; CLAIR_MODEL_NAME_OR_PATH="$2"; shift 2 ;;
        --clair-batch-size) require_arg "$1" "${2:-}"; CLAIR_BATCH_SIZE="$2"; shift 2 ;;
        --clair-max-new-tokens) require_arg "$1" "${2:-}"; CLAIR_MAX_NEW_TOKENS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if ! [[ "$FIT_JOB" =~ ^[0-9]+$ ]]; then
    echo "--fit-job must be a Slurm job ID."
    exit 1
fi
case "$CONDITION" in
    no_loc|loc_text|loc_embed|loc_encoding|loc_additive_satclip) ;;
    *) echo "Unsupported --condition: $CONDITION"; exit 1 ;;
esac
case "$SIZE" in
    2B|4B|8B) ;;
    *) echo "--size must be 2B, 4B or 8B."; exit 1 ;;
esac
if [ "$SIZE" = 2B ]; then
    SHORT_BATCH="${SHORT_BATCH:-256}"
    BBOX_BATCH="${BBOX_BATCH:-512}"
    CAPTION_BATCH="${CAPTION_BATCH:-384}"
elif [ -z "$SHORT_BATCH" ] || [ -z "$BBOX_BATCH" ] || [ -z "$CAPTION_BATCH" ]; then
    echo "4B/8B packed evaluation requires profiler-selected --short-batch, --bbox-batch and --caption-batch values."
    echo "The 2B values are intentionally not reused for a larger model."
    exit 1
fi
for value in \
    "$SHORT_BATCH" "$BBOX_BATCH" "$CAPTION_BATCH" \
    "$SHORT_WORKERS" "$BBOX_WORKERS" "$CAPTION_WORKERS"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Batch sizes and worker counts must be positive integers."
        exit 1
    fi
done
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a non-negative integer."
    exit 1
fi
if [ "$CONDITION" = no_loc ] && [ "${#SHUFFLED_STEPS[@]}" -gt 0 ]; then
    echo "Shuffled-coordinate evaluation is not meaningful for no_loc."
    exit 1
fi
if ! [[ "$CLAIR_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "$CLAIR_MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CLAIR batch size and max-new-tokens must be positive integers."
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
if [ -z "${FINETUNING_OUTPUT_ROOT:-}" ] || [ -z "${EVALUATION_OUTPUT_ROOT:-}" ]; then
    echo "FINETUNING_OUTPUT_ROOT and EVALUATION_OUTPUT_ROOT are required."
    exit 1
fi
PARTITION="${PARTITION:-${SLURM_DEFAULT_PARTITION:-}}"
if [ -z "$PARTITION" ]; then
    echo "Set SLURM_DEFAULT_PARTITION or pass --partition."
    exit 1
fi
for config in "${EVALUATION_CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "Evaluation config is not a file: $config"
        exit 1
    fi
done

validate_step() {
    local step="$1"
    if [ "$step" != final ] && ! [[ "$step" =~ ^[1-9][0-9]*$ ]]; then
        echo "Checkpoint step must be a positive integer or final: $step"
        exit 1
    fi
}
for step in "${CORRECT_STEPS[@]}" "${SHUFFLED_STEPS[@]}"; do
    if [ -n "$step" ]; then
        validate_step "$step"
    fi
done

if [ -n "$RESUME_JOB" ]; then
    if ! [[ "$RESUME_JOB" =~ ^[0-9]+$ ]]; then
        echo "--resume-job must be the original packed trajectory Slurm job ID."
        exit 1
    fi
    if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
        echo "--resume-job requires the original existing --manifest."
        exit 1
    fi
    if [ "${#CORRECT_STEPS[@]}" -gt 0 ] || [ "${#SHUFFLED_STEPS[@]}" -gt 0 ]; then
        echo "Do not pass checkpoint steps when resuming; the original manifest is authoritative."
        exit 1
    fi
else
    if [ "${#CORRECT_STEPS[@]}" -eq 0 ] && [ "${#SHUFFLED_STEPS[@]}" -eq 0 ]; then
        echo "Provide at least one --correct-step or --shuffled-step."
        exit 1
    fi
    if [ -z "$MANIFEST" ]; then
        MANIFEST="outputs/submission_manifests/packed_trajectory_fit_${FIT_JOB}_$(date +%Y%m%d_%H%M%S).tsv"
    elif [ -e "$MANIFEST" ]; then
        echo "Refusing to overwrite existing trajectory plan: $MANIFEST"
        exit 1
    fi
fi

adapter_dir_for_step() {
    local step="$1"
    local fit_root="${FINETUNING_OUTPUT_ROOT%/}/bigearthnet_${FIT_JOB}"
    if [ "$step" = final ]; then
        printf '%s/qlora_adapter' "$fit_root"
    else
        printf '%s/qlora_adapter_steps/step_%06d' "$fit_root" "$step"
    fi
}

entry_id_for() {
    local step="$1"
    local setting="$2"
    if [ "$step" = final ]; then
        printf 'final_%s' "$setting"
    else
        printf 'step_%06d_%s' "$step" "$setting"
    fi
}

if [ -z "$RESUME_JOB" ]; then
    declare -A seen_entries=()
    plan_lines=()
    prefix="${RUN_PREFIX:-qwen3-json-${CONDITION}-${SIZE}-full-seed${SEED}}"
    for setting in correct shuffled; do
        if [ "$setting" = correct ]; then
            selected_steps=("${CORRECT_STEPS[@]}")
        else
            selected_steps=("${SHUFFLED_STEPS[@]}")
        fi
        for step in "${selected_steps[@]}"; do
            entry_id="$(entry_id_for "$step" "$setting")"
            if [ -n "${seen_entries[$entry_id]:-}" ]; then
                echo "Duplicate trajectory entry: $entry_id"
                exit 1
            fi
            seen_entries[$entry_id]=1
            adapter_dir="$(adapter_dir_for_step "$step")"
            if [ "$DRY_RUN" = false ] && [ -z "$DEPENDENCY" ] && [ ! -d "$adapter_dir" ]; then
                echo "Adapter directory is unavailable: $adapter_dir"
                echo "No job was submitted. Use --dependency for an unfinished fit."
                exit 1
            fi
            run_label="${prefix}-step${step}"
            if [ "$setting" = shuffled ]; then
                run_label="${run_label}-shuffled"
            fi
            plan_lines+=("SLURM_JOB_ID/entries/${entry_id}"$'\t'"${FIT_JOB}"$'\t'"${CONDITION}"$'\t'"${SEED}"$'\t'"${step}"$'\t'"${setting}"$'\t'"${adapter_dir}"$'\t'"${run_label}")
        done
    done
    expected_count="${#plan_lines[@]}"
else
    expected_count=$(awk -F '\t' 'NR > 1 && NF { count++ } END { print count + 0 }' "$MANIFEST")
    if [ "$expected_count" -eq 0 ]; then
        echo "Resume manifest contains no entries: $MANIFEST"
        exit 1
    fi
    manifest_identity=$(awk -F '\t' '
        NR == 1 {
            expected = "evaluation_job\tfit_job\tcondition\tseed\tstep\tcoordinate_setting\tadapter_dir\trun_label"
            if ($0 != expected) exit 2
            next
        }
        NF {
            identity = $2 "\t" $3 "\t" $4
            if (!first) first = identity
            if (identity != first) exit 3
        }
        END { if (first) print first }
    ' "$MANIFEST") || {
        echo "Resume manifest is malformed or mixes multiple fits."
        exit 1
    }
    expected_identity="${FIT_JOB}"$'\t'"${CONDITION}"$'\t'"${SEED}"
    if [ "$manifest_identity" != "$expected_identity" ]; then
        echo "Resume arguments do not match the manifest fit, condition and seed."
        echo "Manifest: $manifest_identity"
        echo "Arguments: $expected_identity"
        exit 1
    fi
fi

COMMON_OVERRIDES=(
    --data.init_args.evaluation_batch_sizes.short_answer "$SHORT_BATCH"
    --data.init_args.evaluation_batch_sizes.bounding_box "$BBOX_BATCH"
    --data.init_args.evaluation_batch_sizes.captioning "$CAPTION_BATCH"
    --data.init_args.evaluation_num_workers_by_bucket.short_answer "$SHORT_WORKERS"
    --data.init_args.evaluation_num_workers_by_bucket.bounding_box "$BBOX_WORKERS"
    --data.init_args.evaluation_num_workers_by_bucket.captioning "$CAPTION_WORKERS"
)
for config in "${EVALUATION_CONFIGS[@]}"; do
    COMMON_OVERRIDES+=(--config "$config")
done

arguments_plan="${MANIFEST%.tsv}_arguments.txt"
arguments_text="$({
    printf 'model_size=%q\n' "$SIZE"
    printf 'argument=%q\n' "${COMMON_OVERRIDES[@]}"
})"
if [ -n "$RESUME_JOB" ]; then
    if [ ! -f "$arguments_plan" ]; then
        echo "Resume argument plan is missing: $arguments_plan"
        exit 1
    fi
    if [ "$(cat "$arguments_plan")" != "$arguments_text" ]; then
        echo "Resume evaluation arguments differ from the original submission."
        echo "Repeat the original --config flags exactly."
        exit 1
    fi
fi

if [ "$DRY_RUN" = false ] && [ -z "$RESUME_JOB" ]; then
    mkdir -p "$(dirname "$MANIFEST")"
    printf 'evaluation_job\tfit_job\tcondition\tseed\tstep\tcoordinate_setting\tadapter_dir\trun_label\n' > "$MANIFEST"
    printf '%s\n' "${plan_lines[@]}" >> "$MANIFEST"
    printf '%s\n' "$arguments_text" > "$arguments_plan"
fi
manifest_absolute="$(realpath -m "$MANIFEST")"

trajectory_cmd=(
    sbatch --parsable
    "--job-name=trajectory-${FIT_JOB}-${CONDITION}"
    "--partition=$PARTITION"
)
if [ -n "$TIME_LIMIT" ]; then trajectory_cmd+=("--time=$TIME_LIMIT"); fi
if [ -n "$MEMORY" ]; then trajectory_cmd+=("--mem=$MEMORY"); fi
if [ -n "$CPUS" ]; then trajectory_cmd+=("--cpus-per-task=$CPUS"); fi
if [ -n "$DEPENDENCY" ]; then trajectory_cmd+=("--dependency=$DEPENDENCY"); fi
export_values="ALL,TRAJECTORY_MODEL_SIZE=${SIZE}"
if [ -n "$RESUME_JOB" ]; then
    export_values="${export_values},TRAJECTORY_OUTPUT_ID=${RESUME_JOB}"
fi
trajectory_cmd+=("--export=$export_values" scripts/evaluate_trajectory_job.sbatch "$manifest_absolute")
trajectory_cmd+=("${COMMON_OVERRIDES[@]}")

echo "Packed trajectory evaluation"
echo "Fit: $FIT_JOB"
echo "Condition: $CONDITION"
echo "Size: $SIZE"
echo "Seed: $SEED"
echo "Entries: $expected_count"
echo "Plan: $manifest_absolute"
echo "Resume output: ${RESUME_JOB:-<new trajectory>}"
printf 'Command:'
printf ' %q' "${trajectory_cmd[@]}"
printf '\n'
if [ "$DRY_RUN" = true ]; then
    if [ -z "$RESUME_JOB" ]; then
        printf 'Planned entries:\n'
        printf '  %s\n' "${plan_lines[@]}"
    fi
    echo "[Dry run - not writing or submitting]"
    exit 0
fi

trajectory_job="$("${trajectory_cmd[@]}")"
trajectory_job="${trajectory_job%%;*}"
if ! [[ "$trajectory_job" =~ ^[0-9]+$ ]]; then
    echo "Could not extract packed trajectory job ID: $trajectory_job"
    exit 1
fi
echo "Submitted packed trajectory job $trajectory_job."

output_id="${RESUME_JOB:-$trajectory_job}"
resolved_manifest="${EVALUATION_OUTPUT_ROOT%/}/trajectory_${output_id}/trajectory_manifest.tsv"
clair_job=""
if [ "$SUBMIT_CLAIR" = true ]; then
    if [ -z "$CLAIR_MODEL_NAME_OR_PATH" ]; then
        echo "--submit-clair requires a judge model."
        exit 1
    fi
    clair_output="$(
        sbatch \
            "--job-name=clair-fit-${FIT_JOB}" \
            "--partition=${PARTITION}" \
            "--dependency=afterok:${trajectory_job}" \
            "--export=ALL,CLAIR_MODEL_NAME_OR_PATH=${CLAIR_MODEL_NAME_OR_PATH},CLAIR_TRAJECTORY_MANIFEST=${resolved_manifest},CLAIR_FIT_JOB=${FIT_JOB},CLAIR_EXPECTED_EXPORTS=${expected_count}" \
            scripts/score_clair_job.sbatch \
            --batch-size "$CLAIR_BATCH_SIZE" \
            --max-new-tokens "$CLAIR_MAX_NEW_TOKENS"
    )"
    printf '%s\n' "$clair_output"
    clair_job="$(printf '%s\n' "$clair_output" | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' | tail -n 1)"
    if [ -z "$clair_job" ]; then
        echo "Could not extract dependent CLAIR job ID."
        exit 1
    fi
fi

jobs_manifest="${MANIFEST%.tsv}_jobs.tsv"
printf 'trajectory_job\toutput_id\tfit_job\tcondition\tsize\tseed\tentry_count\tclair_job\tplan_manifest\tresolved_manifest\n' > "$jobs_manifest"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$trajectory_job" "$output_id" "$FIT_JOB" "$CONDITION" "$SIZE" "$SEED" \
    "$expected_count" "$clair_job" "$manifest_absolute" "$resolved_manifest" >> "$jobs_manifest"
echo "Wrote packed-job manifest: $jobs_manifest"
