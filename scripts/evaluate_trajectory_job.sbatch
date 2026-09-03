#!/bin/bash
#SBATCH --job-name=bentxt-trajectory-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
# Evaluate several saved adapters from one fit within one GPU allocation.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 PLAN_MANIFEST [evaluation overrides ...]"
    exit 1
fi

PLAN_MANIFEST="$1"
shift

cd "$SLURM_SUBMIT_DIR"
if [ ! -f .env ]; then
    echo "Missing .env."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

if [ -z "${EVALUATION_OUTPUT_ROOT:-}" ]; then
    echo "EVALUATION_OUTPUT_ROOT is required."
    exit 1
fi
if [ ! -f "$PLAN_MANIFEST" ]; then
    echo "Trajectory plan does not exist: $PLAN_MANIFEST"
    exit 1
fi
if ! [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "SLURM_JOB_ID must be available inside the allocation."
    exit 1
fi

MODEL_SIZE="${TRAJECTORY_MODEL_SIZE:-}"
case "$MODEL_SIZE" in
    2B) MODEL_NAME="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit" ;;
    4B) MODEL_NAME="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit" ;;
    8B) MODEL_NAME="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit" ;;
    *) echo "TRAJECTORY_MODEL_SIZE must be 2B, 4B or 8B."; exit 1 ;;
esac

OUTPUT_ID="${TRAJECTORY_OUTPUT_ID:-$SLURM_JOB_ID}"
if ! [[ "$OUTPUT_ID" =~ ^[0-9]+$ ]]; then
    echo "TRAJECTORY_OUTPUT_ID must be a Slurm job ID."
    exit 1
fi

EVALUATION_OUTPUT_ROOT="${EVALUATION_OUTPUT_ROOT%/}"
TRAJECTORY_ROOT="${EVALUATION_OUTPUT_ROOT}/trajectory_${OUTPUT_ID}"
RESOLVED_MANIFEST="${TRAJECTORY_ROOT}/trajectory_manifest.tsv"
EXPECTED_HEADER=$'evaluation_job\tfit_job\tcondition\tseed\tstep\tcoordinate_setting\tadapter_dir\trun_label'
IFS= read -r header < "$PLAN_MANIFEST"
if [ "$header" != "$EXPECTED_HEADER" ]; then
    echo "Unexpected trajectory manifest header: $header"
    exit 1
fi

evaluation_keys=()
fit_jobs=()
conditions=()
seeds=()
steps=()
settings=()
adapter_dirs=()
run_labels=()
declare -A seen_keys=()

while IFS=$'\t' read -r evaluation_key fit_job condition seed step setting adapter_dir run_label extra; do
    if [ -z "$evaluation_key" ]; then
        continue
    fi
    if [ -n "${extra:-}" ]; then
        echo "Trajectory row contains unexpected extra fields: $evaluation_key"
        exit 1
    fi
    case "$condition" in
        no_loc|loc_text|loc_embed|loc_encoding|loc_additive_satclip) ;;
        *) echo "Unsupported condition in trajectory plan: $condition"; exit 1 ;;
    esac
    case "$setting" in
        correct|shuffled) ;;
        *) echo "Unsupported coordinate setting in trajectory plan: $setting"; exit 1 ;;
    esac
    if [ "$condition" = no_loc ] && [ "$setting" != correct ]; then
        echo "no_loc cannot use coordinate setting $setting."
        exit 1
    fi
    if ! [[ "$fit_job" =~ ^[0-9]+$ ]] || ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "Fit job and seed must be non-negative integers: $evaluation_key"
        exit 1
    fi
    if [ "$step" != final ] && ! [[ "$step" =~ ^[1-9][0-9]*$ ]]; then
        echo "Step must be a positive integer or final: $step"
        exit 1
    fi
    if [[ "$evaluation_key" != SLURM_JOB_ID/entries/* ]]; then
        echo "Evaluation key must start with SLURM_JOB_ID/entries/: $evaluation_key"
        exit 1
    fi
    entry_id="${evaluation_key#SLURM_JOB_ID/entries/}"
    if ! [[ "$entry_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "Unsafe trajectory entry identifier: $entry_id"
        exit 1
    fi
    resolved_key="${OUTPUT_ID}/entries/${entry_id}"
    if [ -n "${seen_keys[$resolved_key]:-}" ]; then
        echo "Duplicate trajectory evaluation key: $resolved_key"
        exit 1
    fi
    seen_keys[$resolved_key]=1
    if [ ! -d "$adapter_dir" ]; then
        echo "Adapter directory is unavailable: $adapter_dir"
        exit 1
    fi

    evaluation_keys+=("$resolved_key")
    fit_jobs+=("$fit_job")
    conditions+=("$condition")
    seeds+=("$seed")
    steps+=("$step")
    settings+=("$setting")
    adapter_dirs+=("$adapter_dir")
    run_labels+=("$run_label")
done < <(tail -n +2 "$PLAN_MANIFEST")

entry_count="${#evaluation_keys[@]}"
if [ "$entry_count" -eq 0 ]; then
    echo "Trajectory plan contains no evaluations."
    exit 1
fi
for ((index = 1; index < entry_count; index++)); do
    if [ "${fit_jobs[$index]}" != "${fit_jobs[0]}" ] || \
       [ "${conditions[$index]}" != "${conditions[0]}" ] || \
       [ "${seeds[$index]}" != "${seeds[0]}" ]; then
        echo "One trajectory job may contain only one fit, condition and seed."
        exit 1
    fi
done

case "${conditions[0]}" in
    no_loc) CONDITION_CONFIG="" ;;
    loc_text) CONDITION_CONFIG="configs/finetuning/loc_text.yaml" ;;
    loc_embed) CONDITION_CONFIG="configs/finetuning/loc_embed.yaml" ;;
    loc_encoding) CONDITION_CONFIG="configs/finetuning/loc_encoding.yaml" ;;
    loc_additive_satclip) CONDITION_CONFIG="configs/finetuning/loc_additive_satclip.yaml" ;;
esac

mkdir -p "$TRAJECTORY_ROOT"
candidate_manifest="${TRAJECTORY_ROOT}/trajectory_manifest.candidate.${SLURM_JOB_ID}.tsv"
printf '%s\n' "$EXPECTED_HEADER" > "$candidate_manifest"
for ((index = 0; index < entry_count; index++)); do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${evaluation_keys[$index]}" "${fit_jobs[$index]}" \
        "${conditions[$index]}" "${seeds[$index]}" "${steps[$index]}" \
        "${settings[$index]}" "${adapter_dirs[$index]}" \
        "${run_labels[$index]}" >> "$candidate_manifest"
done
if [ -f "$RESOLVED_MANIFEST" ]; then
    if ! cmp -s "$candidate_manifest" "$RESOLVED_MANIFEST"; then
        echo "Resume manifest differs from the existing trajectory manifest."
        echo "Candidate retained for inspection: $candidate_manifest"
        exit 1
    fi
    rm -f "$candidate_manifest"
else
    mv "$candidate_manifest" "$RESOLVED_MANIFEST"
fi

ARGUMENTS_FILE="${TRAJECTORY_ROOT}/evaluation_arguments.txt"
candidate_arguments="${TRAJECTORY_ROOT}/evaluation_arguments.candidate.${SLURM_JOB_ID}.txt"
{
    printf 'model_size=%q\n' "$MODEL_SIZE"
    printf 'model_name=%q\n' "$MODEL_NAME"
    printf 'condition_config=%q\n' "$CONDITION_CONFIG"
    printf 'argument=%q\n' "$@"
} > "$candidate_arguments"
if [ -f "$ARGUMENTS_FILE" ]; then
    if ! cmp -s "$candidate_arguments" "$ARGUMENTS_FILE"; then
        echo "Resume arguments differ from the original trajectory evaluation."
        echo "Candidate retained for inspection: $candidate_arguments"
        exit 1
    fi
    rm -f "$candidate_arguments"
else
    mv "$candidate_arguments" "$ARGUMENTS_FILE"
fi

printf '%s\n' "$SLURM_JOB_ID" > "${TRAJECTORY_ROOT}/latest_slurm_job_id.txt"
printf '%s\n' "$MODEL_SIZE" > "${TRAJECTORY_ROOT}/model_size.txt"
printf '%s\n' "$PLAN_MANIFEST" > "${TRAJECTORY_ROOT}/plan_manifest_path.txt"

echo "Trajectory fit: ${fit_jobs[0]}"
echo "Condition: ${conditions[0]}"
echo "Seed: ${seeds[0]}"
echo "Model size: $MODEL_SIZE"
echo "Entries: $entry_count"
echo "Output root: $TRAJECTORY_ROOT"
echo "Resolved manifest: $RESOLVED_MANIFEST"

completed=0
for ((index = 0; index < entry_count; index++)); do
    evaluation_key="${evaluation_keys[$index]}"
    entry_id="${evaluation_key##*/}"
    output_dir="${TRAJECTORY_ROOT}/entries/${entry_id}"
    completion_marker="${output_dir}/.trajectory_complete"
    manifest_line=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
        "$evaluation_key" "${fit_jobs[$index]}" "${conditions[$index]}" \
        "${seeds[$index]}" "${steps[$index]}" "${settings[$index]}" \
        "${adapter_dirs[$index]}" "${run_labels[$index]}")
    entry_hash=$(printf '%s' "$manifest_line" | sha256sum | awk '{print $1}')

    if [ -f "$completion_marker" ] && \
       [ "$(cat "$completion_marker")" = "$entry_hash" ] && \
       [ -s "${output_dir}/predictions.jsonl" ] && \
       [ -s "${output_dir}/scored_predictions/summary.json" ]; then
        echo "Skipping completed entry $((index + 1))/$entry_count: $entry_id"
        completed=$((completed + 1))
        continue
    fi

    if [ -e "$output_dir" ]; then
        archive_root="${TRAJECTORY_ROOT}/incomplete_attempts"
        mkdir -p "$archive_root"
        archive_dir="${archive_root}/${entry_id}_slurm${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
        echo "Preserving incomplete entry at: $archive_dir"
        mv "$output_dir" "$archive_dir"
    fi
    mkdir -p "$output_dir"

    evaluation_args=(
        --model.init_args.model_name_or_path "$MODEL_NAME"
        "$@"
    )
    if [ "${settings[$index]}" = shuffled ]; then
        evaluation_args+=(--data.init_args.coordinate_perturbation shuffled)
    fi

    echo "Starting entry $((index + 1))/$entry_count: $entry_id"
    env \
        CONDITION_CONFIG="$CONDITION_CONFIG" \
        EVAL_ADAPTER_DIR="${adapter_dirs[$index]}" \
        EVAL_OUTPUT_DIR="$output_dir" \
        RUN_LABEL="${run_labels[$index]}" \
        MODEL_SIZE="$MODEL_SIZE" \
        bash scripts/evaluate_job.sbatch "${evaluation_args[@]}"

    if [ ! -s "${output_dir}/predictions.jsonl" ] || \
       [ ! -s "${output_dir}/scored_predictions/summary.json" ]; then
        echo "Entry did not produce the required prediction and score artifacts: $entry_id"
        exit 1
    fi
    marker_tmp="${completion_marker}.tmp.${SLURM_JOB_ID}"
    printf '%s\n' "$entry_hash" > "$marker_tmp"
    mv "$marker_tmp" "$completion_marker"
    completed=$((completed + 1))
    echo "Completed entry $completed/$entry_count: $entry_id"
done

if [ "$completed" -ne "$entry_count" ]; then
    echo "Internal completion-count mismatch: $completed of $entry_count"
    exit 1
fi
printf 'fit_job\tcondition\tseed\tmodel_size\tentry_count\n%s\t%s\t%s\t%s\t%s\n' \
    "${fit_jobs[0]}" "${conditions[0]}" "${seeds[0]}" "$MODEL_SIZE" \
    "$entry_count" > "${TRAJECTORY_ROOT}/trajectory_complete.tsv"
echo "Trajectory evaluation completed: $entry_count/$entry_count entries."
