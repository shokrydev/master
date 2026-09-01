#!/bin/bash
# Submit offline CLAIR scoring for one existing predictions.jsonl export.

set -euo pipefail

PREDICTIONS=""
MODEL_NAME_OR_PATH="${CLAIR_MODEL_NAME_OR_PATH:-unsloth/Qwen3.8-27B-unsloth-bnb-4bit}"
JOB_NAME="bentxt-clair"
PARTITION=""
DEPENDENCY=""
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictions) PREDICTIONS="${2:-}"; shift 2 ;;
        --model) MODEL_NAME_OR_PATH="${2:-}"; shift 2 ;;
        --name) JOB_NAME="${2:-}"; shift 2 ;;
        --partition) PARTITION="${2:-}"; shift 2 ;;
        --dependency) DEPENDENCY="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -f .env ]; then
    echo "Missing .env."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
PARTITION="${PARTITION:-${SLURM_DEFAULT_PARTITION:-}}"
if [ -z "$PREDICTIONS" ] || [ -z "$MODEL_NAME_OR_PATH" ] || [ -z "$PARTITION" ]; then
    echo "Usage: $0 --predictions /path/predictions.jsonl [--model HF_ID_OR_DIR] [--name NAME] [--dependency SPEC] [--dry-run] [scorer args...]"
    exit 1
fi
if [ -z "$DEPENDENCY" ] && [ ! -f "$PREDICTIONS" ]; then
    echo "Predictions file does not exist: $PREDICTIONS"
    exit 1
fi
cmd=(sbatch "--job-name=$JOB_NAME" "--partition=$PARTITION")
if [ -n "$DEPENDENCY" ]; then
    cmd+=("--dependency=$DEPENDENCY")
fi
cmd+=("--export=ALL,CLAIR_MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH")
cmd+=(scripts/score_clair_job.sbatch "$PREDICTIONS" "${EXTRA_ARGS[@]}")
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [ "$DRY_RUN" = true ]; then
    echo "[Dry run - not submitting]"
else
    "${cmd[@]}"
fi
