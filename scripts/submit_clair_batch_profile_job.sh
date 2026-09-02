#!/bin/bash
set -euo pipefail

PREDICTIONS=""
PARTITION=""
DEPENDENCY=""
DRY_RUN=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictions) PREDICTIONS="${2:-}"; shift 2 ;;
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
if [ -z "$PREDICTIONS" ] || [ -z "$PARTITION" ]; then
    echo "Usage: $0 --predictions /path/predictions.jsonl [--dependency SPEC] [--dry-run] [profiler args...]"
    exit 1
fi
if [ -z "$DEPENDENCY" ] && [ ! -f "$PREDICTIONS" ]; then
    echo "Predictions file does not exist: $PREDICTIONS"
    exit 1
fi

cmd=(sbatch --job-name=profile-clair-batches "--partition=$PARTITION")
if [ -n "$DEPENDENCY" ]; then
    cmd+=("--dependency=$DEPENDENCY")
fi
cmd+=(scripts/profile_clair_batch_size.sbatch --predictions "$PREDICTIONS")
cmd+=("${EXTRA_ARGS[@]}")
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [ "$DRY_RUN" = true ]; then
    echo "[Dry run - not submitting]"
else
    "${cmd[@]}"
fi
