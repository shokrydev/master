#!/bin/bash
# ============================================================================
# BigEarthNet.txt Batch-Size Profiling Submission Helper
# ============================================================================
# Usage:
#   ./scripts/submit_profile_job.sh --dry-run
#   ./scripts/submit_profile_job.sh --size 8B --condition loc_embed
#   ./scripts/submit_profile_job.sh --partition big_job --time 1-00:00:00 --size 8B
# ============================================================================

set -e

if [ ! -f .env ]; then
    echo "Missing .env. Copy .env.example to .env and fill in the server-local paths."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

SIZE="8B"
CONDITION="loc_embed"
JOB_NAME=""
PARTITION="${SLURM_DEFAULT_PARTITION:-}"
TIME_LIMIT=""
DRY_RUN=false
PROFILE_ARGS=()

require_arg() {
    if [ -z "${2:-}" ]; then
        echo "Missing value for $1"
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            require_arg "$1" "${2:-}"
            SIZE="$2"
            PROFILE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --condition)
            require_arg "$1" "${2:-}"
            CONDITION="$2"
            PROFILE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --name)
            require_arg "$1" "${2:-}"
            JOB_NAME="$2"
            shift 2
            ;;
        --partition)
            require_arg "$1" "${2:-}"
            PARTITION="$2"
            shift 2
            ;;
        --time)
            require_arg "$1" "${2:-}"
            TIME_LIMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            PROFILE_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$SIZE" in
    2B|4B|8B)
        ;;
    *)
        echo "Invalid --size '$SIZE'. Use 2B, 4B or 8B."
        exit 1
        ;;
esac

case "$CONDITION" in
    no_loc|loc_text|loc_embed)
        ;;
    *)
        echo "Invalid --condition '$CONDITION'. Use no_loc, loc_text or loc_embed."
        exit 1
        ;;
esac

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="profile-${CONDITION}-${SIZE}"
fi

if [ -z "$PARTITION" ]; then
    echo "Missing Slurm partition. Set SLURM_DEFAULT_PARTITION in .env or pass --partition."
    exit 1
fi

REQUIRED_ENV_VARS=(
    BIGEARTHNET_V2_LMDB_ROOT
    BIGEARTHNET_TXT_PARQUET_PATH
    BIGEARTHNET_ENCODER_DIR
    HF_HOME
)
if [ "$CONDITION" = "loc_embed" ]; then
    REQUIRED_ENV_VARS+=(SATCLIP_CHECKPOINT_PATH)
fi

MISSING_ENV_VARS=()
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    if [ -z "${!VAR_NAME:-}" ]; then
        MISSING_ENV_VARS+=("$VAR_NAME")
    fi
done

mkdir -p logs outputs/batch_profiles

echo "=============================================="
echo "BigEarthNet.txt Batch-Size Profiling Submission"
echo "=============================================="
echo "Size: $SIZE"
echo "Condition: $CONDITION"
echo "Job name: $JOB_NAME"
echo "Slurm partition: $PARTITION"
echo "Slurm time limit: ${TIME_LIMIT:-<partition default>}"
echo "Required paths:"
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    VALUE="${!VAR_NAME:-<missing>}"
    echo "  $VAR_NAME=$VALUE"
done
echo "Profiler args: ${PROFILE_ARGS[*]}"
echo "=============================================="

FULL_CMD=(
    sbatch
    "--job-name=$JOB_NAME"
    "--partition=$PARTITION"
)
if [ -n "$TIME_LIMIT" ]; then
    FULL_CMD+=("--time=$TIME_LIMIT")
fi
FULL_CMD+=("scripts/profile_bentxt_batch_size.sbatch")
FULL_CMD+=("${PROFILE_ARGS[@]}")

printf 'Command:'
printf ' %q' "${FULL_CMD[@]}"
printf '\n\n'

if [ "$DRY_RUN" = true ]; then
    if [ "${#MISSING_ENV_VARS[@]}" -gt 0 ]; then
        echo "Missing required env vars for real submission: ${MISSING_ENV_VARS[*]}"
    fi
    echo "[Dry run - not submitting]"
else
    if [ "${#MISSING_ENV_VARS[@]}" -gt 0 ]; then
        echo "Missing required env vars: ${MISSING_ENV_VARS[*]}"
        exit 1
    fi
    echo "Submitting job..."
    "${FULL_CMD[@]}"
fi
