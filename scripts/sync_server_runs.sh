#!/bin/bash
# Sync lightweight Slurm and Lightning evidence for selected server runs.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/sync_server_runs.sh \
    --host USER@HOST \
    --remote-repo /path/to/repo \
    --remote-finetuning-output-root /path/to/finetuning_outputs \
    --remote-evaluation-output-root /path/to/evaluation_outputs \
    --jobs 11270 11271 11272

  ./scripts/sync_server_runs.sh

  ./scripts/sync_server_runs.sh --jobs-from-squeue

  ./scripts/sync_server_runs.sh --jobs-from-manifest outputs/submission_manifests/2b_trajectory.tsv

The second form reads connection defaults from .env and job ids from
planning/run_registry.md.

Options:
  --host HOST                 SSH host, for example mohamed@mars
  --remote-repo PATH          Repository path on the server, used for logs/
  --remote-finetuning-output-root PATH
                              Server FINETUNING_OUTPUT_ROOT
  --remote-evaluation-output-root PATH
                              Server EVALUATION_OUTPUT_ROOT
  --jobs JOB...               Slurm job ids to sync
  --jobs-from-squeue          Read active job ids from remote Slurm squeue
  --jobs-from-manifest PATH   Read Slurm/output ids from a trajectory TSV manifest
  --jobs-from-registry PATH    Read job ids from a registry, default planning/run_registry.md
  --dry-run                   Print rsync commands without copying
  -h, --help                  Show this help

Environment defaults:
  SERVER_SYNC_HOST
  SERVER_REPO_ROOT
  SERVER_FINETUNING_OUTPUT_ROOT
  SERVER_EVALUATION_OUTPUT_ROOT

Copied by default:
  outputs/finetuning/<job>/logs/ and lightweight finetuning evidence
  outputs/evaluation/<job>/logs/ and lightweight evaluation evidence

Large adapter/checkpoint files are intentionally excluded.
Shared thumbnails under the remote finetuning output root are copied to
outputs/thumbnails/ when present.
EOF
}

require_arg() {
    if [ -z "${2:-}" ]; then
        echo "Missing value for $1"
        exit 1
    fi
}

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

HOST="${SERVER_SYNC_HOST:-}"
REMOTE_REPO="${SERVER_REPO_ROOT:-}"
REMOTE_FINETUNING_OUTPUT_ROOT="${SERVER_FINETUNING_OUTPUT_ROOT:-}"
REMOTE_EVALUATION_OUTPUT_ROOT="${SERVER_EVALUATION_OUTPUT_ROOT:-}"
LOCAL_OUTPUT_ROOT="outputs"
DRY_RUN=false
REGISTRY_PATH="planning/run_registry.md"
JOBS_FROM_SQUEUE=false
JOBS_MANIFEST=""
JOBS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            require_arg "$1" "${2:-}"
            HOST="$2"
            shift 2
            ;;
        --remote-repo)
            require_arg "$1" "${2:-}"
            REMOTE_REPO="$2"
            shift 2
            ;;
        --remote-finetuning-output-root)
            require_arg "$1" "${2:-}"
            REMOTE_FINETUNING_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --remote-evaluation-output-root)
            require_arg "$1" "${2:-}"
            REMOTE_EVALUATION_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --jobs)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                JOBS+=("$1")
                shift
            done
            ;;
        --jobs-from-squeue)
            JOBS_FROM_SQUEUE=true
            shift
            ;;
        --jobs-from-manifest)
            require_arg "$1" "${2:-}"
            JOBS_MANIFEST="$2"
            shift 2
            ;;
        --jobs-from-registry)
            require_arg "$1" "${2:-}"
            REGISTRY_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

selection_count=0
if [ "${#JOBS[@]}" -gt 0 ]; then
    selection_count=$((selection_count + 1))
fi
if [ "$JOBS_FROM_SQUEUE" = true ]; then
    selection_count=$((selection_count + 1))
fi
if [ -n "$JOBS_MANIFEST" ]; then
    selection_count=$((selection_count + 1))
fi
if [ "$selection_count" -gt 1 ]; then
    echo "Use only one of --jobs, --jobs-from-squeue, or --jobs-from-manifest."
    exit 1
fi
if [ -n "$JOBS_MANIFEST" ]; then
    if [ ! -f "$JOBS_MANIFEST" ]; then
        echo "Submission manifest does not exist: $JOBS_MANIFEST"
        exit 1
    fi
    mapfile -t JOBS < <(
        awk -F '\t' '
            NR == 1 {
                for (column = 1; column <= NF; column++) {
                    if ($column == "output_id") output_column = column
                }
                next
            }
            {
                value = output_column ? $output_column : $1
                sub(/\/.*/, "", value)
                if (value ~ /^[0-9]+$/ && !seen[value]++) print value
            }
        ' "$JOBS_MANIFEST"
    )
    if [ "${#JOBS[@]}" -eq 0 ]; then
        echo "No evaluation job IDs found in manifest: $JOBS_MANIFEST"
        exit 1
    fi
    echo "Using ${#JOBS[@]} evaluation job IDs from $JOBS_MANIFEST."
fi
if [ "${#JOBS[@]}" -eq 0 ] && [ "$JOBS_FROM_SQUEUE" = false ]; then
    if [ ! -f "$REGISTRY_PATH" ]; then
        echo "Missing --jobs and registry file does not exist: $REGISTRY_PATH"
        usage
        exit 1
    fi
    mapfile -t JOBS < <(grep -oE '`[0-9]{5,}`' "$REGISTRY_PATH" | tr -d '`' | awk '!seen[$0]++')
    if [ "${#JOBS[@]}" -eq 0 ]; then
        echo "No job ids found in registry: $REGISTRY_PATH"
        exit 1
    fi
    echo "No --jobs supplied; using ${#JOBS[@]} job ids from $REGISTRY_PATH."
fi

if [ -z "$HOST" ] || [ -z "$REMOTE_REPO" ] || [ -z "$REMOTE_FINETUNING_OUTPUT_ROOT" ] || [ -z "$REMOTE_EVALUATION_OUTPUT_ROOT" ]; then
    echo "Missing server connection settings."
    echo "Pass the --host and --remote-*-output-root options or set SERVER_SYNC_HOST,"
    echo "SERVER_REPO_ROOT, SERVER_FINETUNING_OUTPUT_ROOT and"
    echo "SERVER_EVALUATION_OUTPUT_ROOT in .env."
    usage
    exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required."
    exit 1
fi
if ! command -v ssh >/dev/null 2>&1; then
    echo "ssh is required."
    exit 1
fi

REMOTE_REPO="${REMOTE_REPO%/}"
REMOTE_FINETUNING_OUTPUT_ROOT="${REMOTE_FINETUNING_OUTPUT_ROOT%/}"
REMOTE_EVALUATION_OUTPUT_ROOT="${REMOTE_EVALUATION_OUTPUT_ROOT%/}"

CONTROL_DIR=""
CONTROL_PATH=""
SSH_OPTS=()
RSYNC_SSH="ssh"
CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/geovlm-sync-ssh.XXXXXX")"
CONTROL_PATH="$CONTROL_DIR/control-%r@%h:%p"
SSH_OPTS=(
    -o ControlMaster=auto
    -o ControlPersist=10m
    -o "ControlPath=$CONTROL_PATH"
)
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=$CONTROL_PATH"
cleanup() {
    ssh "${SSH_OPTS[@]}" -O exit "$HOST" >/dev/null 2>&1 || true
    rm -rf "$CONTROL_DIR"
}
trap cleanup EXIT

echo "Opening SSH connection to $HOST ..."
ssh "${SSH_OPTS[@]}" "$HOST" true

if [ "$JOBS_FROM_SQUEUE" = true ]; then
    mapfile -t JOBS < <(ssh "${SSH_OPTS[@]}" "$HOST" "squeue -h -o '%i' -u \"\$USER\"" | awk '!seen[$0]++')
    if [ "${#JOBS[@]}" -eq 0 ]; then
        echo "No active Slurm jobs found for the remote user."
        exit 0
    fi
    echo "Using ${#JOBS[@]} active job ids from remote squeue: ${JOBS[*]}"
fi

remote_dir_exists() {
    local remote_dir="$1"
    local quoted
    quoted=$(printf '%q' "$remote_dir")
    ssh "${SSH_OPTS[@]}" "$HOST" "test -d $quoted"
}

run_rsync() {
    if [ "$DRY_RUN" = true ]; then
        printf 'Dry run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [ "$DRY_RUN" = false ]; then
    mkdir -p "$LOCAL_OUTPUT_ROOT"
fi

for JOB in "${JOBS[@]}"; do
    REMOTE_FINETUNING_RUN_DIR="$REMOTE_FINETUNING_OUTPUT_ROOT/bigearthnet_$JOB"
    REMOTE_EVALUATION_RUN_DIR="$REMOTE_EVALUATION_OUTPUT_ROOT/bigearthnet_$JOB"
    REMOTE_BATCH_PROFILE_DIR="$REMOTE_EVALUATION_OUTPUT_ROOT/batch_profile_$JOB"
    REMOTE_CLAIR_BATCH_PROFILE_DIR="$REMOTE_EVALUATION_OUTPUT_ROOT/clair_batch_profile_$JOB"
    REMOTE_CLAIR_DIR="$REMOTE_EVALUATION_OUTPUT_ROOT/clair_$JOB"
    REMOTE_TRAJECTORY_DIR="$REMOTE_EVALUATION_OUTPUT_ROOT/trajectory_$JOB"
    if remote_dir_exists "$REMOTE_FINETUNING_RUN_DIR"; then
        RUN_KIND="finetuning"
        SYNC_RUN_DIR="$REMOTE_FINETUNING_RUN_DIR"
    elif remote_dir_exists "$REMOTE_EVALUATION_RUN_DIR"; then
        RUN_KIND="evaluation"
        SYNC_RUN_DIR="$REMOTE_EVALUATION_RUN_DIR"
    elif remote_dir_exists "$REMOTE_BATCH_PROFILE_DIR"; then
        RUN_KIND="evaluation"
        SYNC_RUN_DIR="$REMOTE_BATCH_PROFILE_DIR"
    elif remote_dir_exists "$REMOTE_CLAIR_BATCH_PROFILE_DIR"; then
        RUN_KIND="evaluation"
        SYNC_RUN_DIR="$REMOTE_CLAIR_BATCH_PROFILE_DIR"
    elif remote_dir_exists "$REMOTE_CLAIR_DIR"; then
        RUN_KIND="evaluation"
        SYNC_RUN_DIR="$REMOTE_CLAIR_DIR"
    elif remote_dir_exists "$REMOTE_TRAJECTORY_DIR"; then
        RUN_KIND="evaluation"
        SYNC_RUN_DIR="$REMOTE_TRAJECTORY_DIR"
    else
        echo "Warning: remote run directory not found for job $JOB:"
        echo "  $REMOTE_FINETUNING_RUN_DIR"
        echo "  $REMOTE_EVALUATION_RUN_DIR"
        echo "  $REMOTE_BATCH_PROFILE_DIR"
        echo "  $REMOTE_CLAIR_BATCH_PROFILE_DIR"
        echo "  $REMOTE_CLAIR_DIR"
        echo "  $REMOTE_TRAJECTORY_DIR"
        continue
    fi

    RUN_DEST="$LOCAL_OUTPUT_ROOT/$RUN_KIND/$JOB"
    LOG_DEST="$RUN_DEST/logs"
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$LOG_DEST"
    fi

    echo "=============================================="
    echo "Syncing run $JOB"
    echo "Kind: $RUN_KIND"
    echo "Destination: $RUN_DEST"

    REMOTE_LOGS_DIR="$REMOTE_REPO/logs"
    if remote_dir_exists "$REMOTE_LOGS_DIR"; then
        run_rsync \
            rsync -av --prune-empty-dirs \
            -e "$RSYNC_SSH" \
            --include="*/" \
            --include="*_${JOB}.out" \
            --include="*_${JOB}.err" \
            --exclude="*" \
            "$HOST:$REMOTE_LOGS_DIR/" \
            "$LOG_DEST/"
    else
        echo "Warning: remote logs directory not found: $REMOTE_LOGS_DIR"
    fi

    run_rsync \
        rsync -av --prune-empty-dirs \
        -e "$RSYNC_SSH" \
        --exclude="qlora_adapter/***" \
        --exclude="qlora_adapter_best_val/***" \
        --exclude="*.safetensors" \
        --exclude="*.bin" \
        --exclude="*.pt" \
        --exclude="*.ckpt" \
        --include="lightning_logs/***" \
        --include="*/" \
        --include="*.jsonl" \
        --include="*.json" \
        --include="*.csv" \
        --include="*.tsv" \
        --include="*.log" \
        --include="*.yaml" \
        --include="*.yml" \
        --include="*.txt" \
        --exclude="*" \
        "$HOST:$SYNC_RUN_DIR/" \
        "$RUN_DEST/"
done

REMOTE_THUMBNAILS_DIR="$REMOTE_FINETUNING_OUTPUT_ROOT/thumbnails"
if remote_dir_exists "$REMOTE_THUMBNAILS_DIR"; then
    LOCAL_THUMBNAILS_DIR="$LOCAL_OUTPUT_ROOT/thumbnails"
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$LOCAL_THUMBNAILS_DIR"
    fi
    echo "=============================================="
    echo "Syncing shared thumbnails"
    echo "Destination: $LOCAL_THUMBNAILS_DIR"
    run_rsync \
        rsync -av --prune-empty-dirs \
        -e "$RSYNC_SSH" \
        --include="*/" \
        --include="*.png" \
        --include="*.jpg" \
        --include="*.jpeg" \
        --include="*.webp" \
        --include="*.json" \
        --exclude="*" \
        "$HOST:$REMOTE_THUMBNAILS_DIR/" \
        "$LOCAL_THUMBNAILS_DIR/"
fi

echo "Done."
