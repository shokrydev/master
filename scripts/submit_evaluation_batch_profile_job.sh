#!/bin/bash
# Submit the task-aware capacity and throughput 2B evaluation profiler.

set -e

if [ ! -f .env ]; then
    echo "Missing .env. Copy .env.example to .env and fill in server paths."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

adapter_dir=""
job_name="profile-eval-batches-2B"
partition="${SLURM_DEFAULT_PARTITION:-}"
dependency=""
dry_run=false
profile_args=()

require_arg() {
    if [ -z "${2:-}" ]; then
        echo "Missing value for $1"
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter-dir)
            require_arg "$1" "${2:-}"
            adapter_dir="$2"
            profile_args+=("$1" "$2")
            shift 2
            ;;
        --name)
            require_arg "$1" "${2:-}"
            job_name="$2"
            shift 2
            ;;
        --partition)
            require_arg "$1" "${2:-}"
            partition="$2"
            shift 2
            ;;
        --dependency)
            require_arg "$1" "${2:-}"
            dependency="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        *)
            profile_args+=("$1")
            shift
            ;;
    esac
done

if [ -z "$adapter_dir" ]; then
    echo "Missing --adapter-dir."
    exit 1
fi
if [ -z "$partition" ]; then
    echo "Missing partition. Set SLURM_DEFAULT_PARTITION or pass --partition."
    exit 1
fi

required_env_vars=(
    BIGEARTHNET_V2_LMDB_ROOT
    BIGEARTHNET_TXT_PARQUET_PATH
    BIGEARTHNET_ENCODER_DIR
    EVALUATION_OUTPUT_ROOT
    HF_HOME
    SATCLIP_L40_CHECKPOINT_PATH
)
missing_env_vars=()
for var_name in "${required_env_vars[@]}"; do
    if [ -z "${!var_name:-}" ]; then
        missing_env_vars+=("$var_name")
    fi
done

mkdir -p logs
full_cmd=(
    sbatch
    "--job-name=$job_name"
    "--partition=$partition"
)
if [ -n "$dependency" ]; then
    full_cmd+=("--dependency=$dependency")
fi
full_cmd+=(scripts/profile_bentxt_evaluation_batch_size.sbatch)
full_cmd+=("${profile_args[@]}")

echo "Evaluation generation batch profiler"
echo "Adapter: $adapter_dir"
echo "Dependency: ${dependency:-<none>}"
echo "Candidate args: ${profile_args[*]}"
printf 'Command:'
printf ' %q' "${full_cmd[@]}"
printf '\n'

if [ "$dry_run" = true ]; then
    if [ "${#missing_env_vars[@]}" -gt 0 ]; then
        echo "Missing required env vars for real submission: ${missing_env_vars[*]}"
    fi
    echo "[Dry run - not submitting]"
    exit 0
fi
if [ "${#missing_env_vars[@]}" -gt 0 ]; then
    echo "Missing required env vars: ${missing_env_vars[*]}"
    exit 1
fi
if [ -z "$dependency" ] && [ ! -d "$adapter_dir" ]; then
    echo "Adapter directory is not a directory: $adapter_dir"
    exit 1
fi
"${full_cmd[@]}"
