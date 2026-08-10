#!/bin/bash
# ============================================================================
# BigEarthNet.txt Evaluation Submission Helper
# ============================================================================
# Usage:
#   ./scripts/submit_evaluation_job.sh --condition loc_text --size 2B --adapter-dir /path/to/adapter
#   ./scripts/submit_evaluation_job.sh --condition loc_embed --size 8B --adapter-dir /path/to/adapter --dry-run
#   ./scripts/submit_evaluation_job.sh --condition loc_embed --size 2B --adapter-dir /path/to/adapter --coordinate-perturbation shuffled
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

SIZE=""
CONDITION=""
ADAPTER_DIR=""
JOB_NAME=""
RUN_LABEL=""
COORDINATE_PERTURBATION=""
PARTITION="${SLURM_DEFAULT_PARTITION:-}"
TIME_LIMIT=""
MEMORY=""
CPUS=""
DRY_RUN=false
EXTRA_ARGS=()

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
            shift 2
            ;;
        --condition)
            require_arg "$1" "${2:-}"
            CONDITION="$2"
            shift 2
            ;;
        --adapter-dir)
            require_arg "$1" "${2:-}"
            ADAPTER_DIR="$2"
            shift 2
            ;;
        --name)
            require_arg "$1" "${2:-}"
            JOB_NAME="$2"
            shift 2
            ;;
        --run-label)
            require_arg "$1" "${2:-}"
            RUN_LABEL="$2"
            shift 2
            ;;
        --coordinate-perturbation)
            require_arg "$1" "${2:-}"
            COORDINATE_PERTURBATION="$2"
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
        --mem)
            require_arg "$1" "${2:-}"
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            require_arg "$1" "${2:-}"
            CPUS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$SIZE" ]; then
    echo "Missing --size. Use 2B, 4B or 8B."
    exit 1
fi
case "$SIZE" in
    2B)
        MODEL_NAME="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
        ;;
    4B)
        MODEL_NAME="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
        ;;
    8B)
        MODEL_NAME="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
        ;;
    *)
        echo "Invalid --size '$SIZE'. Use 2B, 4B or 8B."
        exit 1
        ;;
esac

if [ -z "$CONDITION" ]; then
    echo "Missing --condition. Use no_loc, loc_text, loc_embed, loc_encoding or loc_additive_satclip."
    exit 1
fi
case "$CONDITION" in
    no_loc)
        CONDITION_CONFIG=""
        ;;
    loc_text)
        CONDITION_CONFIG="configs/finetuning/loc_text.yaml"
        ;;
    loc_embed)
        CONDITION_CONFIG="configs/finetuning/loc_embed.yaml"
        ;;
    loc_encoding)
        CONDITION_CONFIG="configs/finetuning/loc_encoding.yaml"
        ;;
    loc_additive_satclip)
        CONDITION_CONFIG="configs/finetuning/loc_additive_satclip.yaml"
        ;;
    *)
        echo "Invalid --condition '$CONDITION'. Use no_loc, loc_text, loc_embed, loc_encoding or loc_additive_satclip."
        exit 1
        ;;
esac

case "$COORDINATE_PERTURBATION" in
    ""|shuffled|antipodal)
        ;;
    *)
        echo "Invalid --coordinate-perturbation '$COORDINATE_PERTURBATION'. Use shuffled or antipodal."
        exit 1
        ;;
esac
if [ "$CONDITION" = "no_loc" ] && [ -n "$COORDINATE_PERTURBATION" ]; then
    echo "--coordinate-perturbation is only meaningful for location-conditioned runs."
    exit 1
fi

if [ -z "$ADAPTER_DIR" ]; then
    echo "Missing --adapter-dir."
    exit 1
fi
if [ -z "$RUN_LABEL" ]; then
    if [ -n "$COORDINATE_PERTURBATION" ]; then
        RUN_LABEL="${CONDITION}-${SIZE}-${COORDINATE_PERTURBATION}-eval"
    else
        RUN_LABEL="${CONDITION}-${SIZE}-eval"
    fi
fi
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="$RUN_LABEL"
fi
if [ -z "$PARTITION" ]; then
    echo "Missing Slurm partition. Set SLURM_DEFAULT_PARTITION in .env or pass --partition."
    exit 1
fi

REQUIRED_ENV_VARS=(
    BIGEARTHNET_V2_LMDB_ROOT
    BIGEARTHNET_TXT_PARQUET_PATH
    BIGEARTHNET_ENCODER_DIR
    EVALUATION_OUTPUT_ROOT
    HF_HOME
)
if [ "$CONDITION" = "loc_embed" ]; then
    REQUIRED_ENV_VARS+=(SATCLIP_CHECKPOINT_PATH)
fi
if [ "$CONDITION" = "loc_additive_satclip" ]; then
    REQUIRED_ENV_VARS+=(SATCLIP_L40_CHECKPOINT_PATH)
fi
MISSING_ENV_VARS=()
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    if [ -z "${!VAR_NAME:-}" ]; then
        MISSING_ENV_VARS+=("$VAR_NAME")
    fi
done

mkdir -p logs

echo "=============================================="
echo "BigEarthNet.txt Evaluation Submission"
echo "=============================================="
echo "Base config: configs/evaluation/bigearthnet_txt.yaml"
echo "Condition config: ${CONDITION_CONFIG:-<none>}"
echo "Condition: $CONDITION"
echo "Size: $SIZE"
echo "Model: $MODEL_NAME"
echo "Adapter dir: $ADAPTER_DIR"
echo "Coordinate perturbation: ${COORDINATE_PERTURBATION:-<none>}"
echo "Run label: $RUN_LABEL"
echo "Job name: $JOB_NAME"
echo "Slurm partition: $PARTITION"
echo "Slurm time limit: ${TIME_LIMIT:-<partition default>}"
echo "Slurm memory: ${MEMORY:-<sbatch default>}"
echo "Slurm CPUs per task: ${CPUS:-<sbatch default>}"
echo "Evaluation output root: $EVALUATION_OUTPUT_ROOT"
echo "Required paths:"
for VAR_NAME in "${REQUIRED_ENV_VARS[@]}"; do
    VALUE="${!VAR_NAME:-<missing>}"
    echo "  $VAR_NAME=$VALUE"
done
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "=============================================="

SCRIPT="scripts/evaluate_job.sbatch"
FULL_CMD=(
    sbatch
    "--job-name=$JOB_NAME"
    "--partition=$PARTITION"
)
if [ -n "$TIME_LIMIT" ]; then
    FULL_CMD+=("--time=$TIME_LIMIT")
fi
if [ -n "$MEMORY" ]; then
    FULL_CMD+=("--mem=$MEMORY")
fi
if [ -n "$CPUS" ]; then
    FULL_CMD+=("--cpus-per-task=$CPUS")
fi
FULL_CMD+=("--export=ALL,CONDITION_CONFIG=$CONDITION_CONFIG,EVAL_ADAPTER_DIR=$ADAPTER_DIR,RUN_LABEL=$RUN_LABEL,MODEL_SIZE=$SIZE" "$SCRIPT")
FULL_CMD+=("--model.init_args.model_name_or_path" "$MODEL_NAME")
if [ -n "$COORDINATE_PERTURBATION" ]; then
    FULL_CMD+=("--data.init_args.coordinate_perturbation" "$COORDINATE_PERTURBATION")
fi
FULL_CMD+=("${EXTRA_ARGS[@]}")

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
        echo "Set them before submitting so Slurm can inherit them via --export=ALL."
        exit 1
    fi
    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "Adapter directory is not a directory: $ADAPTER_DIR"
        exit 1
    fi
    echo "Submitting job..."
    "${FULL_CMD[@]}"
fi
