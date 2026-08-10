#!/bin/bash
# Submit a short 2B early-convergence diagnostic, optionally followed by
# dependent correct/shuffled benchmark evaluations.
#
# Usage:
#   ./scripts/submit_early_convergence_job.sh --condition no_loc --name no-loc-2B-1000
#   ./scripts/submit_early_convergence_job.sh --condition loc_embed --name loc-embed-2B-1000 --submit-evaluations --config path/to/ablation.yaml

set -e

SUBMIT_EVALUATIONS=false
TRAIN_ARGS=()
EVAL_CONFIG_ARGS=()
CONDITION="loc_embed"
JOB_NAME=""
DRY_RUN=false

ARGS=("$@")
INDEX=0
while [ "$INDEX" -lt "${#ARGS[@]}" ]; do
    ARG="${ARGS[$INDEX]}"
    if [ "$ARG" = "--size" ] || [[ "$ARG" == --size=* ]]; then
        echo "submit_early_convergence_job.sh is fixed to --size 2B."
        exit 1
    fi
    case "$ARG" in
        --submit-evaluations)
            SUBMIT_EVALUATIONS=true
            INDEX=$((INDEX + 1))
            ;;
        --condition)
            if [ $((INDEX + 1)) -ge "${#ARGS[@]}" ]; then
                echo "Missing value for --condition"
                exit 1
            fi
            CONDITION="${ARGS[$((INDEX + 1))]}"
            TRAIN_ARGS+=("$ARG" "$CONDITION")
            INDEX=$((INDEX + 2))
            ;;
        --name)
            if [ $((INDEX + 1)) -ge "${#ARGS[@]}" ]; then
                echo "Missing value for --name"
                exit 1
            fi
            JOB_NAME="${ARGS[$((INDEX + 1))]}"
            TRAIN_ARGS+=("$ARG" "$JOB_NAME")
            INDEX=$((INDEX + 2))
            ;;
        --config)
            if [ $((INDEX + 1)) -ge "${#ARGS[@]}" ]; then
                echo "Missing value for --config"
                exit 1
            fi
            CONFIG_PATH="${ARGS[$((INDEX + 1))]}"
            TRAIN_ARGS+=("$ARG" "$CONFIG_PATH")
            EVAL_CONFIG_ARGS+=("$ARG" "$CONFIG_PATH")
            INDEX=$((INDEX + 2))
            ;;
        --dry-run)
            DRY_RUN=true
            TRAIN_ARGS+=("$ARG")
            INDEX=$((INDEX + 1))
            ;;
        *)
            TRAIN_ARGS+=("$ARG")
            INDEX=$((INDEX + 1))
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/submit_finetuning_job.sh"

if [ "$SUBMIT_EVALUATIONS" = false ]; then
    FIT_VALIDATION_CONFIG="configs/finetuning/bigearthnet_txt_early_convergence_diagnostic.yaml" \
        "$SCRIPT" --size 2B "${TRAIN_ARGS[@]}"
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    FIT_VALIDATION_CONFIG="configs/finetuning/bigearthnet_txt_early_convergence_diagnostic.yaml" \
        "$SCRIPT" --size 2B "${TRAIN_ARGS[@]}"
    echo
    echo "[Dry run - dependent evaluations will use the real training job ID]"
    exit 0
fi

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="${CONDITION}-2B-early"
fi

TRAIN_OUTPUT="$(
    FIT_VALIDATION_CONFIG="configs/finetuning/bigearthnet_txt_early_convergence_diagnostic.yaml" \
        "$SCRIPT" --size 2B "${TRAIN_ARGS[@]}"
)"
printf '%s\n' "$TRAIN_OUTPUT"

TRAIN_JOB_ID="$(
    printf '%s\n' "$TRAIN_OUTPUT" \
        | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' \
        | tail -n 1
)"
if [ -z "$TRAIN_JOB_ID" ]; then
    echo "Could not extract the Slurm training job ID; evaluations were not submitted."
    exit 1
fi

if [ -z "${FINETUNING_OUTPUT_ROOT:-}" ]; then
    if [ ! -f .env ]; then
        echo "Missing .env; cannot derive the future adapter directory."
        exit 1
    fi
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
if [ -z "${FINETUNING_OUTPUT_ROOT:-}" ]; then
    echo "FINETUNING_OUTPUT_ROOT is required to derive the future adapter directory."
    exit 1
fi

ADAPTER_DIR="${FINETUNING_OUTPUT_ROOT%/}/bigearthnet_${TRAIN_JOB_ID}/qlora_adapter"
EVAL_SCRIPT="$SCRIPT_DIR/submit_evaluation_job.sh"
DEPENDENCY="afterok:${TRAIN_JOB_ID}"

"$EVAL_SCRIPT" \
    --condition "$CONDITION" \
    --size 2B \
    --adapter-dir "$ADAPTER_DIR" \
    --name "eval-${TRAIN_JOB_ID}-${CONDITION}-correct" \
    --run-label "${JOB_NAME}-j${TRAIN_JOB_ID}" \
    --dependency "$DEPENDENCY" \
    "${EVAL_CONFIG_ARGS[@]}"

if [ "$CONDITION" != "no_loc" ]; then
    "$EVAL_SCRIPT" \
        --condition "$CONDITION" \
        --size 2B \
        --adapter-dir "$ADAPTER_DIR" \
        --name "eval-${TRAIN_JOB_ID}-${CONDITION}-shuffled" \
        --run-label "${JOB_NAME}-shuffled-j${TRAIN_JOB_ID}" \
        --coordinate-perturbation shuffled \
        --dependency "$DEPENDENCY" \
        "${EVAL_CONFIG_ARGS[@]}"
fi
