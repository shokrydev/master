#!/bin/bash
# Submit a short 2B early-convergence diagnostic.

set -e

for ARG in "$@"; do
    if [ "$ARG" = "--size" ] || [[ "$ARG" == --size=* ]]; then
        echo "submit_early_convergence_job.sh is fixed to --size 2B."
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/submit_finetuning_job.sh"

FIT_VALIDATION_CONFIG="configs/finetuning/bigearthnet_txt_early_convergence_diagnostic.yaml" \
    "$SCRIPT" --size 2B "$@"
