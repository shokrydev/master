#!/bin/bash
# Submit a short 2B diagnostic for the first 100 optimizer steps.

set -e

for ARG in "$@"; do
    if [ "$ARG" = "--size" ] || [[ "$ARG" == --size=* ]]; then
        echo "submit_initial_convergence_job.sh is fixed to --size 2B."
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/submit_finetuning_job.sh"

FIT_VALIDATION_CONFIG="configs/finetuning/bigearthnet_txt_initial_convergence_diagnostic.yaml" \
    "$SCRIPT" --size 2B "$@"
