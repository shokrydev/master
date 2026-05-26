#!/bin/bash
# ============================================================================
# Short Validation Job Submission Helper
# ============================================================================
# Usage:
#   ./scripts/submit_smoke_job.sh
#   ./scripts/submit_smoke_job.sh --condition loc_text
#   ./scripts/submit_smoke_job.sh --caption-target location_redacted_caption
#   ./scripts/submit_smoke_job.sh --dry-run
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_CONFIG="configs/finetuning/bigearthnet_txt_smoke.yaml" \
    "$SCRIPT_DIR/submit_finetuning_job.sh" "$@"
