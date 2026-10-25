#!/usr/bin/env bash
# One 40-epoch joint run on all OpenASL + How2Sign training samples.
# Stage 3 freezing policy: train ST-GCN, mT5, generation and geometry; freeze VAE.
# Reuse the existing Stage 1 warmup, without requiring a new Stage 2 first.
set -euo pipefail
export MERGE_DATASETS=1
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-${OUTPUT_ROOT:-/data/hwu/Uni_sign_repo/out}/unified_stage1_openasl/best_checkpoint.pth}
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_unified_geosign.sh" 3 \
    --allow_plain_geosign_init "$@"
