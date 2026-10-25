#!/bin/bash
# Kept for the original How2Sign-only entrypoint; the probe runner is now
# dataset-parametric — this just forwards to it. Prefer:
#   bash script/contrastive_gt_smd.sh <stage> <How2Sign|OpenASL>
exec bash "$(dirname "$0")/contrastive_gt_smd.sh" "${1:-all}" How2Sign
