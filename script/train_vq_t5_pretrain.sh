#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# Validate the configured cache before launching distributed training.
/home/student/hwu/miniconda3/envs/Uni-Sign/bin/python \
  fine_tuning_vq_embeddings.py \
  --config configs/youtube3d_vq_embeddings_pretrain.cfg \
  --validate-cache

deepspeed --include localhost:0,1,3 --master_port 29513 \
  fine_tuning_vq_embeddings.py \
  --config configs/youtube3d_vq_embeddings_pretrain.cfg
