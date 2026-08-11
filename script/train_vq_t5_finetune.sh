#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

pretrain_checkpoint=/data/hwu/Uni_sign_repo/out/t5_youtube3d_vq_pretrain/best_checkpoint.pth
if [[ ! -f "$pretrain_checkpoint" ]]; then
  echo "Pretraining checkpoint not found: $pretrain_checkpoint" >&2
  exit 1
fi

/home/student/hwu/miniconda3/envs/Uni-Sign/bin/python \
  fine_tuning_vq_embeddings.py \
  --config configs/how2sign_vq_embeddings_finetune.cfg \
  --validate-cache

deepspeed --include localhost:0,1,3 --master_port 29514 \
  fine_tuning_vq_embeddings.py \
  --config configs/how2sign_vq_embeddings_finetune.cfg \
  --finetune "$pretrain_checkpoint" \
  --dtype fp32 \

  

