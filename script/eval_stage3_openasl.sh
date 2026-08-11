#!/usr/bin/env bash
set -euo pipefail

ckpt_path=/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl/best_checkpoint.pth
output_dir=out/eval_stage1_pretraining_openasl

if [[ ! -f "$ckpt_path" ]]; then
  echo "Checkpoint not found: $ckpt_path" >&2
  exit 1
fi

# Single-GPU, pose-only evaluation on the OpenASL dev and test splits.
deepspeed --include localhost:2 --master_port 29512 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir "$output_dir" \
  --finetune "$ckpt_path" \
  --dataset OpenASL \
  --task SLT \
  --dtype fp32 \
  --eval \
  --skip_bleurt
