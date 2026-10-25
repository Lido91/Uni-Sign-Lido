#!/bin/bash
# Contrastive probe: GT translation vs SignSMD text — any supported dataset.
# Answers "does the SMD carry clip-specific semantic signal?" via text-text
# retrieval — no pose, no DeepSpeed, single GPU (doc/signsmd_contrastive_probe_plan.md).
#   bash script/contrastive_gt_smd.sh <stage> <dataset>
#     stage:   all (default) | zeroshot | train | eval | smoke
#     dataset: How2Sign (default) | OpenASL | both (= pooled How2Sign+OpenASL)
#   bash script/contrastive_gt_smd.sh                  # How2Sign train + eval
#   bash script/contrastive_gt_smd.sh all OpenASL      # OpenASL  train + eval
#   bash script/contrastive_gt_smd.sh all both         # joint training, per-dataset eval
#   bash script/contrastive_gt_smd.sh zeroshot OpenASL # pretrained-encoder floor (~chance)
#
# Selection: How2Sign uses a held-out 5% train slice (no dev split);
# OpenASL validates on its real dev split; "both" averages the two val R@1s.
# Eval is always PER dataset (mixed test pools would reward corpus-style
# cues instead of GT<->SMD alignment).
set -e

stage=${1:-all}
dataset=${2:-How2Sign}
if [ "$dataset" = both ]; then
  dataset=How2Sign,OpenASL
fi

# {} is resolved per dataset by contrastive_gt_smd.py
signsmd_root="data/{}/signsmd2d_fingers_v0_1/texts"
ds_tag=$(echo "$dataset" | tr '[:upper:],' '[:lower:]_')
run_dir=out/contrastive_gt_smd_$ds_tag

gpu=0

common="--dataset $dataset \
  --signsmd_text_root $signsmd_root \
  --signsmd_on_missing skip \
  --output_dir $run_dir"

# Zero-shot floor: raw mT5 mean-pooled embeddings, no training, no head.
# Expected ~= chance (1/N) — the trained probe is the meaningful number.
if [ "$stage" = zeroshot ]; then
  CUDA_VISIBLE_DEVICES=$gpu python contrastive_gt_smd.py $common \
    --eval --head none --bf16 \
    --output_dir ${run_dir}_zeroshot
fi

# batch_size is also the InfoNCE negative count + 1 — raise it as far as
# memory allows. Levers if it OOMs: keep --grad_checkpointing (~halves
# activation memory), lower --smd_max_tokens (attention memory shrinks
# quadratically; the fingers texts are compact), or --freeze_encoder
# (linear-probe regime, batch 512+ fits). Gradient accumulation does NOT
# add negatives — only the per-forward batch counts.
if [ "$stage" = train ] || [ "$stage" = all ]; then
  CUDA_VISIBLE_DEVICES=$gpu python contrastive_gt_smd.py $common \
    --epochs 20 \
    --batch_size 128 \
    --lr 1e-4 \
    --lr_encoder 1e-5 \
    --bf16 \
    --grad_checkpointing
fi

if [ "$stage" = eval ] || [ "$stage" = all ]; then
  CUDA_VISIBLE_DEVICES=$gpu python contrastive_gt_smd.py $common \
    --eval --bf16 \
    --resume $run_dir/best_checkpoint.pth
fi

# 128 pairs, 1 epoch: verifies data pairing, forward/backward, checkpointing,
# and the eval/CSV path end to end in a couple of minutes.
if [ "$stage" = smoke ]; then
  CUDA_VISIBLE_DEVICES=$gpu python contrastive_gt_smd.py $common \
    --limit 128 --epochs 1 --batch_size 8 --bf16 \
    --output_dir out/contrastive_gt_smd_smoke
  CUDA_VISIBLE_DEVICES=$gpu python contrastive_gt_smd.py $common \
    --limit 128 --eval --bf16 \
    --resume out/contrastive_gt_smd_smoke/best_checkpoint.pth \
    --output_dir out/contrastive_gt_smd_smoke
fi
