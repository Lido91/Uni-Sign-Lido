#!/bin/bash
# Baseline (ST-GCN + mT5) + SignSMD text fusion: stage-selectable runner.
#   bash script/train_signsmd_fusion.sh pretrain   # Stage A: OpenASL pretraining (needs OpenASL texts)
#   bash script/train_signsmd_fusion.sh finetune   # Stage B: How2Sign finetune
#   bash script/train_signsmd_fusion.sh eval       # Stage C: How2Sign evaluation
#   bash script/train_signsmd_fusion.sh all        # A -> B -> C (default)
#
# The fused model has the SAME parameters as the baseline models.Uni_Sign,
# so any baseline mT5 checkpoint can warm-start either stage via --finetune.
#
# Frame sampling defaults to --frame_sampling linear (deterministic), so the
# pose stream and the SignSMD text cover exactly the same time; pass
# --frame_sampling unisign_random to ablate the baseline random sampling.
set -e

stage=${1:-all}

# Offline-compiled SignSMD description roots (one <clip>.txt per sample).
# Step-aligned contracts (regenerate with generate_signsmd2d.py first):
#   coarse: data/<ds>/signsmd2d_v0_2/texts        (interval prose)
#   stats:  data/<ds>/signsmd2d_stats_v0_1/texts  (numeric timelines)
h2s_signsmd_root=data/How2Sign/signsmd2d_v0_2/texts
openasl_signsmd_root=data/OpenASL/signsmd2d_v0_2/texts

# Optional baseline warm start for Stage A (leave empty to train from scratch).
baseline_ckpt=

gpus=localhost:0,1,2,3
port=29515

pretrain_dir=/data/hwu/Uni_sign_repo/out/signsmd_fusion_openasl
finetune_dir=/data/hwu/Uni_sign_repo/out/signsmd_fusion_openasl_h2s
eval_dir=/data/hwu/Uni_sign_repo/out/eval_signsmd_fusion_h2s

if [ "$stage" = pretrain ] || [ "$stage" = all ]; then
  finetune_arg=
  if [ -n "$baseline_ckpt" ]; then
    finetune_arg="--finetune $baseline_ckpt"
  fi
  deepspeed --include $gpus --master_port $port fine_tuning_signsmd.py \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --epochs 20 \
    --opt AdamW \
    --lr 3e-4 \
    --output_dir $pretrain_dir \
    --dataset OpenASL \
    --task SLT \
    --dtype fp32 \
    --skip_nonfinite_loss \
    --signsmd_text_root $openasl_signsmd_root \
    $finetune_arg
fi

if [ "$stage" = finetune ] || [ "$stage" = all ]; then
  deepspeed --include $gpus --master_port $port fine_tuning_signsmd.py \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --epochs 20 \
    --opt AdamW \
    --lr 3e-4 \
    --output_dir $finetune_dir \
    --finetune $pretrain_dir/best_checkpoint.pth \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --skip_nonfinite_loss \
    --signsmd_text_root $h2s_signsmd_root
fi

if [ "$stage" = eval ] || [ "$stage" = all ]; then
  deepspeed --include localhost:0 --master_port $port fine_tuning_signsmd.py \
    --batch-size 8 \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --finetune $finetune_dir/best_checkpoint.pth \
    --output_dir $eval_dir \
    --eval \
    --signsmd_text_root $h2s_signsmd_root
fi
