#!/bin/bash
# SignSMD text fusion: How2Sign finetune warm-started from the OpenASL
# pose-only baseline checkpoint (identical parameter set, strict=True load).
#   bash script/finetune_h2s_signsmd_from_openasl.sh            # finetune + eval
#   bash script/finetune_h2s_signsmd_from_openasl.sh finetune   # finetune only
#   bash script/finetune_h2s_signsmd_from_openasl.sh eval       # eval only
#   bash script/finetune_h2s_signsmd_from_openasl.sh smoke      # empty-text sanity eval
#                                                               # (expected ~= baseline)
set -e

stage=${1:-all}

ckpt_path=/data/hwu/Uni_sign_repo/out/openasl_pose_pretrain/openasl_pose_only_slt.pth
signsmd_root=data/How2Sign/signsmd2d_stats_v0_1/texts

gpus=localhost:0,1,3
port=29516

finetune_dir=out/signsmd_fusion_h2s_from_openasl
eval_dir=out/eval_signsmd_fusion_h2s_from_openasl

if [ "$stage" = finetune ] || [ "$stage" = all ]; then
  deepspeed --include $gpus --master_port $port fine_tuning_signsmd.py \
   --batch-size 8 \
   --gradient-accumulation-steps 8 \
    --epochs 20 \
    --opt AdamW \
    --lr 1e-4 \
    --output_dir $finetune_dir \
    --finetune $ckpt_path \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --signsmd_text_root $signsmd_root
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
    --signsmd_text_root $signsmd_root
fi

# Pipeline sanity check before the SignSMD compiler has produced any texts:
# evaluates the OpenASL checkpoint on How2Sign test with EMPTY descriptions.
# The result should be almost identical to the plain baseline eval of the
# same checkpoint (the input only differs by one </s> token).
if [ "$stage" = smoke ]; then
  mkdir -p $signsmd_root
  deepspeed --include localhost:0 --master_port $port fine_tuning_signsmd.py \
    --batch-size 8 \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --finetune $ckpt_path \
    --output_dir out/signsmd_fusion_smoke \
    --eval \
    --signsmd_on_missing empty \
    --signsmd_text_root $signsmd_root
fi
