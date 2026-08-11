#!/bin/bash
# E3 of doc/signsmd_dual_stream_fusion_plan.md: dual-stream SignSMD fusion.
# Line-for-line mirror of the early-fusion control run
# (script/finetune_h2s_signsmd_from_openasl.sh) differing ONLY in
# --fusion dual and the output dirs, so E3-vs-E1 isolates the fusion
# architecture.
#   bash script/finetune_h2s_signsmd_dual_from_openasl.sh            # finetune + eval
#   bash script/finetune_h2s_signsmd_dual_from_openasl.sh finetune   # finetune only
#   bash script/finetune_h2s_signsmd_dual_from_openasl.sh eval       # eval + masked-SMD diagnostic
#   bash script/finetune_h2s_signsmd_dual_from_openasl.sh smoke      # empty-text sanity eval
#                                                                    # (expected ~= baseline)
set -e

stage=${1:-all}

ckpt_path=/data/hwu/Uni_sign_repo/out/openasl_pose_pretrain/openasl_pose_only_slt.pth
signsmd_root=data/How2Sign/signsmd2d_fingers_v0_1/texts

gpus=localhost:0,1,2,3
port=29517  # control uses 29516; distinct so both runs can coexist

finetune_dir=/data/hwu/Uni_sign_repo/out/signsmd_dual_h2s_from_openasl_finger
eval_dir=/data/hwu/Uni_sign_repo/out/eval_signsmd_dual_h2s_from_openasl_finger

if [ "$stage" = finetune ] || [ "$stage" = all ]; then
  deepspeed --include $gpus --master_port $port fine_tuning_signsmd.py \
   --batch-size 8 \
   --gradient-accumulation-steps 8 \
    --epochs 20 \
    --opt AdamW \
    --lr 1e-4 \
    --fusion dual \
    --output_dir $finetune_dir \
    --finetune $ckpt_path \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --signsmd_text_root $signsmd_root \
    --skip_nonfinite_loss
fi

if [ "$stage" = eval ] || [ "$stage" = all ]; then
  deepspeed --include localhost:0 --master_port $port fine_tuning_signsmd.py \
    --batch-size 8 \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --fusion dual \
    --finetune $finetune_dir/best_checkpoint.pth \
    --output_dir $eval_dir \
    --eval \
    --signsmd_text_root $signsmd_root

  # pose-stream-integrity diagnostic: same checkpoint with the SMD segment
  # masked; compare against the pose-only baseline (E0) and the unmasked
  # eval above — a large unmasked-vs-masked gap = genuine SMD contribution
  deepspeed --include localhost:0 --master_port $port fine_tuning_signsmd.py \
    --batch-size 8 \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --fusion dual \
    --finetune $finetune_dir/best_checkpoint.pth \
    --output_dir ${eval_dir}_mask_smd \
    --eval \
    --eval_mask_smd \
    --signsmd_text_root $signsmd_root \
    --skip_nonfinite_loss

    
fi

# Pipeline sanity check before the SignSMD compiler has produced any texts:
# evaluates the OpenASL checkpoint on How2Sign test with EMPTY descriptions.
# Like the control's smoke, this should land ~= the plain baseline eval of
# the same checkpoint: an empty SMD stream reduces the dual model to
# (almost) the baseline computation.
if [ "$stage" = smoke ]; then
  mkdir -p $signsmd_root
  deepspeed --include localhost:0 --master_port $port fine_tuning_signsmd.py \
    --batch-size 8 \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --fusion dual \
    --finetune $ckpt_path \
    --output_dir out/signsmd_dual_smoke \
    --eval \
    --signsmd_on_missing empty \
    --signsmd_text_root $signsmd_root
fi

# Later runs of the variant ladder (doc/signsmd_dual_stream_fusion_plan.md §3):
# E5: append --fusion_alpha 0.1
#     (--output_dir out/signsmd_dual_alpha01_h2s_from_openasl)
# E6: append --fusion_gate sample --smd_dropout 0.3
#     (--output_dir out/signsmd_dual_gate_h2s_from_openasl; the warm start
#      loads with strict=False and reports only smd_gate_* as missing)
