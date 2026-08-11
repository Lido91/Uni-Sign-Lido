#!/bin/bash
# Pose2d KL-VAE path (ST-GCN fully replaced): stage-selectable runner.
#   bash script/train_pose2d_klvae.sh pretrain   # Stage A: OpenASL pretraining
#   bash script/train_pose2d_klvae.sh finetune   # Stage B: How2Sign finetune
#   bash script/train_pose2d_klvae.sh eval       # Stage C: How2Sign evaluation
#   bash script/train_pose2d_klvae.sh all        # A -> B -> C (default)
# The one-shot experiment pipeline (with --eval-on-how2sign-test selection in
# Stage A) lives in script/train_openasl_then_how2sign_klvae.sh.
set -e

stage=${1:-all}

vae_root=/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d
# {'net': ...} torch checkpoint; the Lightning file
# $vae_root/lightning_ckpt/min-MPJPE2D_hand-epoch=9.ckpt is equally supported
# by vae_encoder.py's loader.
vae_ckpt=$vae_root/net_best_hand.pth
vae_args="--vae_checkpoint $vae_ckpt --vae_mean_path $vae_root/meta/mean.npy --vae_std_path $vae_root/meta/std.npy"

gpus=localhost:0,1,2,3
port=29514

pretrain_dir=out/pose2d_klvae_openasl
finetune_dir=out/pose2d_klvae_openasl_h2s
eval_dir=out/eval_pose2d_klvae_h2s

if [ "$stage" = pretrain ] || [ "$stage" = all ]; then
  deepspeed --include $gpus --master_port $port fine_tuning_pose2d_klvae.py \
    --batch-size 16 \
    --gradient-accumulation-steps 8 \
    --epochs 20 \
    --opt AdamW \
    --lr 3e-4 \
    --output_dir $pretrain_dir \
    --dataset OpenASL \
    --task SLT \
    --dtype fp32 \
    $vae_args
fi

if [ "$stage" = finetune ] || [ "$stage" = all ]; then
  deepspeed --include $gpus --master_port $port fine_tuning_pose2d_klvae.py \
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
    $vae_args
fi

if [ "$stage" = eval ] || [ "$stage" = all ]; then
  deepspeed --include localhost:0 --master_port $port fine_tuning_pose2d_klvae.py \
    --batch-size 8 \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --finetune $finetune_dir/best_checkpoint.pth \
    --output_dir $eval_dir \
    --eval \
    $vae_args
fi
