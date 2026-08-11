#!/bin/bash
# Full pipeline for the pose2d KL-VAE path (ST-GCN fully replaced):
#   Stage A  OpenASL pretraining, checkpoint selected on How2Sign test
#            (same --eval-on-how2sign-test convention as
#            script/train_openasl_then_how2sign.sh)
#   Stage B  How2Sign finetuning from the Stage-A best checkpoint
#   Stage C  final evaluation (BLEU/ROUGE/BLEURT) on How2Sign test
#
# Encoder: MoMADiff joint How2Sign+OpenASL 138D KL-VAE, posterior mean
# [B,T/4,512] -> ONE Linear(512,768) head -> mT5.
set -e

vae_root=/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d
vae_ckpt=$vae_root/net_best_hand.pth
vae_mean=$vae_root/meta/mean.npy
vae_std=$vae_root/meta/std.npy

gpus=localhost:0,1,3
port=29513

pretrain_dir=out/pose2d_klvae_pretrain_openasl
finetune_dir=out/pose2d_klvae_finetune_openasl_h2s
eval_dir=out/eval_pose2d_klvae_h2s

# ---- Stage A: OpenASL pretraining (select best by How2Sign test) ------------
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
  --eval-on-how2sign-test \
  --vae_checkpoint $vae_ckpt \
  --vae_mean_path $vae_mean \
  --vae_std_path $vae_std

# ---- Stage B: How2Sign finetuning from the Stage-A best checkpoint ----------
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
  --vae_checkpoint $vae_ckpt \
  --vae_mean_path $vae_mean \
  --vae_std_path $vae_std

# ---- Stage C: final How2Sign evaluation (incl. BLEURT) ----------------------
deepspeed --include localhost:0 --master_port $port fine_tuning_pose2d_klvae.py \
  --batch-size 8 \
  --dataset How2Sign \
  --task SLT \
  --dtype fp32 \
  --vae_checkpoint $vae_ckpt \
  --vae_mean_path $vae_mean \
  --vae_std_path $vae_std \
  --finetune $finetune_dir/best_checkpoint.pth \
  --output_dir $eval_dir \
  --eval
