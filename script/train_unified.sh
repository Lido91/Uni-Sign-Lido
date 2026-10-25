#!/bin/bash
# JanusFlow-style unified (SLT + T2P) training, three stages:
#   bash script/train_unified.sh 1     # Stage 1: adapter warmup  (OpenASL, T2P-only; mT5/ST-GCN/VAE frozen)
#   bash script/train_unified.sh 2     # Stage 2: unified pretrain (OpenASL, SLT+T2P mixed; mT5 unfrozen)
#   bash script/train_unified.sh 3     # Stage 3: SFT             (How2Sign, SLT+T2P mixed; +ST-GCN unfrozen)
#   bash script/train_unified.sh all   # 1 -> 2 -> 3 sequentially (set -e stops on failure)
#   bash script/train_unified.sh eval  # SLT eval + T2P val loss of the stage-3 best
# T2P generation quality is measured separately: see script/eval_backtrans.sh.
#
# CRASH RECOVERY: every stage auto-resumes. Each epoch, the full training
# state (weights + ZeRO optimizer shards + lr scheduler + epoch + best
# metric) is snapshotted to <stage_dir>/resume_state and the wandb run id
# to <stage_dir>/wandb_run_id.txt. After an interruption just rerun the
# SAME command (e.g. `bash script/train_unified.sh all`): finished stages
# fast-forward through their epoch loop, the interrupted stage continues
# from its last completed epoch on the SAME wandb curve. Resuming needs
# the same GPU count. To deliberately retrain a stage from scratch, delete
# its resume_state/ and wandb_run_id.txt first.
set -e

stage=${1:?usage: train_unified.sh 1|2|3|eval}

vae_root=/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d
# Per-dim latent standardization stats (raw posterior latents are ~0.31
# median per-dim std — bad diffusion SNR). Generate once BEFORE stage 1:
#   python check_latent_scale.py --dataset OpenASL --num_batches 200 \
#     --vae_checkpoint $vae_root/net_best_hand.pth \
#     --save_latent_stats /data/hwu/Uni_sign_repo/out/latent_stats
latent_stats=/data/hwu/Uni_sign_repo/out/latent_stats
vae_args="--vae_checkpoint $vae_root/net_best_hand.pth --vae_mean_path $vae_root/meta/mean.npy --vae_std_path $vae_root/meta/std.npy --latent_mean_path $latent_stats/latent_mean.npy --latent_std_path $latent_stats/latent_std.npy"

# Trained ST-GCN SLT checkpoint (How2Sign pose-only): seeds mT5+ST-GCN in
# every stage and acts as the back-translation judge.
slt_ckpt=/data/hwu/Uni_sign_repo/out/how2sign_pose_only/how2sign_pose_only_slt.pth

gpus=localhost:2,3
port=29515

wandb_args="--use_wandb --wandb_project uni-sign-unified"

out_root=/data/hwu/Uni_sign_repo/out
stage1_dir=$out_root/unified_stage1_openasl
stage2_dir=$out_root/unified_stage2_openasl
stage3_dir=$out_root/unified_stage3_h2s

if [ "$stage" = 1 ] || [ "$stage" = all ]; then
  deepspeed --include $gpus --master_port $port train_unified.py \
    --train_stage 1 \
    --dataset OpenASL \
    --task SLT \
    --dtype fp32 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --epochs 10 \
    --opt AdamW \
    --lr 1e-4 \
    --skip_nonfinite_loss \
    --output_dir $stage1_dir \
    --slt_init $slt_ckpt \
    $wandb_args --wandb_run_name unified_stage1_openasl \
    $vae_args
fi

if [ "$stage" = 2 ] || [ "$stage" = all ]; then
  deepspeed --include $gpus --master_port $port train_unified.py \
    --train_stage 2 \
    --dataset OpenASL \
    --task SLT \
    --dtype fp32 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --epochs 20 \
    --opt AdamW \
    --lr 1e-4 \
    --skip_nonfinite_loss \
    --output_dir $stage2_dir \
    --slt_init $slt_ckpt \
    --finetune $stage1_dir/best_checkpoint.pth \
    $wandb_args --wandb_run_name unified_stage2_openasl \
    $vae_args
fi

if [ "$stage" = 3 ] || [ "$stage" = all ]; then
  deepspeed --include $gpus --master_port $port train_unified.py \
    --train_stage 3 \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --epochs 20 \
    --opt AdamW \
    --lr 3e-5 \
    --skip_nonfinite_loss \
    --output_dir $stage3_dir \
    --slt_init $slt_ckpt \
    --finetune $stage2_dir/best_checkpoint.pth \
    $wandb_args --wandb_run_name unified_stage3_h2s \
    $vae_args
fi

if [ "$stage" = eval ]; then
  deepspeed --include localhost:0 --master_port $port train_unified.py \
    --train_stage 3 \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --batch-size 8 \
    --slt_init $slt_ckpt \
    --finetune $stage3_dir/best_checkpoint.pth \
    --output_dir $out_root/eval_unified_h2s \
    --eval \
    $vae_args
fi
