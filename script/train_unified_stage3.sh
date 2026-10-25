#!/usr/bin/env bash
# Stage 3: How2Sign SLT + SLG; initialize from Stage 2, keep VAE frozen.
# bash script/train_unified_stage3.sh
# GPUS=1,2,3 DRY_RUN=1 bash script/train_unified_stage3.sh
# GPUS=localhost:1,2,3 is also accepted; GPUS selects physical GPU IDs.
# BATCH_SIZE=8 by default (per GPU); use BATCH_SIZE=4 for less memory.
# Re-run with the same GPU count to resume the last completed epoch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

env_bin=${UNISIGN_ENV_BIN:-/home/student/hwu/miniconda3/envs/Uni-Sign/bin}
export PATH="$env_bin:$PATH"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/unisign-stage3-matplotlib}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/unisign-stage3-triton}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

out_root=${OUTPUT_ROOT:-/data/hwu/Uni_sign_repo/out}
output_dir=${OUTPUT_DIR:-$out_root/unified_stage3_h2s}
checkpoint=${STAGE2_CHECKPOINT:-$out_root/unified_stage2_openasl/best_checkpoint.pth}
slt_ckpt=${SLT_CHECKPOINT:-$out_root/how2sign_pose_only/how2sign_pose_only_slt.pth}
vae_root=${VAE_ROOT:-/data/hwu/momadiff_repo/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d}
latent_stats=${LATENT_STATS:-$out_root/latent_stats}
gpus=${GPUS:-1,2,3}
# Existing launch scripts use GPUS=localhost:..., while this script adds
# the host itself. Normalize before constructing DeepSpeed's --include.
gpus=${gpus#localhost:}
if [[ ! "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Invalid GPUS: ${GPUS-}; use 1,2,3 or localhost:1,2,3" >&2
    exit 2
fi
# --include selects physical devices. Do not inherit a competing mask.
unset CUDA_VISIBLE_DEVICES
port=${MASTER_PORT:-29515}
batch_size=${BATCH_SIZE:-8}
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid BATCH_SIZE: $batch_size; use a positive integer" >&2
    exit 2
fi

for path in "$env_bin/deepspeed" "$checkpoint" "$slt_ckpt" \
    "$vae_root/net_best_hand.pth" "$vae_root/meta/mean.npy" \
    "$vae_root/meta/std.npy" "$latent_stats/latent_mean.npy" \
    "$latent_stats/latent_std.npy" data/How2Sign/labels.train \
    data/How2Sign/labels.test; do
    if [[ ! -s "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

command=("$env_bin/deepspeed" --include "localhost:$gpus" --master_port "$port"
    train_unified.py
    --train_stage 3 --dataset How2Sign --task SLT --dtype fp32
    --batch-size "$batch_size" --gradient-accumulation-steps 1
    --epochs 20 --opt AdamW --lr 3e-5 --skip_nonfinite_loss
    --val_split test --resume auto
    --output_dir "$output_dir" --slt_init "$slt_ckpt" --finetune "$checkpoint"
    --vae_checkpoint "$vae_root/net_best_hand.pth"
    --vae_mean_path "$vae_root/meta/mean.npy"
    --vae_std_path "$vae_root/meta/std.npy"
    --latent_mean_path "$latent_stats/latent_mean.npy"
    --latent_std_path "$latent_stats/latent_std.npy"
    --use_wandb --wandb_project uni-sign-unified
    --wandb_run_name unified_stage3_h2s
    "$@")

echo "Stage 3: How2Sign SLT + SLG; GPUs: $gpus; batch per GPU: $batch_size; epochs: 20"
echo "Stage-2 checkpoint: $checkpoint"
echo "Output: $output_dir; W&B: uni-sign-unified/unified_stage3_h2s"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
exec "${command[@]}"
