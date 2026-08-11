#!/bin/bash
# Back-translation evaluation ladder (single GPU).
#   bash script/eval_backtrans.sh ceilings  # gt / gt_conf1 / vae_recon — no
#                                           # trained generation model needed
#   bash script/eval_backtrans.sh gen       # text->pose of the unified model
set -e

what=${1:?usage: eval_backtrans.sh ceilings|gen}

vae_root=/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d
latent_stats=/data/hwu/Uni_sign_repo/out/latent_stats
vae_args="--vae_checkpoint $vae_root/net_best_hand.pth --vae_mean_path $vae_root/meta/mean.npy --vae_std_path $vae_root/meta/std.npy --latent_mean_path $latent_stats/latent_mean.npy --latent_std_path $latent_stats/latent_std.npy"

# Frozen judge: trained ST-GCN SLT checkpoint (How2Sign pose-only).
slt_ckpt=/data/hwu/Uni_sign_repo/out/how2sign_pose_only/how2sign_pose_only_slt.pth

out_root=/data/hwu/Uni_sign_repo/out
unified_ckpt=$out_root/unified_stage3_h2s/best_checkpoint.pth
out_dir=$out_root/backtrans_eval
common="--dataset How2Sign --split test --batch-size 8 --max_length 400 \
        --slt_checkpoint $slt_ckpt --output_dir $out_dir"

if [ "$what" = ceilings ]; then
  python back_translation_eval.py $common --mode gt
  python back_translation_eval.py $common --mode gt_conf1
  python back_translation_eval.py $common --mode vae_recon $vae_args
fi

if [ "$what" = gen ]; then
  python back_translation_eval.py $common --mode gen $vae_args \
    --unified_checkpoint $unified_ckpt \
    --trans_infer_timesteps 9 \
    --ddim_steps 100 \
    --guidance_param 3.0
fi
