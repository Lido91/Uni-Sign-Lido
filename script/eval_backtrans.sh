#!/bin/bash
# Back-translation evaluation ladder (single GPU).
#   bash script/eval_backtrans.sh ceilings  # gt / gt_conf1 / vae_recon — no
#                                           # trained generation model needed
#   bash script/eval_backtrans.sh gen       # text->pose of the unified model
#   bash script/eval_backtrans.sh pred [dir]  # external test_predictions dump
#                                             # (MoMADiff eval_t2m.py); dir
#                                             # defaults to the Trans-B
#                                             # h2s_openasl_pose2d eval below
set -e

what=${1:?usage: eval_backtrans.sh ceilings|gen|pred}

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

if [ "$what" = pred ]; then
  # Per-sample {SENTENCE_NAME}.pkl dump from MoMADiff eval_t2m.py
  # (save_predictions: true). Feats are in the KL-VAE-meta-normalized 138D
  # space, so mean/std here MUST be the meta stats of the KL-VAE that
  # MoMADiff run was configured with.
  momadiff_root=/home/student/hwu/Workplace/MoMADiff
  default_pred_dir=$momadiff_root/checkpoints/how2sign_openasl_pose2d/Trans-B_EMA_h2s_openasl_pose2d_masked-only_diff1000/eval_default_ddim100/latest_ema/test_predictions
  pred_dir=${2:-$default_pred_dir}
  python back_translation_eval.py $common --mode pred_dir \
    --pred_dir "$pred_dir" \
    --vae_mean_path $vae_root/meta/mean.npy \
    --vae_std_path $vae_root/meta/std.npy
fi
