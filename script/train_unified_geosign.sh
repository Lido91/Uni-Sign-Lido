#!/usr/bin/env bash
# Enhanced Stage 2/3: unchanged Geo-Sign SLT objective + existing SLG diffusion.
# bash script/train_unified_geosign.sh 2|3 [extra train_unified.py arguments]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage=${1:?usage: train_unified_geosign.sh 2|3}
shift
case "$stage" in 2|3) ;; *) echo 'Only joint Stage 2/3 are supported' >&2; exit 2 ;; esac

env_bin=${UNISIGN_ENV_BIN:-/home/student/hwu/miniconda3/envs/Uni-Sign/bin}
export PATH="$env_bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/unisign-geosign-matplotlib}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/unisign-geosign-triton}
# Explicit selection avoids inherited GPUS accidentally selecting occupied GPU 0.
gpus=${GEOSIGN_GPUS:-1,2,3}
gpus=${gpus#localhost:}
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GEOSIGN_GPUS' >&2; exit 2; }
unset CUDA_VISIBLE_DEVICES

out_root=${OUTPUT_ROOT:-/data/hwu/Uni_sign_repo/out}
vae_root=${VAE_ROOT:-/data/hwu/momadiff_repo/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d}
latent_stats=${LATENT_STATS:-$out_root/latent_stats}
slt_ckpt=${SLT_CHECKPOINT:-$out_root/how2sign_pose_only/how2sign_pose_only_slt.pth}
extra=()
merged=${MERGE_DATASETS:-0}
case "$merged" in 0|1) ;; *) echo 'MERGE_DATASETS must be 0 or 1' >&2; exit 2 ;; esac
if [[ "$stage" == 2 ]]; then
    dataset=OpenASL
    run_name=unified_stage2_openasl_geosign
    checkpoint=${INIT_CHECKPOINT:-$out_root/unified_stage1_openasl/best_checkpoint.pth}
    lr=${LR:-1e-4}
    extra+=(--allow_plain_geosign_init)
    if [[ "$merged" == 1 ]]; then run_name=unified_stage2_merged_geosign; fi
else
    dataset=How2Sign
    run_name=unified_stage3_h2s_geosign
    checkpoint=${INIT_CHECKPOINT:-$out_root/unified_stage2_openasl_geosign/best_checkpoint.pth}
    lr=${LR:-3e-5}
    if [[ "$merged" == 1 ]]; then
        run_name=unified_stage3_merged_geosign
        checkpoint=${INIT_CHECKPOINT:-$out_root/unified_stage2_merged_geosign/best_checkpoint.pth}
    fi
fi
if [[ "$merged" == 1 ]]; then extra+=(--merge_datasets); fi
output_dir=${OUTPUT_DIR:-$out_root/$run_name}
for path in "$env_bin/deepspeed" "$checkpoint" "$slt_ckpt" \
    "$vae_root/net_best_hand.pth" "$vae_root/meta/mean.npy" "$vae_root/meta/std.npy" \
    "$latent_stats/latent_mean.npy" "$latent_stats/latent_std.npy"; do
    [[ -s "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

command=("$env_bin/deepspeed" --include "localhost:$gpus"
    --master_port "${MASTER_PORT:-29517}" train_unified.py
    --train_stage "$stage" --dataset "$dataset" --task SLT
    --dtype fp32 --zero_stage 0
    --batch-size "${BATCH_SIZE:-8}" --gradient-accumulation-steps "${ACCUM_STEPS:-8}"
    --epochs "${EPOCHS:-40}" --opt AdamW --lr "$lr" --skip_nonfinite_loss
    --val_split test --resume auto --output_dir "$output_dir"
    --slt_init "$slt_ckpt" --finetune "$checkpoint"
    --vae_checkpoint "$vae_root/net_best_hand.pth"
    --vae_mean_path "$vae_root/meta/mean.npy" --vae_std_path "$vae_root/meta/std.npy"
    --latent_mean_path "$latent_stats/latent_mean.npy"
    --latent_std_path "$latent_stats/latent_std.npy"
    --use_hyperbolic --init_c 1.0 --hyp_dim 256 --hyp_lr 0.001 --hyp_stabilize 100
    --hyp_text_emb_src decoder --hyp_text_cmp token --alpha 0.7 --label_smoothing_hyp 0.2
    --use_wandb --wandb_project uni-sign-unified --wandb_run_name "$run_name"
    "${extra[@]}" "$@")
echo "Geo-Sign Stage $stage: $dataset; physical GPUs: $gpus"
if [[ "$merged" == 1 ]]; then
    echo "Training: OpenASL + How2Sign, concatenated natural sampling; test metrics kept separate"
    echo "Checkpoint selection: $dataset test BLEU-4 / hand DTW"
fi
echo "Initialize: $checkpoint; output: $output_dir"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
    exit 0
fi
"$env_bin/python" -B -c 'import geoopt; print("geoopt", geoopt.__version__)'
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
exec "${command[@]}"
