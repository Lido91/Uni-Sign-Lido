#!/usr/bin/env bash
# Current-path runner for the shared-mT5 SLT + text-to-2D-pose implementation.
# Usage: bash script/run_bidirectional_checked.sh 1|2|3|eval|ceilings|gen
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode=${1:?usage: run_bidirectional_checked.sh 1|2|3|eval|ceilings|gen}
case "$mode" in 1|2|3|eval|ceilings|gen) ;; *) echo "Unknown mode: $mode" >&2; exit 2 ;; esac

python=${UNISIGN_PYTHON:-/home/student/hwu/miniconda3/envs/Uni-Sign/bin/python}
deepspeed="$(dirname "$python")/deepspeed"
vae_root=${VAE_ROOT:-/data/hwu/momadiff_repo/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d}
out_root=${OUTPUT_ROOT:-/data/hwu/Uni_sign_repo/out}
latent_stats=${LATENT_STATS:-/data/hwu/Uni_sign_repo/out/latent_stats}
slt_ckpt=${SLT_CHECKPOINT:-/data/hwu/Uni_sign_repo/out/how2sign_pose_only/how2sign_pose_only_slt.pth}
stage1_dir=${STAGE1_DIR:-$out_root/unified_stage1_openasl}
stage2_dir=${STAGE2_DIR:-$out_root/unified_stage2_openasl}
stage3_dir=${STAGE3_DIR:-$out_root/unified_stage3_h2s}
gpus=${GPUS:-2,3}
port=${MASTER_PORT:-29515}
eval_gpu=${EVAL_GPU:-0}
val_split=${VAL_SPLIT:-test}
case "$val_split" in dev|test) ;; *) echo "VAL_SPLIT must be dev or test" >&2; exit 2 ;; esac
resume=${RESUME:-auto}
case "$resume" in auto|off) ;; *) echo "RESUME must be auto or off" >&2; exit 2 ;; esac

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/unisign-bidirectional-matplotlib}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/unisign-bidirectional-triton}
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

for path in "$python" "$deepspeed" "$slt_ckpt" \
    "$vae_root/net_best_hand.pth" "$vae_root/meta/mean.npy" "$vae_root/meta/std.npy" \
    "$latent_stats/latent_mean.npy" "$latent_stats/latent_std.npy"; do
    if [[ ! -s "$path" ]]; then echo "Missing required file: $path" >&2; exit 1; fi
done

vae_args=(
    --vae_checkpoint "$vae_root/net_best_hand.pth"
    --vae_mean_path "$vae_root/meta/mean.npy" --vae_std_path "$vae_root/meta/std.npy"
    --latent_mean_path "$latent_stats/latent_mean.npy" --latent_std_path "$latent_stats/latent_std.npy"
)
train_args=(--task SLT --dtype fp32 --batch-size 8 --opt AdamW --skip_nonfinite_loss --slt_init "$slt_ckpt" --val_split "$val_split" --resume "$resume")
wandb_args=()
if [[ ${USE_WANDB:-1} == 1 ]]; then wandb_args=(--use_wandb --wandb_project uni-sign-unified); fi

case "$mode" in
    1)
        "$deepspeed" --include "localhost:$gpus" --master_port "$port" train_unified.py \
            "${train_args[@]}" "${vae_args[@]}" "${wandb_args[@]}" \
            --train_stage 1 --dataset OpenASL --gradient-accumulation-steps 4 \
            --epochs 10 --lr 1e-4 --output_dir "$stage1_dir" --wandb_run_name unified_stage1_openasl
        ;;
    2|3)
        dataset=OpenASL; epochs=20; lr=1e-4; accumulation=4
        previous="$stage1_dir/best_checkpoint.pth"; output="$stage2_dir"; run_name=unified_stage2_openasl
        if [[ "$mode" == 3 ]]; then
            dataset=How2Sign; lr=3e-5; accumulation=1
            previous="$stage2_dir/best_checkpoint.pth"; output="$stage3_dir"; run_name=unified_stage3_h2s
        fi
        if [[ ! -s "$previous" ]]; then echo "Missing previous-stage checkpoint: $previous" >&2; exit 1; fi
        "$deepspeed" --include "localhost:$gpus" --master_port "$port" train_unified.py \
            "${train_args[@]}" "${vae_args[@]}" "${wandb_args[@]}" \
            --train_stage "$mode" --dataset "$dataset" --gradient-accumulation-steps "$accumulation" \
            --epochs "$epochs" --lr "$lr" --output_dir "$output" --finetune "$previous" --wandb_run_name "$run_name"
        ;;
    eval)
        checkpoint="$stage3_dir/best_checkpoint.pth"
        if [[ ! -s "$checkpoint" ]]; then echo "Missing stage-3 checkpoint: $checkpoint" >&2; exit 1; fi
        "$deepspeed" --include "localhost:$eval_gpu" --master_port "$port" train_unified.py \
            "${train_args[@]}" "${vae_args[@]}" --train_stage 3 --dataset How2Sign \
            --finetune "$checkpoint" --output_dir "$out_root/eval_unified_h2s" --eval --skip-bleurt
        ;;
    ceilings|gen)
        backtrans_args=(--dataset How2Sign --split test --task SLT --dtype fp32 --batch-size 8 --max_length 400
            --slt_checkpoint "$slt_ckpt" --output_dir "$out_root/backtrans_eval")
        if [[ "$mode" == ceilings ]]; then
            for ceiling in gt gt_conf1 vae_recon; do
                CUDA_VISIBLE_DEVICES="$eval_gpu" "$python" back_translation_eval.py \
                    "${backtrans_args[@]}" "${vae_args[@]}" --mode "$ceiling"
            done
        else
            checkpoint=${GEN_CHECKPOINT:-$stage3_dir/best_slg_checkpoint.pth}
            if [[ ! -s "$checkpoint" ]]; then echo "Missing SLG checkpoint: $checkpoint" >&2; exit 1; fi
            CUDA_VISIBLE_DEVICES="$eval_gpu" "$python" back_translation_eval.py \
                "${backtrans_args[@]}" "${vae_args[@]}" --mode gen --unified_checkpoint "$checkpoint" \
                --trans_infer_timesteps 9 --ddim_steps 100 --guidance_param 3.0
        fi
        ;;
esac
