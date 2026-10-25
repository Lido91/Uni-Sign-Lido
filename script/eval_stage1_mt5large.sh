# mT5-Large full fine-tuning on OpenASL. Keep checkpoints separate from the
# incompatible mT5-base and mT5-XL runs.
output_dir=/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl_mt5large
mt5_model=google/mt5-large
ckpt_path=/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl_mt5large/best_checkpoint.pth


export HF_HOME=/data/hwu/hf_cache
export HF_HUB_CACHE=/data/hwu/hf_cache/hub

deepspeed --include localhost:0,1,3 --master_port 29511 fine_tuning.py \
   --batch-size 8\
   --gradient-accumulation-steps 32 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir "$output_dir" \
   --dataset OpenASL \
   --task SLT \
   --llm_path "$mt5_model" \
   --dtype bf16 \
   --zero_stage 2 \
   --llm_grad_ckpt \
   --compute_fp32_loss \
   --skip_nonfinite_loss \
   --finetune "$ckpt_path" \
   --eval 




# #!/usr/bin/env bash
# set -euo pipefail

# ckpt_path=/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl/best_checkpoint.pth
# output_dir=out/eval_stage1_pretraining_openasl

# if [[ ! -f "$ckpt_path" ]]; then
#   echo "Checkpoint not found: $ckpt_path" >&2
#   exit 1
# fi

# # Single-GPU, pose-only evaluation on the OpenASL dev and test splits.
# deepspeed --include localhost:2 --master_port 29512 fine_tuning.py \
#   --batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --epochs 20 \
#   --opt AdamW \
#   --lr 3e-4 \
#   --output_dir "$output_dir" \
#   --finetune "$ckpt_path" \
#   --dataset OpenASL \
#   --task SLT \
#   --dtype fp32 \
#   --eval \
#   --skip_bleurt

