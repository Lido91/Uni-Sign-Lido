# Pose-only How2Sign fine-tuning from the OpenASL mT5-Large Stage 1 checkpoint.
# Keep checkpoints separate from the incompatible mT5-base and mT5-XL runs.
output_dir=/data/hwu/Uni_sign_repo/out/stage3_finetuning_openasl_h2s_mt5large
ckpt_path=/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl_mt5large/best_checkpoint.pth
mt5_model=google/mt5-large

export HF_HOME=/data/hwu/hf_cache
export HF_HUB_CACHE=/data/hwu/hf_cache/hub

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir "$output_dir" \
  --finetune "$ckpt_path" \
  --dataset How2Sign \
  --task SLT \
  --llm_path "$mt5_model" \
  --dtype bf16 \
  --zero_stage 2 \
  --llm_grad_ckpt \
  --compute_fp32_loss \
  --skip_nonfinite_loss

