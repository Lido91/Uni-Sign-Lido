# Pose-only How2Sign fine-tuning from the OpenASL mT5-XL Stage 1 checkpoint.
# Keep this output separate from checkpoints produced with mT5-base.
output_dir=out/stage3_finetuning_openasl_h2s_mt5xl
ckpt_path=out/stage1_pretraining_openasl_mt5xl/best_checkpoint.pth
mt5_model=google/mt5-xl

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir "$output_dir" \
  --finetune "$ckpt_path" \
  --dataset How2Sign \
  --task SLT \
  --llm_path "$mt5_model" \
  --dtype bf16 \
  --llm_grad_ckpt \
  --compute_fp32_loss
