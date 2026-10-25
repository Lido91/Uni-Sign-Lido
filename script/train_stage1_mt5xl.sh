# mT5-XL full fine-tuning. Keep its checkpoints separate from the incompatible
# mT5-base run. The Hugging Face weights are downloaded on first use.
output_dir=out/stage1_pretraining_openasl_mt5xl
mt5_model=google/mt5-xl

export HF_HOME=/data/hwu/hf_cache
export HF_HUB_CACHE=/data/hwu/hf_cache/hub


deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
   --batch-size 1 \
   --gradient-accumulation-steps 128 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir "$output_dir" \
   --dataset OpenASL \
   --task SLT \
   --llm_path "$mt5_model" \
   --dtype bf16 \
   --llm_grad_ckpt \
   --compute_fp32_loss \
   --eval-on-how2sign-test
