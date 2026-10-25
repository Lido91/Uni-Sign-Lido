# mT5-Large full fine-tuning on OpenASL. Keep checkpoints separate from the
# incompatible mT5-base and mT5-XL runs.
output_dir=/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl_mt5large
mt5_model=google/mt5-large

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
   --eval-on-how2sign-test
