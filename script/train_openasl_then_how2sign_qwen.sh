# How2Sign pose-only fine-tuning from the OpenASL Qwen pre-training checkpoint.
# Plain PyTorch DDP (no deepspeed) — runs in the motion-smd `smd` conda env.
# Effective batch = 4 x 3 GPUs x 2 accum = 24 (same as the mT5 How2Sign run).
output_dir=/data/hwu/Uni_sign_repo/out/stage3_finetuning_openasl_h2s_qwen3_4b

ckpt_path=/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl_qwen3_4b/best_checkpoint.pth

CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 --master_port 29512 train_qwen.py \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --epochs 20 \
  --lr 3e-4 \
  --output_dir "$output_dir" \
  --finetune "$ckpt_path" \
  --dataset How2Sign \
  --dtype bf16 \
  --llm_grad_ckpt \
  --save-last-only \
  --task SLT
