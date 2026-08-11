# OpenASL pose-only sign pre-training with Qwen3-4B-Instruct + LoRA.
# Plain PyTorch DDP (no deepspeed) — runs in the motion-smd `smd` conda env.
# Effective batch = 8 x 3 GPUs x 16 accum = 384 (same as the mT5 stage-1 run).
# Lower --batch-size (and raise accumulation) if OOM.
output_dir=/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl_qwen3_4b

CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 --master_port 29512 train_qwen.py \
   --batch-size 8 \
   --gradient-accumulation-steps 16 \
   --epochs 20 \
   --lr 3e-4 \
   --output_dir "$output_dir" \
   --dataset OpenASL \
   --task SLT \
   --dtype bf16 \
   --llm_grad_ckpt \
   --eval-on-how2sign-test
