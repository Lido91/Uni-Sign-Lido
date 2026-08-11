ckpt_path=out/openasl_pose_pretrain/openasl_pose_only_slt.pth

# single gpu inference
# RGB-pose setting
# ckpt_path=out/how2sign_pose_only_direct/best_checkpoint.pth

deepspeed --include localhost:2 --master_port 29512 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir out/test \
  --finetune "$ckpt_path" \
  --dataset How2Sign \
  --task SLT \
  --eval \
  --skip_bleurt
   # --rgb_support

# # pose-only setting
#deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
#   --batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --epochs 20 \
#   --opt AdamW \
#   --lr 3e-4 \
#   --output_dir out/test \
#   --finetune $ckpt_path \
#   --dataset CSL_Daily \
#   --task SLT \
#   --eval \
