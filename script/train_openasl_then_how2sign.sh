# output_dir=out/stage1_pretraining_openasl

# deepspeed --include localhost:0,1,3 --master_port 29511 fine_tuning.py \
#    --batch-size 16 \
#    --gradient-accumulation-steps 8 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --output_dir "$output_dir" \
#    --dataset OpenASL \
#    --task SLT \
#    --dtype fp32 \
#    --eval-on-how2sign-test




output_dir=out/stage3_finetuning_openasl_h2s

# RGB-pose setting
ckpt_path=out/stage1_pretraining_openasl/best_checkpoint.pth

deepspeed --include localhost:0,1,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset How2Sign \
   --dtype fp32 \
  --task SLT
