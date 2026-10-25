#!/bin/bash
# Finetune the official How2Sign pose-only SLT model on DETO-RVQ wholebody
# tokenizer reconstructions (DETO_RVQ_wholebody_3_semantic_tokfull), projected
# to 2D by SOKE project_fk_to_rtmpose2d.py. The conversion keeps the per-rank
# layout (rtmpose2d/{train,test}_rank_N), so the ranks are first symlinked
# into rtmpose2d/{train,test} for --pose_root_override. How2Sign has no dev
# split, so best_checkpoint.pth is selected on test BLEU-4.
#   bash script/finetune_h2s_tokfull.sh [gpus]   # default 0,1
set -e

gpus=${1:-0,1}

pose_root=/home/student/hwu/Workplace/MotionGPT/results/mgpt/DETO_RVQ_wholebody_3_semantic_tokfull/rtmpose2d
ckpt_path=/data/hwu/Uni_sign_repo/out/how2sign_pose_only/how2sign_pose_only_slt.pth
output_dir=/data/hwu/Uni_sign_repo/out/finetune_h2s_tokfull

for split in train test; do
  mkdir -p $pose_root/$split
  find $pose_root/${split}_rank_* -name '*.pkl' -exec ln -sf -t $pose_root/$split {} +
  echo "$split: $(ls $pose_root/$split | wc -l) pose pkls"
done

mkdir -p $output_dir

deepspeed --include localhost:$gpus --master_port 29516 fine_tuning.py \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --dataset How2Sign \
  --task SLT \
  --finetune $ckpt_path \
  --output_dir $output_dir \
  --save-last-only \
  --pose_root_override "$pose_root/{}"
