# Finetune the OpenASL pose-only SLT model on FK-projected 2D poses
# (SOKE OpenASL SMPL-X -> orthographic 2D, same projection as the generated
# poses judged by script/eval_openasl_backtrans_epochs.sh).
# Pose source: SOKE project_fk_to_rtmpose2d.py output; {} -> train/test.
# There is no FK dev split, so fine_tuning.py selects best_checkpoint.pth on
# test BLEU-4.
#
# Prerequisite (once, in the SOKE repo / soke env, ~1 h):
#   cd ~/Workplace/SOKE
#   CUDA_VISIBLE_DEVICES=0 python -m project_fk_to_rtmpose2d \
#       --splits train --data_root data/OpenASL
#   bash script/finetune_openasl_fkpose.sh [gpus]   # default 0,1
set -e

gpus=${1:-0,1}

output_dir=/data/hwu/Uni_sign_repo/out/finetune_openasl_fkpose
ckpt_path=/data/hwu/Uni_sign_repo/out/openasl_pose_pretrain/openasl_pose_only_slt.pth
fk_root=/home/student/hwu/Workplace/SOKE/data/OpenASL/POSE_RTMPOSE2D_FK

for split in train test; do
  [ -d $fk_root/$split ] || { echo "missing $fk_root/$split -- run the FK prerequisite above"; exit 1; }
done

mkdir -p $output_dir

deepspeed --include localhost:$gpus --master_port 29514 fine_tuning.py \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --dataset OpenASL \
  --task SLT \
  --finetune $ckpt_path \
  --output_dir $output_dir \
  --save-last-only \
  --pose_root_override "$fk_root/{}"

# After training, judge generated poses with the FK-adapted model, e.g.:
#deepspeed --include localhost:0 --master_port 29513 fine_tuning.py \
#  --batch-size 64 --dataset OpenASL --task SLT --eval --skip-bleurt \
#  --finetune $output_dir/best_checkpoint.pth \
#  --output_dir /data/hwu/Uni_sign_repo/out/eval_openasl_fkjudge_on_<model> \
#  --pose_root_override <model>/rtmpose2d
