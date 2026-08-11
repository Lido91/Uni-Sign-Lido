# Finetune the How2Sign pose-only SLT model on FK-projected 2D poses
# (SOKE 3D -> orthographic 2D, aligned to the GT clip timeline).
# Pose source: project_fk_to_rtmpose2d.py output; {} is replaced by train/test.

output_dir=/data/hwu/Uni_sign_repo/out/finetune_h2s_fkpose
ckpt_path=/data/hwu/Uni_sign_repo/out/how2sign_pose_only/how2sign_pose_only_slt.pth
fk_pose_root='/home/student/hwu/Workplace/SOKE/data/How2Sign/POSE_RTMPOSE2D_FK/{}'

mkdir -p $output_dir

deepspeed --include localhost:0,1 --master_port 29512 fine_tuning.py \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --dataset How2Sign \
  --task SLT \
  --finetune $ckpt_path \
  --output_dir $output_dir \
  --pose_root_override "$fk_pose_root" \
  2>&1 | tee $output_dir/train.log

# After training:
# ① eval best checkpoint on the projected test set (in-domain)
#deepspeed --include localhost:0 --master_port 29513 fine_tuning.py \
#  --batch-size 8 --dataset How2Sign --task SLT --eval \
#  --finetune $output_dir/best_checkpoint.pth \
#  --output_dir out/eval_h2s_fkpose \
#  --pose_root_override "$fk_pose_root"
#
# ② eval best checkpoint on the real RTMPose GT test set (cross-domain)
#deepspeed --include localhost:0 --master_port 29513 fine_tuning.py \
#  --batch-size 8 --dataset How2Sign --task SLT --eval \
#  --finetune $output_dir/best_checkpoint.pth \
#  --output_dir out/eval_h2s_fkpose_on_gt
