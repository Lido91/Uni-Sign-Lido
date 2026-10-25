#!/bin/bash
# Back-translate every OpenASL pose source (GT FK, DETO reconstruction, all
# generators) with one frozen OpenASL SLT judge, then print a BLEU/ROUGE
# table re-scored from the saved test_tmp_*.txt (no logs). Every source holds
# the same 896/976 OpenASL test clips (SMPL-X FK -> 2D, project_fk_to_rtmpose2d).
# Finished runs are skipped; delete <out_root>/<label> to redo one.
#   bash script/eval_openasl_backtrans_all.sh [judge.pth] [gpu] [master_port]
# Default judge: the FK-finetuned one (script/finetune_openasl_fkpose.sh); pass
# /data/hwu/Uni_sign_repo/out/openasl_pose_pretrain/openasl_pose_only_slt.pth
# for the original RTMPose-trained judge.
set -e

judge=${1:-/data/hwu/Uni_sign_repo/out/finetune_openasl_fkpose/best_checkpoint.pth}
gpu=${2:-0}
port=${3:-29513}

out_root=/data/hwu/Uni_sign_repo/out/backtrans_openasl/$(basename $(dirname $judge))
soke=/home/student/hwu/Workplace/SOKE
mgpt=/home/student/hwu/Workplace/MotionGPT/results/mgpt
deto_epochs=/data/hwu/serv_checkpoints/experiments/mgpt/DETO_RVQ_Hierarchical_T5_Online_LlamaGen_openasl/results_val

# label=pose_dir, in table order
runs="GT_FK=$soke/data/OpenASL/POSE_RTMPOSE2D_FK/test
DETO_recon=$soke/results/mgpt/DETO_OpenASL/rtmpose2d
SOKE=$soke/results/mgpt/SOKE_OpenASL/rtmpose2d
MotionGPT=$mgpt/MotionGPT_OpenASL/rtmpose2d
DETO_old=$mgpt/DETO_RVQ_Hierarchical_T5_Online_LlamaGen_openasl/rtmpose2d
DETO_wb3sem=$mgpt/DETO_RVQ_wholebody_3_semantic_h2s_openasl/rtmpose2d"
for d in $(ls -d $deto_epochs/epoch_*/rtmpose2d | sort -V); do
  runs="$runs
DETO_$(basename $(dirname $d))=$d"
done

summary=""
for run in $runs; do
  label=${run%%=*}
  pose_dir=${run#*=}
  out=$out_root/$label
  summary="$summary $label=$out"
  if [ -s $out/test_tmp_pres.txt ]; then
    echo "===== $label: already done, skipping"
    continue
  fi
  if [ ! -d $pose_dir ]; then
    echo "===== $label: missing $pose_dir, skipping"
    continue
  fi
  mkdir -p $out
  echo "===== $label"
  deepspeed --include localhost:$gpu --master_port $port fine_tuning.py \
    --batch-size 64 --dataset OpenASL --task SLT --eval --skip-bleurt \
    --finetune $judge \
    --output_dir $out \
    --pose_root_override $pose_dir
done

echo -e "\njudge: $judge"
python - $summary <<'EOF'
import contextlib
import io
import os
import sys

from SLRT_metrics import translation_performance


def read(path, tag):
    with open(path) as f:
        return [line.rstrip("\n").split(f", {tag}: ", 1)[1] for line in f]


print(f"{'run':<16}{'n':>5}{'B1':>7}{'B2':>7}{'B3':>7}{'B4':>7}{'ROUGE':>7}")
for arg in sys.argv[1:]:
    label, out = arg.split("=", 1)
    if not os.path.isfile(os.path.join(out, "test_tmp_pres.txt")):
        print(f"{label:<16}  (no results)")
        continue
    hyp = read(os.path.join(out, "test_tmp_pres.txt"), "prediction")
    ref = read(os.path.join(out, "test_tmp_refs.txt"), "ground-truth")
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, rouge = translation_performance(ref, hyp)
    print(f"{label:<16}{len(hyp):>5}"
          + "".join(f"{bleu[f'bleu{n}']:>7.2f}" for n in range(1, 5))
          + f"{rouge:>7.2f}")
EOF
