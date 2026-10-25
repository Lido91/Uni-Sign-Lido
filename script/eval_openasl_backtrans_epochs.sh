#!/bin/bash
# Back-translate DETO-RVQ OpenASL generations from every saved epoch with the
# frozen OpenASL pose-only SLT judge (fine_tuning.py --eval), then print one
# BLEU/ROUGE row per epoch plus the SOKE / MotionGPT / earlier DETO baselines,
# re-scored from the saved test_tmp_*.txt (no logs).
# Pose folders come from SOKE project_fk_to_rtmpose2d.py --motion_dir; they
# hold 896/976 OpenASL test clips (dev is empty and skipped).
#   bash script/eval_openasl_backtrans_epochs.sh [gpu] [master_port]
set -e

gpu=${1:-3}
port=${2:-29513}



judge=/data/hwu/Uni_sign_repo/out/finetune_openasl_fkpose/best_checkpoint.pth
gen_root=/data/hwu/serv_checkpoints/experiments/mgpt/DETO_RVQ_Hierarchical_T5_Online_LlamaGen_openasl/results_val
out_root=/data/hwu/Uni_sign_repo/out/eval_openasl_deto_rvq_epochs

for pose_dir in $(ls -d $gen_root/epoch_*/rtmpose2d | sort -V); do
  epoch=$(basename $(dirname $pose_dir))
  out=$out_root/$epoch
  mkdir -p $out
  echo "===== $epoch"
  deepspeed --include localhost:$gpu --master_port $port fine_tuning.py \
    --batch-size 64 --dataset OpenASL --task SLT --eval --skip-bleurt \
    --finetune $judge \
    --output_dir $out \
    --pose_root_override $pose_dir
done

# Summary rows: every epoch, then the baselines judged on the same 896 clips
res=/data/hwu/Uni_sign_repo/out
runs=""
for d in $(ls -d $out_root/epoch_* | sort -V); do
  runs="$runs $(basename $d)=$d"
done
runs="$runs SOKE=$res/eval_openasl_on_soke_rtmpose2d"
runs="$runs MotionGPT=$res/eval_openasl_on_motiongpt_rtmpose2d"
runs="$runs DETO_old=$res/eval_openasl_on_deto_rvq_rtmpose2d"

python - $runs <<'EOF'
import contextlib
import io
import os
import sys

from SLRT_metrics import translation_performance


def read(path, tag):
    with open(path) as f:
        return [line.rstrip("\n").split(f", {tag}: ", 1)[1] for line in f]


print(f"\n{'run':<12}{'B1':>7}{'B2':>7}{'B3':>7}{'B4':>7}{'ROUGE':>7}")
for arg in sys.argv[1:]:
    label, out = arg.split("=", 1)
    if not os.path.isfile(os.path.join(out, "test_tmp_pres.txt")):
        print(f"{label:<12}  (no results in {out})")
        continue
    hyp = read(os.path.join(out, "test_tmp_pres.txt"), "prediction")
    ref = read(os.path.join(out, "test_tmp_refs.txt"), "ground-truth")
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, rouge = translation_performance(ref, hyp)
    print(f"{label:<12}"
          + "".join(f"{bleu[f'bleu{n}']:>7.2f}" for n in range(1, 5))
          + f"{rouge:>7.2f}")
EOF
