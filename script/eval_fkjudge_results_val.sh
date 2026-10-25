#!/bin/bash
# Back-translate every epoch of a MotionGPT/DETO results_val folder with the
# FK-finetuned pose-only SLT judges, then print one BLEU/ROUGE table per
# dataset, re-scored from the saved test_tmp_*.txt (no logs).
# Each results_val/epoch_N/rtmpose2d comes from SOKE project_fk_to_rtmpose2d.py
# (--motion_dir); epochs whose conversion is missing/incomplete are skipped,
# and finished evals are skipped too, so just rerun after converting more.
#   bash script/eval_fkjudge_results_val.sh [results_val] [gpus] [base_port]
#   bash script/eval_fkjudge_results_val.sh "" 0,1,3   # default dir, 3 GPUs
set -e
set -o pipefail

results_val=${1:-/data/hwu/serv_checkpoints/experiments/mgpt/DETO_RVQ_Hierarchical_T5_Online_LlamaGen_h2s_openasl_joint/results_val}
gpus=${2:-0}
port=${3:-29513}

# Joint H2S+OpenASL models hold both test sets; for an OpenASL-only model use
# datasets="OpenASL"
datasets="OpenASL How2Sign"
judge_OpenASL=/data/hwu/Uni_sign_repo/out/finetune_openasl_fkpose/best_checkpoint.pth
judge_How2Sign=/data/hwu/Uni_sign_repo/out/finetune_h2s_fkpose/best_checkpoint.pth

out_root=/data/hwu/Uni_sign_repo/out/eval_fkjudge/$(basename $(dirname $results_val))

jobs=()
for epoch_dir in $(ls -d $results_val/epoch_*/ | sort -V); do
  epoch=$(basename $epoch_dir)
  pose_dir=${epoch_dir}rtmpose2d
  # the meta (input_count) is written before conversion starts
  if [ ! -f $pose_dir/rtmpose2d_meta.json ]; then
    echo "===== $epoch: not converted yet, skipping"
    continue
  fi
  want=$(grep -o '"input_count": [0-9]*' $pose_dir/rtmpose2d_meta.json | grep -o '[0-9]*$')
  have=$(ls $pose_dir | grep -c '\.pkl$')
  if [ "$have" != "$want" ]; then
    echo "===== $epoch: $have/$want converted, skipping"
    continue
  fi
  for ds in $datasets; do
    out=$out_root/$ds/$epoch
    if [ -s $out/test_tmp_pres.txt ]; then
      echo "===== $epoch $ds: already done, skipping"
      continue
    fi
    jobs+=("$epoch $ds $pose_dir $out")
  done
done

# One eval cannot use several GPUs (fine_tuning.py --eval runs on rank 0 only
# and writes test_tmp_*.txt only when world size is 1), so run evals in
# parallel instead: worker k takes every n-th job on GPU k, own master port.
IFS=, read -ra gpu_list <<< "$gpus"
n_gpu=${#gpu_list[@]}
echo "${#jobs[@]} evals on GPU(s) $gpus"
pids=""
for k in "${!gpu_list[@]}"; do
  gpu=${gpu_list[$k]}
  (
    fail=0
    for ((i = k; i < ${#jobs[@]}; i += n_gpu)); do
      read epoch ds pose_dir out <<< "${jobs[$i]}"
      judge_var=judge_$ds
      mkdir -p $out
      echo "[gpu$gpu] ===== $epoch $ds"
      deepspeed --include localhost:$gpu --master_port $((port + k)) fine_tuning.py \
        --batch-size 64 --dataset $ds --task SLT --eval --skip-bleurt \
        --finetune ${!judge_var} \
        --output_dir $out \
        --pose_root_override $pose_dir 2>&1 | sed -u "s/^/[gpu$gpu] /" \
        || { echo "[gpu$gpu] FAILED $epoch $ds"; fail=1; }
    done
    exit $fail
  ) &
  pids="$pids $!"
done
failed=0
for pid in $pids; do
  wait $pid || failed=1
done
[ $failed = 0 ] || echo "WARNING: some evals failed (see FAILED above); rerun to retry them"

python - $out_root $datasets <<'EOF'
import contextlib
import io
import os
import sys

from SLRT_metrics import translation_performance


def read(path, tag):
    with open(path) as f:
        return [line.rstrip("\n").split(f", {tag}: ", 1)[1] for line in f]


out_root, datasets = sys.argv[1], sys.argv[2:]
for ds in datasets:
    root = os.path.join(out_root, ds)
    if not os.path.isdir(root):
        continue
    epochs = sorted(os.listdir(root), key=lambda e: int(e.split("_")[-1]))
    print(f"\n{ds} ({root})")
    print(f"{'epoch':<12}{'n':>6}{'B1':>7}{'B2':>7}{'B3':>7}{'B4':>7}{'ROUGE':>7}")
    for epoch in epochs:
        out = os.path.join(root, epoch)
        if not os.path.isfile(os.path.join(out, "test_tmp_pres.txt")):
            print(f"{epoch:<12}  (no results)")
            continue
        hyp = read(os.path.join(out, "test_tmp_pres.txt"), "prediction")
        ref = read(os.path.join(out, "test_tmp_refs.txt"), "ground-truth")
        with contextlib.redirect_stdout(io.StringIO()):
            bleu, rouge = translation_performance(ref, hyp)
        print(f"{epoch:<12}{len(hyp):>6}"
              + "".join(f"{bleu[f'bleu{n}']:>7.2f}" for n in range(1, 5))
              + f"{rouge:>7.2f}")
EOF
