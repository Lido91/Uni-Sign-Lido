#!/bin/bash
# Dual-stream stats-SignSMD pipeline:
#   Stage A: OpenASL task training with NO Uni-Sign task checkpoint.
#            The pose/ST-GCN path starts randomly, while mT5 still starts from
#            ./pretrained_weight/mt5-base (this is not random mT5 training).
#   Stage B: How2Sign finetuning from Stage A best_checkpoint.pth.
#   Stage C: How2Sign normal and SMD-masked evaluation.
#
# Usage:
#   bash script/train_openasl_then_h2s_signsmd_dual_stats.sh            # all
#   bash script/train_openasl_then_h2s_signsmd_dual_stats.sh pretrain   # Stage A
#   bash script/train_openasl_then_h2s_signsmd_dual_stats.sh finetune   # Stage B
#   bash script/train_openasl_then_h2s_signsmd_dual_stats.sh eval       # Stage C
#
# OpenASL stats texts must be complete before Stage A. Generation resumes
# existing partial output by default:
#   python generate_signsmd2d.py --dataset OpenASL \
#     --splits train dev test --variant stats --max_length 256
#
# Optional overrides:
#   GPUS=localhost:0,1,2,3 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 \
#   SIGNSMD_MAX_TOKENS=512 RUN_BLEURT=0 \
#     bash script/train_openasl_then_h2s_signsmd_dual_stats.sh all
# Output directories can be overridden with PRETRAIN_DIR, FINETUNE_DIR, and
# EVAL_DIR. Existing training outputs are rejected unless
# ALLOW_EXISTING_OUTPUT=1 is set explicitly (this is not an exact resume).

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

stage=${1:-all}
case "$stage" in
  pretrain|finetune|eval|all) ;;
  *)
    echo "Usage: $0 [pretrain|finetune|eval|all]" >&2
    exit 2
    ;;
esac

gpus=${GPUS:-localhost:0,1,2,3}
port=${MASTER_PORT:-29518}
batch_size=${BATCH_SIZE:-4}
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}
signsmd_max_tokens=${SIGNSMD_MAX_TOKENS:-512}
run_bleurt=${RUN_BLEURT:-0}
allow_existing_output=${ALLOW_EXISTING_OUTPUT:-0}

openasl_signsmd_root=data/OpenASL/signsmd2d_stats_v0_1/texts
h2s_signsmd_root=data/How2Sign/signsmd2d_stats_v0_1/texts

# Put the large FP32 checkpoints on /data rather than the nearly-full repo disk.
pretrain_dir=${PRETRAIN_DIR:-/data/hwu/Uni_sign_repo/out/signsmd_dual_stats_openasl_from0}
finetune_dir=${FINETUNE_DIR:-/data/hwu/Uni_sign_repo/out/signsmd_dual_stats_openasl_from0_h2s}
eval_dir=${EVAL_DIR:-/data/hwu/Uni_sign_repo/out/eval_signsmd_dual_stats_openasl_from0_h2s}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "deepspeed is not available; activate the training environment first." >&2
  exit 1
fi

check_stats_coverage() {
  local dataset=$1
  shift
  python - "$dataset" "$@" <<'PY'
import gzip
import pickle
import sys
from pathlib import Path

dataset, *splits = sys.argv[1:]
root = Path("data") / dataset / "signsmd2d_stats_v0_1" / "texts"
if not root.is_dir():
    raise SystemExit(f"missing stats text directory: {root}")

missing_total = 0
for split in splits:
    label_path = Path("data") / dataset / f"labels.{split}"
    with gzip.open(label_path, "rb") as f:
        labels = pickle.load(f)
    missing = []
    for sample in labels.values():
        video_path = sample["video_path"]
        text_path = root / video_path.replace(".mp4", ".txt")
        if not text_path.is_file():
            missing.append(video_path)
    print(
        f"[{dataset}/{split}] stats coverage: "
        f"{len(labels) - len(missing)}/{len(labels)}"
    )
    if missing:
        print(f"  first missing examples: {missing[:3]}", file=sys.stderr)
        missing_total += len(missing)

if missing_total:
    raise SystemExit(
        f"{dataset} stats corpus is incomplete: "
        f"{missing_total} label rows have no text"
    )
PY
}

require_fresh_training_dir() {
  local output_dir=$1
  if [ "$allow_existing_output" = 1 ]; then
    return
  fi
  if [ -e "$output_dir/best_checkpoint.pth" ] || \
     [ -e "$output_dir/last_checkpoint.pth" ] || \
     [ -e "$output_dir/log.txt" ]; then
    echo "training output already exists: $output_dir" >&2
    echo "This launcher does not perform an exact resume." >&2
    echo "Choose a new output directory, or set ALLOW_EXISTING_OUTPUT=1 to restart/overwrite intentionally." >&2
    exit 1
  fi
}

if [ "$stage" = pretrain ] || [ "$stage" = all ]; then
  check_stats_coverage OpenASL train dev test
  require_fresh_training_dir "$pretrain_dir"

  # No --finetune here: pose/ST-GCN/task layers start randomly; mT5 is loaded
  # by models.Uni_Sign from ./pretrained_weight/mt5-base.
  deepspeed --include "$gpus" --master_port "$port" fine_tuning_signsmd.py \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$grad_accum_steps" \
    --epochs 20 \
    --opt AdamW \
    --lr 2e-4 \
    --fusion dual \
    --output_dir "$pretrain_dir" \
    --dataset OpenASL \
    --task SLT \
    --dtype fp32 \
    --save-last-only \
    --skip_nonfinite_loss \
    --frame_sampling linear \
    --max_length 256 \
    --signsmd_max_tokens "$signsmd_max_tokens" \
    --signsmd_text_root "$openasl_signsmd_root"
fi

if [ "$stage" = finetune ] || [ "$stage" = all ]; then
  check_stats_coverage How2Sign train test
  if [ ! -f "$pretrain_dir/best_checkpoint.pth" ]; then
    echo "missing Stage A checkpoint: $pretrain_dir/best_checkpoint.pth" >&2
    exit 1
  fi
  require_fresh_training_dir "$finetune_dir"

  deepspeed --include "$gpus" --master_port "$port" fine_tuning_signsmd.py \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$grad_accum_steps" \
    --epochs 20 \
    --opt AdamW \
    --lr 1e-4 \
    --fusion dual \
    --output_dir "$finetune_dir" \
    --finetune "$pretrain_dir/best_checkpoint.pth" \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --save-last-only \
    --skip_nonfinite_loss \
    --frame_sampling linear \
    --max_length 256 \
    --signsmd_max_tokens "$signsmd_max_tokens" \
    --signsmd_text_root "$h2s_signsmd_root"
fi

if [ "$stage" = eval ] || [ "$stage" = all ]; then
  check_stats_coverage How2Sign train test
  if [ ! -f "$finetune_dir/best_checkpoint.pth" ]; then
    echo "missing Stage B checkpoint: $finetune_dir/best_checkpoint.pth" >&2
    exit 1
  fi

  bleurt_args=(--skip_bleurt)
  if [ "$run_bleurt" = 1 ]; then
    bleurt_args=()
  fi

  deepspeed --include localhost:0 --master_port "$port" fine_tuning_signsmd.py \
    --batch-size "$batch_size" \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --fusion dual \
    --finetune "$finetune_dir/best_checkpoint.pth" \
    --output_dir "$eval_dir" \
    --eval \
    --frame_sampling linear \
    --max_length 256 \
    --signsmd_max_tokens "$signsmd_max_tokens" \
    --signsmd_text_root "$h2s_signsmd_root" \
    "${bleurt_args[@]}"

  deepspeed --include localhost:0 --master_port "$port" fine_tuning_signsmd.py \
    --batch-size "$batch_size" \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --fusion dual \
    --finetune "$finetune_dir/best_checkpoint.pth" \
    --output_dir "${eval_dir}_mask_smd" \
    --eval \
    --eval_mask_smd \
    --frame_sampling linear \
    --max_length 256 \
    --signsmd_max_tokens "$signsmd_max_tokens" \
    --signsmd_text_root "$h2s_signsmd_root" \
    "${bleurt_args[@]}"
fi
