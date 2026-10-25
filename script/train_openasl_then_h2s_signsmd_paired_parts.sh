#!/bin/bash
# Time-aligned part-paired SignSMD pipeline:
#   Stage A: OpenASL paired pretraining.
#   Stage B: How2Sign paired finetuning from Stage A best checkpoint.
#   Stage C: normal How2Sign evaluation + all-text-masked diagnostic.
#
# Usage:
#   bash script/train_openasl_then_h2s_signsmd_paired_parts.sh all
#   bash script/train_openasl_then_h2s_signsmd_paired_parts.sh pretrain
#   bash script/train_openasl_then_h2s_signsmd_paired_parts.sh finetune
#   bash script/train_openasl_then_h2s_signsmd_paired_parts.sh eval
#
# Required corpora (generation resumes by default):
#   python generate_signsmd2d.py --dataset OpenASL \
#     --splits train dev test --variant parts --max_length 256
#   python generate_signsmd2d.py --dataset How2Sign \
#     --splits train test --variant parts --max_length 256
#
# Common overrides:
#   GPUS=localhost:0,1,2,3 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 \
#   PART_MAX_TOKENS=128 TEXT_EVENT_DROPOUT=0.3 RUN_BLEURT=0 \
#     bash script/train_openasl_then_h2s_signsmd_paired_parts.sh all

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
port=${MASTER_PORT:-29528}
batch_size=${BATCH_SIZE:-4}
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}
part_max_tokens=${PART_MAX_TOKENS:-128}
text_event_dropout=${TEXT_EVENT_DROPOUT:-0.3}
run_bleurt=${RUN_BLEURT:-0}
allow_existing_output=${ALLOW_EXISTING_OUTPUT:-0}
python_bin=${PYTHON_BIN:-python}

signsmd_contract=signsmd2d_parts_v0_2
openasl_root=data/OpenASL/$signsmd_contract
h2s_root=data/How2Sign/$signsmd_contract
openasl_text_root=$openasl_root/texts
openasl_event_root=$openasl_root/events
h2s_text_root=$h2s_root/texts
h2s_event_root=$h2s_root/events

pretrain_dir=${PRETRAIN_DIR:-/data/hwu/Uni_sign_repo/out/signsmd_paired_parts_openasl_from0}
finetune_dir=${FINETUNE_DIR:-/data/hwu/Uni_sign_repo/out/signsmd_paired_parts_openasl_from0_h2s}
eval_dir=${EVAL_DIR:-/data/hwu/Uni_sign_repo/out/eval_signsmd_paired_parts_openasl_from0_h2s}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "deepspeed is not available; activate the training environment first." >&2
  exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter is not available: $python_bin" >&2
  exit 1
fi

check_parts_coverage() {
  local dataset=$1
  shift
  "$python_bin" - "$dataset" "$signsmd_contract" "$@" <<'PY'
import gzip
import json
import pickle
import sys
from pathlib import Path

dataset, contract, *splits = sys.argv[1:]
root = Path("data") / dataset / contract
text_root = root / "texts"
event_root = root / "events"
parts = ("body", "left", "right", "face_all")

missing_total = 0
invalid_total = 0
for split in splits:
    label_path = Path("data") / dataset / f"labels.{split}"
    with gzip.open(label_path, "rb") as f:
        labels = pickle.load(f)
    missing = []
    invalid = []
    for sample in labels.values():
        video_path = sample["video_path"]
        clip = video_path[:-4] if video_path.endswith(".mp4") else video_path
        required = [text_root / part / f"{clip}.txt" for part in parts]
        event_path = event_root / f"{clip}.json"
        required.append(event_path)
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing.append((video_path, absent))
            continue
        try:
            payload = json.loads(event_path.read_text())
            if payload.get("contract") != contract:
                raise ValueError(
                    f"contract={payload.get('contract')!r}, expected {contract!r}"
                )
            if payload.get("clip") != clip:
                raise ValueError(
                    f"clip={payload.get('clip')!r}, expected {clip!r}"
                )
            if not isinstance(payload.get("events"), list):
                raise ValueError("events is not a list")
        except Exception as exc:
            invalid.append((video_path, repr(exc)))
    complete = len(labels) - len(missing) - len(invalid)
    print(f"[{dataset}/{split}] parts coverage: {complete}/{len(labels)}")
    if missing:
        print(f"  first missing: {missing[:2]}", file=sys.stderr)
    if invalid:
        print(f"  first invalid: {invalid[:2]}", file=sys.stderr)
    missing_total += len(missing)
    invalid_total += len(invalid)

if missing_total or invalid_total:
    raise SystemExit(
        f"{dataset} parts corpus is incomplete: missing={missing_total}, "
        f"invalid={invalid_total}"
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
    echo "Choose a new directory or set ALLOW_EXISTING_OUTPUT=1 intentionally." >&2
    exit 1
  fi
}

paired_args=(
  --fusion paired
  --frame_sampling linear
  --max_length 256
  --signsmd_part_max_tokens "$part_max_tokens"
  --text_event_dropout "$text_event_dropout"
  --events_required
)

if [ "$stage" = pretrain ] || [ "$stage" = all ]; then
  check_parts_coverage OpenASL train dev test
  require_fresh_training_dir "$pretrain_dir"

  deepspeed --include "$gpus" --master_port "$port" fine_tuning_signsmd.py \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$grad_accum_steps" \
    --epochs 20 \
    --opt AdamW \
    --lr 2e-4 \
    --output_dir "$pretrain_dir" \
    --dataset OpenASL \
    --task SLT \
    --dtype fp32 \
    --save-last-only \
    --skip_nonfinite_loss \
    --signsmd_text_root "$openasl_text_root" \
    --signsmd_event_root "$openasl_event_root" \
    "${paired_args[@]}"
fi

if [ "$stage" = finetune ] || [ "$stage" = all ]; then
  check_parts_coverage How2Sign train test
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
    --output_dir "$finetune_dir" \
    --finetune "$pretrain_dir/best_checkpoint.pth" \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --save-last-only \
    --skip_nonfinite_loss \
    --signsmd_text_root "$h2s_text_root" \
    --signsmd_event_root "$h2s_event_root" \
    "${paired_args[@]}"
fi

if [ "$stage" = eval ] || [ "$stage" = all ]; then
  check_parts_coverage How2Sign train test
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
    --finetune "$finetune_dir/best_checkpoint.pth" \
    --output_dir "$eval_dir" \
    --eval \
    --signsmd_text_root "$h2s_text_root" \
    --signsmd_event_root "$h2s_event_root" \
    "${paired_args[@]}" \
    "${bleurt_args[@]}"

  deepspeed --include localhost:0 --master_port "$port" fine_tuning_signsmd.py \
    --batch-size "$batch_size" \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --finetune "$finetune_dir/best_checkpoint.pth" \
    --output_dir "${eval_dir}_mask_text" \
    --eval \
    --eval_mask_text \
    --signsmd_text_root "$h2s_text_root" \
    --signsmd_event_root "$h2s_event_root" \
    "${paired_args[@]}" \
    "${bleurt_args[@]}"
fi
