#!/bin/bash
# One-command SignSMD parts_v0_2 generation + paired Stage-1 pretraining.
#
# Default (`all`):
#   1. generate/resume OpenASL train/dev/test parts corpus;
#   2. generate/resume How2Sign train/test parts corpus;
#   3. validate four TXT files + one event JSON for every label row;
#   4. train OpenASL Stage-1 and select checkpoints on How2Sign test,
#      matching script/train_stage1.sh's training hyperparameters.
#
# Usage:
#   bash script/generate_parts_and_train_stage1.sh all
#   bash script/generate_parts_and_train_stage1.sh generate
#   bash script/generate_parts_and_train_stage1.sh stage1
#
# Optional overrides:
#   PYTHON_BIN=python GPUS=localhost:0,1,3 MASTER_PORT=29511 \
#   OUTPUT_DIR=out/my_stage1_parts ALLOW_EXISTING_OUTPUT=1 \
#     bash script/generate_parts_and_train_stage1.sh all

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

mode=${1:-all}
case "$mode" in
  generate|stage1|all) ;;
  *)
    echo "Usage: $0 [generate|stage1|all]" >&2
    exit 2
    ;;
esac

python_bin=${PYTHON_BIN:-python}
deepspeed_bin=${DEEPSPEED_BIN:-deepspeed}
gpus=${GPUS:-localhost:0,1,3}
port=${MASTER_PORT:-29511}
output_dir=${OUTPUT_DIR:-/data/hwu/Uni_sign_repo/out/stage1_pretraining_openasl_signsmd_paired_parts}
allow_existing_output=${ALLOW_EXISTING_OUTPUT:-0}
overwrite_generation=${OVERWRITE_GENERATION:-0}
coverage_checked=0

# These defaults intentionally mirror script/train_stage1.sh.
batch_size=${BATCH_SIZE:-16}
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}
epochs=${EPOCHS:-20}
learning_rate=${LEARNING_RATE:-3e-4}

# Paired-only controls. The mT5 encoder input length remains the baseline
# prompt+pose length; each short event phrase gets an independent token cap.
max_length=${MAX_LENGTH:-256}
part_max_tokens=${PART_MAX_TOKENS:-128}
text_event_dropout=${TEXT_EVENT_DROPOUT:-0.3}

contract=signsmd2d_parts_v0_2
openasl_root=data/OpenASL/$contract
h2s_root=data/How2Sign/$contract

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter is not available: $python_bin" >&2
  echo "Activate the Uni-Sign training environment or set PYTHON_BIN." >&2
  exit 1
fi

generate_args=(
  --variant parts
  --max_length "$max_length"
)
if [ "$overwrite_generation" = 1 ]; then
  generate_args+=(--overwrite)
fi

generate_corpora() {
  "$python_bin" generate_signsmd2d.py \
    --dataset OpenASL \
    --splits train dev test \
    "${generate_args[@]}"

  "$python_bin" generate_signsmd2d.py \
    --dataset How2Sign \
    --splits train test \
    "${generate_args[@]}"
}

check_parts_coverage() {
  local dataset=$1
  shift
  "$python_bin" - "$dataset" "$contract" "$max_length" "$@" <<'PY'
import gzip
import json
import pickle
import sys
from pathlib import Path

dataset, contract, max_length, *splits = sys.argv[1:]
max_length = int(max_length)
root = Path("data") / dataset / contract
parts = ("body", "left", "right", "face_all")
missing_total = 0
invalid_total = 0

for split in splits:
    with gzip.open(Path("data") / dataset / f"labels.{split}", "rb") as f:
        labels = pickle.load(f)
    missing = []
    invalid = []
    for sample in labels.values():
        video_path = sample["video_path"]
        clip = video_path[:-4] if video_path.endswith(".mp4") else video_path
        text_paths = [root / "texts" / part / f"{clip}.txt" for part in parts]
        event_path = root / "events" / f"{clip}.json"
        absent = [str(path) for path in (*text_paths, event_path)
                  if not path.is_file()]
        if absent:
            missing.append((video_path, absent))
            continue
        try:
            payload = json.loads(event_path.read_text(encoding="utf-8"))
            if payload.get("contract") != contract:
                raise ValueError(f"wrong contract {payload.get('contract')!r}")
            if payload.get("clip") != clip:
                raise ValueError(f"wrong clip {payload.get('clip')!r}")
            num_steps = payload.get("num_steps")
            if not isinstance(num_steps, int) or not 1 <= num_steps <= max_length:
                raise ValueError(f"invalid num_steps {num_steps!r}")
            events = payload.get("events")
            if not isinstance(events, list):
                raise ValueError("events is not a list")
            for event in events:
                if event.get("part") not in parts:
                    raise ValueError(f"invalid event part {event.get('part')!r}")
                start, end = event.get("start"), event.get("end")
                if (not isinstance(start, int) or not isinstance(end, int)
                        or not 0 <= start <= end < num_steps):
                    raise ValueError(f"invalid event span [{start}, {end}]")
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
        f"{dataset} parts corpus failed preflight: "
        f"missing={missing_total}, invalid={invalid_total}"
    )
PY
}

check_all_corpora() {
  check_parts_coverage OpenASL train dev test
  check_parts_coverage How2Sign train test
}

require_fresh_output() {
  if [ "$allow_existing_output" = 1 ]; then
    return
  fi
  if [ -e "$output_dir/best_checkpoint.pth" ] || \
     [ -e "$output_dir/last_checkpoint.pth" ] || \
     [ -e "$output_dir/log.txt" ]; then
    echo "Stage-1 output already exists: $output_dir" >&2
    echo "Choose OUTPUT_DIR, or set ALLOW_EXISTING_OUTPUT=1 intentionally." >&2
    exit 1
  fi
}

train_stage1() {
  if ! command -v "$deepspeed_bin" >/dev/null 2>&1; then
    echo "DeepSpeed is not available: $deepspeed_bin" >&2
    echo "Activate the Uni-Sign training environment or set DEEPSPEED_BIN." >&2
    exit 1
  fi
  if [ "$coverage_checked" != 1 ]; then
    check_all_corpora
    coverage_checked=1
  fi
  require_fresh_output

  "$deepspeed_bin" --include "$gpus" --master_port "$port" \
    fine_tuning_signsmd.py \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$grad_accum_steps" \
    --epochs "$epochs" \
    --opt AdamW \
    --lr "$learning_rate" \
    --output_dir "$output_dir" \
    --dataset OpenASL \
    --task SLT \
    --dtype fp32 \
    --eval-on-how2sign-test \
    --fusion paired \
    --frame_sampling linear \
    --max_length "$max_length" \
    --signsmd_part_max_tokens "$part_max_tokens" \
    --text_event_dropout "$text_event_dropout" \
    --events_required \
    --signsmd_text_root "$openasl_root/texts" \
    --signsmd_event_root "$openasl_root/events" \
    --eval_signsmd_text_root "$h2s_root/texts" \
    --eval_signsmd_event_root "$h2s_root/events"
}

if [ "$mode" = generate ] || [ "$mode" = all ]; then
  generate_corpora
  check_all_corpora
  coverage_checked=1
fi

if [ "$mode" = stage1 ] || [ "$mode" = all ]; then
  train_stage1
fi
