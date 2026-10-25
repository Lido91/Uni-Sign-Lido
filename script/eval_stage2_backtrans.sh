#!/usr/bin/env bash
# Foreground Stage-2 OpenASL generation and frozen-SLT back-translation.
# GPU=1 bash script/eval_stage2_backtrans.sh [gen|ceilings|all]
# GEN_CHECKPOINT=/path/checkpoint.pth GPU=2 bash script/eval_stage2_backtrans.sh gen
# DRY_RUN=1 bash script/eval_stage2_backtrans.sh gen
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode=${1:-gen}
case "$mode" in
    gen) modes=(gen) ;;
    ceilings) modes=(gt gt_conf1 vae_recon) ;;
    all) modes=(gt gt_conf1 vae_recon gen) ;;
    -h|--help)
        echo 'Usage: GPU=1 bash script/eval_stage2_backtrans.sh [gen|ceilings|all]'
        echo 'Overrides: GEN_CHECKPOINT, SLT_CHECKPOINT, OUTPUT_DIR, SPLIT, BATCH_SIZE, DDIM_STEPS, SEED'
        echo 'DRY_RUN=1 prints the commands after checking the required files and dataset.'
        exit 0 ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

python=${UNISIGN_PYTHON:-/home/student/hwu/miniconda3/envs/Uni-Sign/bin/python}
out_root=${OUTPUT_ROOT:-/data/hwu/Uni_sign_repo/out}
stage2_dir=${STAGE2_DIR:-$out_root/unified_stage2_openasl}
checkpoint=${GEN_CHECKPOINT:-$stage2_dir/best_slg_checkpoint.pth}
judge=${SLT_CHECKPOINT:-$out_root/openasl_pose_pretrain/openasl_pose_only_slt.pth}
vae_root=${VAE_ROOT:-/data/hwu/momadiff_repo/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d}
latent_stats=${LATENT_STATS:-$out_root/latent_stats}
out_dir=${OUTPUT_DIR:-$stage2_dir/back_translation}
gpu=${GPU:-${EVAL_GPU:-1}}
split=${SPLIT:-test}
case "$split" in dev|test) ;; *) echo 'SPLIT must be dev or test' >&2; exit 2 ;; esac

required=("$python" "$judge" "$vae_root/net_best_hand.pth"
    "$vae_root/meta/mean.npy" "$vae_root/meta/std.npy"
    "$latent_stats/latent_mean.npy" "$latent_stats/latent_std.npy")
if [[ "$mode" != ceilings ]]; then required+=("$checkpoint"); fi
for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then echo "Missing required file: $path" >&2; exit 1; fi
done

# Read-only inventory; fail before loading models if coverage is incomplete.
"$python" - "$split" <<'PY'
import gzip
import pickle
import sys
from pathlib import Path
from config import dev_label_paths, test_label_paths, pose_dirs

split = sys.argv[1]
label = (test_label_paths if split == 'test' else dev_label_paths)['OpenASL']
with gzip.open(label, 'rb') as handle:
    samples = pickle.load(handle)
pose_root = Path(pose_dirs['OpenASL'].format(split))
missing = [s['video_path'] for s in samples.values()
           if not (pose_root / s['video_path'].replace('.mp4', '.pkl')).is_file()]
names = [str(s['name']) for s in samples.values()]
if missing:
    raise SystemExit(f'Missing {len(missing)} poses; examples: {missing[:3]}')
if not names or len(set(names)) != len(names):
    raise SystemExit('Evaluation sample IDs must be nonempty and unique')
print(f'OpenASL {split}: {len(samples)} samples; all pose files exist.')
PY

export CUDA_VISIBLE_DEVICES="$gpu"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/unisign-stage2-backtrans-matplotlib}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/unisign-stage2-backtrans-triton}
export TOKENIZERS_PARALLELISM=false

common=(--dataset OpenASL --split "$split" --task SLT --dtype fp32
    --batch-size "${BATCH_SIZE:-8}" --num_workers "${NUM_WORKERS:-4}"
    --max_length "${MAX_FRAMES:-400}" --frame_sampling vae_linear
    --backtrans_max_frames "${JUDGE_MAX_FRAMES:-256}"
    --num_beams "${NUM_BEAMS:-4}" --max_new_tokens "${MAX_NEW_TOKENS:-100}"
    --seed "${SEED:-42}" --slt_checkpoint "$judge" --output_dir "$out_dir")
vae_args=(--vae_checkpoint "$vae_root/net_best_hand.pth"
    --vae_mean_path "$vae_root/meta/mean.npy" --vae_std_path "$vae_root/meta/std.npy"
    --latent_mean_path "$latent_stats/latent_mean.npy"
    --latent_std_path "$latent_stats/latent_std.npy")

echo "GPU: $gpu; frozen OpenASL judge: $judge"
echo "Stage-2 generation checkpoint: $checkpoint"
echo "Output: $out_dir"
if [[ ${DRY_RUN:-0} != 1 ]]; then
    mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$out_dir"
    "$python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable for the selected GPU; evaluation was not started.')
print(f'CUDA device: {torch.cuda.get_device_name(0)} (logical cuda:0)')
PY
fi
for eval_mode in "${modes[@]}"; do
    args=("${common[@]}" --mode "$eval_mode")
    case "$eval_mode" in
        vae_recon) args+=("${vae_args[@]}" --save_predictions) ;;
        gen) args+=("${vae_args[@]}" --save_predictions
            --unified_checkpoint "$checkpoint" --trans_infer_timesteps "${MAR_STATES:-9}"
            --ddim_steps "${DDIM_STEPS:-100}" --guidance_param "${GUIDANCE:-3.0}") ;;
    esac
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        printf 'Command:'
        printf ' %q' "$python" -u back_translation_eval.py "${args[@]}"
        printf '\n'
        continue
    fi
    echo "Evaluating $eval_mode (foreground; generation uses supplied GT length)."
    "$python" -u back_translation_eval.py "${args[@]}" \
        | tee "$out_dir/${eval_mode}_OpenASL_${split}.log"
done

if [[ ${DRY_RUN:-0} == 1 ]]; then exit 0; fi

# Audit real outputs, including the saved ID set, after successful evaluation.
"$python" - "$out_dir" "$split" "${modes[@]}" <<'PY'
import gzip
import json
import math
import pickle
import sys
from pathlib import Path
from config import dev_label_paths, test_label_paths

root, split, *modes = sys.argv[1:]
root = Path(root)
label = (test_label_paths if split == 'test' else dev_label_paths)['OpenASL']
with gzip.open(label, 'rb') as handle:
    samples = pickle.load(handle)
expected_names = {str(sample['name']) for sample in samples.values()}
summary = {}
for mode in modes:
    result = json.loads((root / f'{mode}_OpenASL_{split}_results.json').read_text())
    if result['num_samples'] != len(samples):
        raise SystemExit(f'{mode}: incomplete evaluated coverage')
    for key, value in result.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise SystemExit(f'{mode}: nonfinite metric {key}')
    if mode in ('gen', 'vae_recon'):
        files = list(Path(result['prediction_dir']).glob('*.pkl'))
        actual_names = set()
        for path in files:
            with path.open('rb') as handle:
                actual_names.add(str(pickle.load(handle)['name']))
        if (len(files) != len(samples) or actual_names != expected_names
                or result['saved_prediction_count'] != len(samples)):
            raise SystemExit(f'{mode}: saved prediction coverage does not match labels')
    summary[mode] = result
    print(f"{mode}: n={result['num_samples']}, BLEU4={result['bleu4']:.4f}, "
          f"ROUGE={result['rouge']:.4f}, hand DTW={result.get('openasl_DTW_MPJPE2D_hand', 'n/a')}")
(root / 'backtrans_summary.json').write_text(json.dumps(summary, indent=2))
print(f'Coverage audit passed. Results and poses: {root}')
PY
