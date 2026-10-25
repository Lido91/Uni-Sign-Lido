#!/usr/bin/env bash
# RA-SFT on How2Sign (doc/signrag_style_retriever_plan.md §10c): the main run and
# its two controls, in parallel (one GPU each), with identical settings.
#   real   : [S3dn-retrieved top-3 training sentences][prompt][pose]
#   empty  : [ ][prompt][pose]          -> same fine-tuning, no retrieval
#   random : [3 random sentences][prompt][pose] -> more tokens, no relevance
# INIT picks the translator's starting point:
#   E1 (default): released Uni-Sign H2S checkpoint, i.e. RA-SFT as continued
#                 fine-tuning of a model already fine-tuned on H2S (ablation);
#                 10 epochs, effective batch 64.          -> ra_sft/H2S_<mode>
#   E2          : released Uni-Sign OpenASL checkpoint, never fine-tuned on H2S;
#                 RA-SFT is the first H2S SFT, as in SignRAG (Stage II
#                 pretraining -> target RA-SFT). 20 epochs, effective batch 192
#                 (= the 3-GPU x 8 x 8 OpenASL->H2S protocol of
#                 finetune_h2s_signsmd_from_openasl.sh). -> ra_sft/H2S_E2_<mode>
# Both use the S3dn retriever (E1 features), whose encoder was fine-tuned on
# the H2S train split as SignRAG's is before its gallery is built.
# A run is skipped once its log.txt holds all epochs; a BLEU summary is printed
# at the end. Model selection follows the existing H2S pipeline (test split).
#
#   bash script/train_rag_h2s.sh                # E1, GPUs: real 3, empty 2, random 1
#   INIT=E2 bash script/train_rag_h2s.sh        # E2
#   INIT=E2 MODES=real GPU_REAL=0,1,2 bash script/train_rag_h2s.sh   # one run on 3 GPUs
# MODES selects the runs (parallel); a GPU list makes a multi-GPU run, and
# grad-accum is set so that 8 x accum x n_gpus = the effective batch.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_HOME=${CUDA_HOME:-/home/student/hwu/miniconda3/envs/Uni-Sign}
export TOKENIZERS_PARALLELISM=false
DS=$CUDA_HOME/bin/deepspeed
OUT=${OUT:-/data/hwu/Uni_sign_repo/out/signrag_retrieval}
RAG=$OUT/rag_inputs/S3dn_E1_h2s
INIT=${INIT:-E1}
case $INIT in
    E1) CKPT=/data/hwu/Uni_sign_repo/out/how2sign_pose_only/how2sign_pose_only_slt.pth
        TAG=H2S; EPOCHS=${EPOCHS:-10}; EFF=${EFF:-64}; PORT0=29541 ;;
    E2) CKPT=/data/hwu/Uni_sign_repo/out/openasl_pose_pretrain/openasl_pose_only_slt.pth
        TAG=H2S_E2; EPOCHS=${EPOCHS:-20}; EFF=${EFF:-192}; PORT0=29544 ;;
    *)  echo "INIT must be E1 or E2"; exit 1 ;;
esac
declare -A GPU=([real]=${GPU_REAL:-3} [empty]=${GPU_EMPTY:-2} [random]=${GPU_RANDOM:-1})
declare -A PORT=([real]=$PORT0 [empty]=$((PORT0 + 1)) [random]=$((PORT0 + 2)))
MODES=${MODES:-real empty random}

run() {  # mode
    local mode=$1 dir=$OUT/ra_sft/${TAG}_$1
    if [[ -f $dir/log.txt ]] && [[ $(wc -l < "$dir/log.txt") -ge $EPOCHS ]]; then
        echo "[skip] $mode"; return 0
    fi
    local n_gpu accum
    n_gpu=$(tr ',' '\n' <<< "${GPU[$mode]}" | wc -l)
    accum=$((EFF / (8 * n_gpu)))
    (( 8 * accum * n_gpu == EFF )) || { echo "[FAIL] $mode: $EFF not divisible by 8 x $n_gpu"; return 1; }
    echo "[run] $mode on GPU ${GPU[$mode]} (accum $accum) $(date)"
    rm -f "$dir/log.txt"
    $DS --include localhost:${GPU[$mode]} --master_port ${PORT[$mode]} fine_tuning_rag.py \
        --finetune "$CKPT" --dataset How2Sign --task SLT --dtype fp32 \
        --batch-size 8 --gradient-accumulation-steps "$accum" --epochs "$EPOCHS" --opt AdamW --lr 1e-4 \
        --rag_mode "$mode" --rag_train_json "$RAG/retrieval_train_s2t.json" \
        --rag_test_json "$RAG/retrieval_test_s2t.json" \
        --save-last-only --skip_bleurt --output_dir "$dir" > /dev/null 2>&1 \
        || echo "[FAIL] $mode (see $dir)"
}

mkdir -p "$OUT/ra_sft"
echo "INIT=$INIT ckpt=$CKPT epochs=$EPOCHS effective_batch=$EFF modes=$MODES start $(date)"
for mode in $MODES; do run "$mode" & done
wait

echo
printf "%-8s %6s %8s %10s\n" run epochs best_bleu4 best_epoch
for mode in real empty random; do
    f=$OUT/ra_sft/${TAG}_$mode/log.txt
    [[ -f $f ]] || { printf "%-8s %6s\n" "$mode" "-"; continue; }
    python3 - "$f" "$mode" <<'EOF'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
b = max(L, key=lambda d: d["test_bleu4"])
print(f"{sys.argv[2]:8s} {len(L):6d} {b['test_bleu4']:8.2f} {b['epoch']:10d}")
EOF
done
if [[ $INIT == E1 ]]; then
    echo "baseline (released checkpoint, no extra fine-tuning): BLEU-4 14.50"
else
    echo "baseline = the empty run (E2 -> H2S SFT without retrieval)"
fi
echo "end $(date)"
