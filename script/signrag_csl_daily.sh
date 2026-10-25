#!/usr/bin/env bash
# SignRAG reproduction on CSL-Daily (doc/signrag_progress.md, line D).
#   retriever : Step 1 features of the released CSL-Daily checkpoint (its encoder
#               was fine-tuned on the CSL-Daily train split, as SignRAG's is before
#               the gallery is built); S1 baseline retrieval; S2dn (fine objective,
#               exact positives, the best H2S retriever); S3dn retrieval; and the
#               RA-SFT inputs (train queries filtered: self / identical text).
#   rasft     : RA-SFT from the CSL-News stage-1 weight (never saw CSL-Daily), with
#               the official Uni-Sign CSL-Daily recipe (20 epochs, batch 32, lr 3e-4).
#               real = [top-3 retrieved train sentences][prompt][pose]; empty = no
#               retrieval (the SFT baseline). Selected on dev; test is reported at
#               the best dev epoch.
# Every stage is skipped when its output exists; a summary is printed at the end.
#
#   GPU=3 bash script/signrag_csl_daily.sh retriever
#   GPU_REAL=0,1 GPU_EMPTY=2,3 bash script/signrag_csl_daily.sh rasft
#   bash script/signrag_csl_daily.sh summary
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

stage=${1:-summary}
export CUDA_HOME=${CUDA_HOME:-/home/student/hwu/miniconda3/envs/Uni-Sign}
export TOKENIZERS_PARALLELISM=false
PY=$CUDA_HOME/bin/python
DS=$CUDA_HOME/bin/deepspeed
OUT=${OUT:-/data/hwu/Uni_sign_repo/out/signrag_retrieval}
CK=/data/hwu/Uni_sign_repo/out
CSL_FT=$CK/csl_daily_pose_only/csl_daily_pose_only_slt.pth   # released Uni-Sign, CSL-Daily SLT
CSL_PT=$CK/csl_pose_pretrain/csl_stage1_weight.pth           # released Uni-Sign, CSL-News stage 1
TAG=E_csl
RAG=$OUT/rag_inputs/S3dn_$TAG
EPOCHS=${EPOCHS:-20}; EFF=${EFF:-32}; LR=${LR:-3e-4}; DTYPE=${DTYPE:-fp32}
declare -A GPUS=([real]=${GPU_REAL:-0,1} [empty]=${GPU_EMPTY:-2,3})
declare -A PORT=([real]=29561 [empty]=29562)
MODES=${MODES:-real empty}
mkdir -p "$OUT"

# ---- retriever -------------------------------------------------------------
fail() { echo "[FAIL] $1 $(date)"; exit 1; }

extract() {   # split...
    for sp in "$@"; do
        if [[ -f $OUT/feats/$TAG/$sp/meta.json ]]; then echo "[skip] extract $sp"; continue; fi
        echo "[run] extract $sp $(date)"
        $PY signrag_retrieval.py extract --dataset CSL_Daily --finetune "$CSL_FT" \
            --splits "$sp" --feat_dir "$OUT/feats/$TAG" || fail "extract $sp"
    done
}

retrieve() {  # run_dir [retrieve args...]
    local dir=$1; shift
    if [[ -f $dir/metrics.json ]]; then echo "[skip] retrieve $(basename "$dir")"; return 0; fi
    echo "[run] retrieve $(basename "$dir") $(date)"
    $PY signrag_retrieval.py retrieve --feat_dir "$OUT/feats/$TAG" --run_dir "$dir" "$@" \
        || fail "retrieve $dir"
    $PY signrag_retrieval.py evaluate --run_dir "$dir" > /dev/null || fail "evaluate $dir"
}

retriever() {
    export CUDA_VISIBLE_DEVICES=${GPU:-3}
    extract train dev test
    retrieve "$OUT/runs/S1_$TAG" --score frame pool --queries test dev train:3000
    if [[ -f $OUT/step2/S2dn_$TAG/done.json ]]; then echo "[skip] train S2dn"
    else
        echo "[run] train S2dn $(date)"
        $PY signrag_contrastive.py --feat_dir "$OUT/feats/$TAG" --run_dir "$OUT/step2/S2dn_$TAG" \
            --dataset CSL_Daily --objective fine || fail "train S2dn"
    fi
    local phi=$OUT/step2/S2dn_$TAG/phi.pth
    retrieve "$OUT/runs/S3dn_$TAG" --phi "$phi" --score frame pool s2t t2s --pair_recall \
        --queries test dev train:3000
    retrieve "$RAG" --phi "$phi" --score s2t --queries train dev test --no_baselines
}

# ---- RA-SFT ----------------------------------------------------------------
run() {  # mode
    local mode=$1 dir=$OUT/ra_sft/CSL_$1
    if [[ -f $dir/log.txt ]] && [[ $(wc -l < "$dir/log.txt") -ge $EPOCHS ]]; then
        echo "[skip] $mode"; return 0
    fi
    local n_gpu accum
    n_gpu=$(tr ',' '\n' <<< "${GPUS[$mode]}" | wc -l)
    accum=$((EFF / (8 * n_gpu)))
    (( 8 * accum * n_gpu == EFF )) || { echo "[FAIL] $mode: $EFF not divisible by 8 x $n_gpu"; return 1; }
    echo "[run] $mode on GPU ${GPUS[$mode]} (accum $accum) $(date)"
    rm -f "$dir/log.txt"
    $DS --include localhost:${GPUS[$mode]} --master_port ${PORT[$mode]} fine_tuning_rag.py \
        --finetune "$CSL_PT" --dataset CSL_Daily --task SLT --dtype "$DTYPE" \
        --batch-size 8 --gradient-accumulation-steps "$accum" --epochs "$EPOCHS" --opt AdamW --lr "$LR" \
        --rag_mode "$mode" --rag_train_json "$RAG/retrieval_train_s2t.json" \
        --rag_dev_json "$RAG/retrieval_dev_s2t.json" --rag_test_json "$RAG/retrieval_test_s2t.json" \
        --save-last-only --output_dir "$dir" > /dev/null 2>&1 \
        || echo "[FAIL] $mode (see $dir)"
}

rasft() {
    [[ -f $RAG/retrieval_test_s2t.json ]] || fail "no RA-SFT inputs in $RAG (run the retriever stage)"
    mkdir -p "$OUT/ra_sft"
    echo "ckpt=$CSL_PT epochs=$EPOCHS effective_batch=$EFF lr=$LR dtype=$DTYPE modes=$MODES start $(date)"
    for mode in $MODES; do run "$mode" & done
    wait
}

# ---- summary ---------------------------------------------------------------
summary() {
    echo
    echo "retrieval (CSL-Daily):"
    for r in S1_$TAG S3dn_$TAG; do
        [[ -f $OUT/runs/$r/metrics.tsv ]] && { echo "  $r"; column -t -s $'\t' "$OUT/runs/$r/metrics.tsv" | sed 's/^/    /'; }
        [[ -f $OUT/runs/$r/pair_recall.tsv ]] && column -t -s $'\t' "$OUT/runs/$r/pair_recall.tsv" | sed 's/^/    /'
    done
    echo
    echo "RA-SFT (best dev epoch; test at that epoch):"
    printf "  %-6s %6s %5s %8s %28s %7s\n" run epochs best dev_B4 "test B1/B2/B3/B4" test_RL
    for mode in real empty; do
        f=$OUT/ra_sft/CSL_$mode/log.txt
        [[ -f $f ]] || { printf "  %-6s %6s\n" "$mode" "-"; continue; }
        $PY - "$f" "$mode" <<'EOF'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
b = max(L, key=lambda d: d["dev_bleu4"])
t = " / ".join(f"{b[f'test_bleu{i}']:.2f}" for i in range(1, 5))
print(f"  {sys.argv[2]:6s} {len(L):6d} {b['epoch']:5d} {b['dev_bleu4']:8.2f} {t:>28s} {b['test_rouge']:7.2f}")
EOF
    done
    echo "  reference: released CSL-Daily checkpoint dev 25.27 / test 25.70 B4;"
    echo "             SignRAG Table 2a: SFT 29.72 -> RA-SFT 31.22 B4"
}

case $stage in
    retriever) retriever; summary ;;
    rasft)     rasft; summary ;;
    summary)   summary ;;
    *)         echo "stage must be retriever, rasft or summary"; exit 1 ;;
esac
echo "end $(date)"
