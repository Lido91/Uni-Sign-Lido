#!/usr/bin/env bash
# CSL-Daily follow-up runs after the first RA-SFT pair (doc/signrag_progress.md, line D),
# queued on four single-GPU lanes so no GPU idles. Every RA-SFT run uses the
# script/signrag_csl_daily.sh recipe (csl_stage1_weight.pth, 20 epochs, effective
# batch 32 = 8 x accum 4 on one GPU, lr 3e-4, fp32, dev selection).
#
#   A (GPU 1): distinct-text RA-SFT inputs -> real_d
#   B (GPU 2): gloss
#   C (GPU 3): random_d
#   D (GPU 0): wait for the oracle eval -> diag evals of CSL_real -> oracle_d
#              -> diag evals of CSL_real_d
#
#   real_d    : top-3 *distinct* S3dn sentences (the first pair repeated one
#               sentence 3x because CSL-Daily repeats texts over signers)
#   oracle_d  : top-3 distinct TF-IDF neighbours of the GT text (train: self and
#               identical text filtered), a ceiling for sentence retrieval
#   gloss     : GT gloss sequence as the hint, a ceiling for word-level hints
#   random_d  : 3 random training sentences (control)
# The seed-43 repeats (real_d_s43, empty_s43) were cancelled on 2026-10-24 at the
# user's request; no seed-repeat runs are queued.
#
#   bash script/csl_daily_overnight.sh            # all lanes, detached by the caller
#   bash script/csl_daily_overnight.sh summary
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_HOME=${CUDA_HOME:-/home/student/hwu/miniconda3/envs/Uni-Sign}
export TOKENIZERS_PARALLELISM=false
PY=$CUDA_HOME/bin/python
DS=$CUDA_HOME/bin/deepspeed
OUT=${OUT:-/data/hwu/Uni_sign_repo/out/signrag_retrieval}
CK=/data/hwu/Uni_sign_repo/out
CSL_PT=$CK/csl_pose_pretrain/csl_stage1_weight.pth
PHI=$OUT/step2/S2dn_E_csl/phi.pth
RAG=$OUT/rag_inputs/S3dn_E_csl                   # first pair (duplicate top-3)
RAGD=$OUT/rag_inputs/S3dn_E_csl_distinct         # distinct top-3 + TF-IDF oracle + random
EPOCHS=20
LOGD=$OUT/csl_daily
mkdir -p "$OUT/ra_sft" "$LOGD"

wait_for() {  # file
    until [[ -f $1 ]]; do sleep 60; done
}

sft() {  # name mode gpu port seed train_json dev_json test_json
    local name=$1 mode=$2 gpu=$3 port=$4 seed=$5 dir=$OUT/ra_sft/CSL_$1
    if [[ -f $dir/log.txt ]] && [[ $(wc -l < "$dir/log.txt") -ge $EPOCHS ]]; then
        echo "[skip] $name"; return 0
    fi
    echo "[run] $name ($mode, seed $seed) on GPU $gpu $(date)"
    rm -f "$dir/log.txt"
    $DS --include localhost:$gpu --master_port "$port" fine_tuning_rag.py \
        --finetune "$CSL_PT" --dataset CSL_Daily --task SLT --dtype fp32 --seed "$seed" \
        --batch-size 8 --gradient-accumulation-steps 4 --epochs $EPOCHS --opt AdamW --lr 3e-4 \
        --rag_mode "$mode" --rag_train_json "$6" --rag_dev_json "$7" --rag_test_json "$8" \
        --save-last-only --output_dir "$dir" > /dev/null 2>&1 || echo "[FAIL] $name (see $dir) $(date)"
    echo "[done] $name $(date)"
}

evalrun() {  # name ckpt mode gpu port dev_json test_json
    local name=$1 dir=$LOGD/$1
    if [[ -f $dir/test_tmp_pres.txt ]]; then echo "[skip] eval $name"; return 0; fi
    [[ -f $2 ]] || { echo "[FAIL] eval $name: no checkpoint $2"; return 1; }
    echo "[run] eval $name on GPU $4 $(date)"
    mkdir -p "$dir"
    $DS --include localhost:$4 --master_port "$5" fine_tuning_rag.py \
        --finetune "$2" --dataset CSL_Daily --task SLT --dtype fp32 --batch-size 8 \
        --rag_mode "$3" --rag_train_json "$RAG/retrieval_train_s2t.json" \
        --rag_dev_json "$6" --rag_test_json "$7" --eval --output_dir "$dir" > "$dir/eval.out" 2>&1 \
        || echo "[FAIL] eval $name (see $dir)"
}

lane_A() {
    if [[ -f $RAGD/metrics.json ]]; then echo "[skip] distinct inputs"
    else
        echo "[run] distinct inputs on GPU 1 $(date)"
        CUDA_VISIBLE_DEVICES=1 $PY signrag_retrieval.py retrieve --feat_dir "$OUT/feats/E_csl" \
            --run_dir "$RAGD" --phi "$PHI" --score s2t --queries train dev test \
            --distinct_text --random_seeds 1 \
            && CUDA_VISIBLE_DEVICES=1 $PY signrag_retrieval.py evaluate --run_dir "$RAGD" > /dev/null \
            || { echo "[FAIL] distinct inputs $(date)"; return 1; }
    fi
    sft real_d real 1 29581 42 "$RAGD/retrieval_train_s2t.json" "$RAGD/retrieval_dev_s2t.json" "$RAGD/retrieval_test_s2t.json"
}

lane_B() {
    # the old subtitle-constrained H2S spotting step was removed on 2026-10-24: the
    # window-level dump of word_retrieval_h2s.py replaces it (doc/signrag_progress.md)
    sft gloss gloss 2 29583 42 "" "" ""
}

lane_C() {
    sft random_d random 3 29585 42 "" "" ""
}

lane_D() {
    while pgrep -f "[o]racle_eval_real" > /dev/null; do sleep 60; done
    local real=$OUT/ra_sft/CSL_real/best_checkpoint.pth
    evalrun diag_real_emptyhint "$real" empty 0 29587 "" ""
    wait_for "$RAGD/metrics.json"
    evalrun diag_real_oracle_d "$real" real 0 29587 \
        "$RAGD/retrieval_dev_tfidf_oracle.json" "$RAGD/retrieval_test_tfidf_oracle.json"
    sft oracle_d real 0 29586 42 "$RAGD/retrieval_train_tfidf_oracle.json" \
        "$RAGD/retrieval_dev_tfidf_oracle.json" "$RAGD/retrieval_test_tfidf_oracle.json"
    local od=$OUT/ra_sft/CSL_oracle_d/best_checkpoint.pth
    evalrun diag_oracle_d_s3dn "$od" real 0 29588 "$RAGD/retrieval_dev_s2t.json" "$RAGD/retrieval_test_s2t.json"
    # lane A prints "[done] real_d <date>" when real_d ends (finished or failed)
    until grep -q "^\[done\] real_d " "$LOGD/overnight_A.out" 2>/dev/null \
        || grep -q "FAIL\] distinct" "$LOGD/overnight_A.out" 2>/dev/null; do
        sleep 60
    done
    local rd=$OUT/ra_sft/CSL_real_d/best_checkpoint.pth
    evalrun diag_real_d_oracle_d "$rd" real 0 29589 "$RAGD/retrieval_dev_tfidf_oracle.json" "$RAGD/retrieval_test_tfidf_oracle.json"
    evalrun diag_real_d_emptyhint "$rd" empty 0 29589 "" ""
}

summary() {
    echo
    echo "CSL-Daily RA-SFT (best dev epoch; test at that epoch):"
    printf "  %-12s %6s %5s %8s %28s %7s\n" run epochs best dev_B4 "test B1/B2/B3/B4" test_RL
    for d in "$OUT"/ra_sft/CSL_*/; do
        local f=$d/log.txt name; name=$(basename "$d"); name=${name#CSL_}
        [[ -f $f ]] || { printf "  %-12s %6s\n" "$name" "-"; continue; }
        $PY - "$f" "$name" <<'EOF'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
b = max(L, key=lambda d: d["dev_bleu4"])
t = " / ".join(f"{b[f'test_bleu{i}']:.2f}" for i in range(1, 5))
print(f"  {sys.argv[2]:12s} {len(L):6d} {b['epoch']:5d} {b['dev_bleu4']:8.2f} {t:>28s} {b['test_rouge']:7.2f}")
EOF
    done
    echo
    echo "eval-only diagnostics (dev / test B4):"
    for d in "$LOGD"/diag_* "$LOGD"/oracle_eval_real; do
        [[ -f $d/eval.out ]] || continue
        printf "  %-26s %s\n" "$(basename "$d")" \
            "$(grep -a "'bleu4'" "$d/eval.out" | sed -E "s/.*'bleu4': ([0-9.]+).*/\1/" | xargs printf '%.2f ')"
    done
}

case ${1:-all} in
    all)
        echo "start $(date)"
        lane_A > "$LOGD/overnight_A.out" 2>&1 &
        lane_B > "$LOGD/overnight_B.out" 2>&1 &
        lane_C > "$LOGD/overnight_C.out" 2>&1 &
        lane_D > "$LOGD/overnight_D.out" 2>&1 &
        wait
        summary
        echo "end $(date)" ;;
    summary) summary ;;
    *) echo "usage: $0 [all|summary]"; exit 1 ;;
esac
