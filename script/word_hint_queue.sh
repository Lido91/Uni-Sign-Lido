#!/usr/bin/env bash
# RA-SFT with word-level and TF-IDF hints on How2Sign and CSL-Daily
# (doc/signrag_progress.md, "Word-level retrieval"). A job starts on the first GPU
# that is free (no job of this queue on it and < FREE_MB used) once its hint files
# exist; jobs are taken in the order below (sentence-level TF-IDF oracle first,
# as requested on 2026-10-24). A run is skipped once its log.txt holds
# all epochs.
#
#   How2Sign  (E1 recipe of train_rag_h2s.sh: released H2S checkpoint, 10 epochs,
#             effective batch 64 = 8 x accum 8 on one GPU, lr 1e-4, fp32; compare
#             with ra_sft/H2S_empty 15.13)
#     H2S_oracle_d      ORACLE  top-3 distinct TF-IDF sentences of the GT translation
#     H2S_oracle_words  ORACLE  words of the same TF-IDF top-3 (tfidf_word_hints.py)
#     H2S_words         —       WLASL window spotting, no GT (word_retrieval_h2s.py)
#     H2S_gtwords       ORACLE  GT content words that have a WLASL gloss
#   CSL-Daily (csl_stage1_weight.pth, 20 epochs, effective batch 32 = 8 x accum 4,
#             lr 3e-4, fp32, dev selection; compare with ra_sft/CSL_empty 25.18)
#     CSL_oracle_words  ORACLE  neighbours' training glosses of the TF-IDF top-3
#     CSL_predgloss     —       CTC gloss spotter, train hints cross-fitted (word_retrieval_csl.py)
#   (CSL sentence-level TF-IDF = CSL_oracle_d, queued in csl_daily_overnight.sh;
#    CSL GT-gloss oracle = CSL_gloss, already running.)
#
#   bash script/word_hint_queue.sh            # detached by the caller
#   bash script/word_hint_queue.sh summary
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_HOME=${CUDA_HOME:-/home/student/hwu/miniconda3/envs/Uni-Sign}
export TOKENIZERS_PARALLELISM=false
PY=$CUDA_HOME/bin/python
DS=$CUDA_HOME/bin/deepspeed
OUT=${OUT:-/data/hwu/Uni_sign_repo/out/signrag_retrieval}
CK=/data/hwu/Uni_sign_repo/out
H2S_CKPT=$CK/how2sign_pose_only/how2sign_pose_only_slt.pth
CSL_CKPT=$CK/csl_pose_pretrain/csl_stage1_weight.pth
HT=$OUT/rag_inputs/tfidf_E1_h2s_distinct
HW=$OUT/word_retrieval/h2s
CW=$OUT/word_retrieval/csl
GPUS=${GPUS:-0 1 2 3}
FREE_MB=${FREE_MB:-12000}

# name | dataset | mode | train json | dev json | test json | readiness marker | extra args
# The remaining H2S runs skip isolated non-finite steps (H2S_oracle_words aborted on a
# NaN loss on 10-24; earlier runs never hit one, so they are unaffected by the flag).
NF=--skip_nonfinite_loss
JOBS=(
  "H2S_oracle_d|How2Sign|real|$HT/retrieval_train_tfidf_oracle.json||$HT/retrieval_test_tfidf_oracle.json||"
  "CSL_predgloss|CSL_Daily|hint|$CW/ctc_E_csl/hints_train.json|$CW/ctc_E_csl/hints_dev.json|$CW/ctc_E_csl/hints_test.json||"
  "CSL_oracle_words|CSL_Daily|hint|$CW/tfidf_oracle_words/hints_train.json|$CW/tfidf_oracle_words/hints_dev.json|$CW/tfidf_oracle_words/hints_test.json||"
  "H2S_oracle_words|How2Sign|hint|$HW/tfidf_oracle_words/hints_train.json||$HW/tfidf_oracle_words/hints_test.json||$NF"
  "H2S_words|How2Sign|hint|$HW/wlasl_hints/hints_train.json||$HW/wlasl_hints/hints_test.json|$HW/wlasl_hints/DONE|$NF"
  "H2S_gtwords|How2Sign|hint|$HW/wlasl_hints/oracle_gtwords_train.json||$HW/wlasl_hints/oracle_gtwords_test.json|$HW/wlasl_hints/DONE|$NF"
)

epochs_of() { [[ $1 == How2Sign ]] && echo 10 || echo 20; }

done_run() {  # name dataset
    local f=$OUT/ra_sft/$1/log.txt
    [[ -f $f ]] && [[ $(wc -l < "$f") -ge $(epochs_of "$2") ]]
}

ready() {     # files...
    local f
    for f in "$@"; do [[ -z $f || -f $f ]] || return 1; done
}

launch() {    # gpu port name dataset mode train dev test [extra args]
    local gpu=$1 port=$2 name=$3 ds=$4 mode=$5 dir=$OUT/ra_sft/$3 more=${9:-}
    local ckpt accum lr extra=()
    if [[ $ds == How2Sign ]]; then ckpt=$H2S_CKPT; accum=8; lr=1e-4; extra=(--skip_bleurt)
    else ckpt=$CSL_CKPT; accum=4; lr=3e-4; extra=(--rag_dev_json "$7"); fi
    mkdir -p "$dir"
    rm -f "$dir/log.txt"
    echo "[run] $name on GPU $gpu $(date)"
    $DS --include localhost:$gpu --master_port "$port" fine_tuning_rag.py \
        --finetune "$ckpt" --dataset "$ds" --task SLT --dtype fp32 --seed 42 \
        --batch-size 8 --gradient-accumulation-steps $accum --epochs "$(epochs_of "$ds")" --opt AdamW --lr $lr \
        --rag_mode "$mode" --rag_train_json "$6" --rag_test_json "$8" "${extra[@]}" $more \
        --save-last-only --output_dir "$dir" > /dev/null 2>&1 || echo "[FAIL] $name (see $dir) $(date)"
    echo "[done] $name $(date)"
}

gpu_free() {  # gpu -> 0 if no queue job runs there and memory use is low
    local pid=${RUNNING[$1]:-}
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && return 1
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    (( used < FREE_MB ))
}

queue() {
    declare -gA RUNNING=()
    local pending=("${JOBS[@]}") port=29601
    echo "queue start $(date): ${#pending[@]} jobs"
    while ((${#pending[@]})); do
        local rest=() job
        for job in "${pending[@]}"; do
            IFS='|' read -r name ds mode tr dv te dep more <<< "$job"
            if done_run "$name" "$ds"; then echo "[skip] $name"; continue; fi
            if ! ready "$tr" "$dv" "$te" "$dep"; then rest+=("$job"); continue; fi
            local g started=0
            for g in $GPUS; do
                if gpu_free "$g"; then
                    launch "$g" $port "$name" "$ds" "$mode" "$tr" "$dv" "$te" "$more" &
                    RUNNING[$g]=$!
                    port=$((port + 1))
                    started=1
                    sleep 120                     # let the job allocate before the next check
                    break
                fi
            done
            ((started)) || rest+=("$job")
        done
        pending=("${rest[@]}")
        ((${#pending[@]})) && sleep 60
    done
    wait
    echo "queue end $(date)"
}

summary() {
    echo
    printf "  %-18s %6s %5s %8s %28s %7s\n" run epochs best sel_B4 "test B1/B2/B3/B4" test_RL
    local job
    for job in "${JOBS[@]}"; do
        IFS='|' read -r name ds _ <<< "$job"
        local f=$OUT/ra_sft/$name/log.txt
        [[ -f $f ]] || { printf "  %-18s %6s\n" "$name" "-"; continue; }
        $PY - "$f" "$name" <<'EOF'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
key = "dev_bleu4" if "dev_bleu4" in L[0] else "test_bleu4"
b = max(L, key=lambda d: d[key])
t = " / ".join(f"{b[f'test_bleu{i}']:.2f}" for i in range(1, 5))
print(f"  {sys.argv[2]:18s} {len(L):6d} {b['epoch']:5d} {b[key]:8.2f} {t:>28s} {b['test_rouge']:7.2f}")
EOF
    done
}

case ${1:-all} in
    all) queue; summary ;;
    summary) summary ;;
    *) echo "usage: $0 [all|summary]"; exit 1 ;;
esac
