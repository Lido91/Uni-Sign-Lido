#!/usr/bin/env bash
# Stage 1 of the utility-aligned RAG plan on CSL-Daily: "translate, retrieve, refine"
# (doc/signrag_style_retriever_plan.md §11.2, doc/signrag_progress.md line G).
# Hints = top-3 distinct training sentences retrieved by TF-IDF with a draft translation
# (draft_retrieval.py). Dev/test drafts come from the released CSL-Daily model
# (csl_daily/baseline_eval: draft BLEU 25.14 / 25.61; test hint retrieval BLEU 10.66).
# Train drafts differ per run:
#   CSL_twopass_syn      references corrupted with noise 0.5 (train hint retrieval BLEU 10.82,
#                        matched to test); launched by hand on 10-25, not part of this queue
#   CSL_twopass_xfit     4 fold translators (folds grouped by sentence ID, each trained on 3/4
#                        of train with the CSL recipe) translate their held-out quarter
#   CSL_twopass_selffit  the released model, trained on all of train, translates train
#                        (C3 ablation: train hints better than test hints)
# RA-SFT and the fold translators use the CSL recipe (csl_stage1_weight, 20 epochs,
# 8 x accum 4, lr 3e-4, fp32, dev selection; compare with ra_sft/CSL_empty 25.18).
# A job starts, in list order, once its inputs exist, on a GPU without a queue job and
# with at least its memory need free; finished jobs are skipped, failed ones not retried.
#
#   ( setsid nohup bash script/twopass_csl.sh > $OUT/rag_inputs/twopass_csl/queue.out 2>&1 & )
#   bash script/twopass_csl.sh summary
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_HOME=${CUDA_HOME:-/home/student/hwu/miniconda3/envs/Uni-Sign}
export TOKENIZERS_PARALLELISM=false
PY=$CUDA_HOME/bin/python
DS=$CUDA_HOME/bin/deepspeed
OUT=${OUT:-/data/hwu/Uni_sign_repo/out/signrag_retrieval}
CK=/data/hwu/Uni_sign_repo/out
CSL_FT=$CK/csl_daily_pose_only/csl_daily_pose_only_slt.pth   # released Uni-Sign, all of train
CSL_PT=$CK/csl_pose_pretrain/csl_stage1_weight.pth           # released Uni-Sign, CSL-News stage 1
T=$OUT/rag_inputs/twopass_csl
FOLDS=4
EPOCHS=20
GPUS=${GPUS:-2 0 1 3}
TRAIN_MB=40000; EVAL_MB=16000                                 # free memory a job needs

# name | free MB needed | inputs that must exist
XFIT_DEPS=$(for k in $(seq 0 $((FOLDS - 1))); do printf "%s " "$T/fold${k}_heldout/test_tmp_pres.txt"; done)
JOBS=(
  "selffit_preds|$EVAL_MB|"
  "fold0|$TRAIN_MB|" "fold1|$TRAIN_MB|" "fold2|$TRAIN_MB|" "fold3|$TRAIN_MB|"
  "CSL_twopass_selffit|$TRAIN_MB|$T/selffit_preds/test_tmp_pres.txt"
  "CSL_twopass_xfit|$TRAIN_MB|$XFIT_DEPS"
)

trained() { [[ -f $1/log.txt ]] && (( $(wc -l < "$1/log.txt") >= EPOCHS )); }

done_job() {
    case $1 in
        selffit_preds) [[ -f $T/selffit_preds/test_tmp_pres.txt ]] ;;
        fold?)         [[ -f $T/${1}_heldout/test_tmp_pres.txt ]] ;;
        *)             trained "$OUT/ra_sft/$1" ;;
    esac
}

ready() { local f; for f in $1; do [[ -f $f ]] || return 1; done; }

prep_folds() {  # folds grouped by sentence ID (S######), as in word_retrieval_csl.py
    [[ -f $T/folds/labels.heldout.fold$((FOLDS - 1)) ]] && return
    mkdir -p "$T/folds"
    $PY - "$T/folds" $FOLDS <<'EOF'
import gzip, pickle, sys
import numpy as np
out, K = sys.argv[1], int(sys.argv[2])
with gzip.open("./data/CSL_Daily/labels.train", "rb") as f:
    data = pickle.load(f)
sents = sorted({v["name"][:7] for v in data.values()})
fold_of = {s: k % K for k, s in enumerate(np.random.RandomState(0).permutation(sents))}
for k in range(K):
    held = {key: v for key, v in data.items() if fold_of[v["name"][:7]] == k}
    rest = {key: v for key, v in data.items() if fold_of[v["name"][:7]] != k}
    for split, d in (("train", rest), ("heldout", held)):
        with gzip.open(f"{out}/labels.{split}.fold{k}", "wb") as f:
            pickle.dump(d, f)
    print(f"fold{k}: train {len(rest)} clips, heldout {len(held)} clips")
EOF
}

translate() {   # gpu port ckpt label out_dir: translate the clips of a label file (test phase)
    mkdir -p "$5"
    $DS --include localhost:$1 --master_port "$2" fine_tuning_rag.py \
        --finetune "$3" --dataset CSL_Daily --task SLT --dtype fp32 --seed 42 --batch-size 8 \
        --rag_mode empty --test_label "$4" --eval --output_dir "$5" > "$5/eval.out" 2>&1
}

train() {       # gpu port out_dir mode [args...]: CSL recipe
    local gpu=$1 port=$2 dir=$3 mode=$4
    shift 4
    mkdir -p "$dir"
    rm -f "$dir/log.txt"
    $DS --include localhost:$gpu --master_port "$port" fine_tuning_rag.py \
        --finetune "$CSL_PT" --dataset CSL_Daily --task SLT --dtype fp32 --seed 42 \
        --batch-size 8 --gradient-accumulation-steps 4 --epochs $EPOCHS --opt AdamW --lr 3e-4 \
        --rag_mode "$mode" "$@" --save-last-only --output_dir "$dir" > "$dir/train.out" 2>&1
}

rasft() {       # gpu port name tag pred_file: draft-retrieved train hints, then RA-SFT
    $PY draft_retrieval.py --dataset CSL_Daily --split train --source preds --pred_file "$5" \
        --tag "$4" --out_dir "$T" > "$T/draft_train_$4.out" 2>&1 || return 1
    grep "BLEU" "$T/draft_train_$4.out"
    train "$1" "$2" "$OUT/ra_sft/$3" real --rag_train_json "$T/retrieval_train_$4.json" \
        --rag_dev_json "$T/retrieval_dev_base.json" --rag_test_json "$T/retrieval_test_base.json"
}

launch() {      # gpu port name
    local g=$1 p=$2 name=$3
    echo "[run] $name on GPU $g $(date)"
    case $name in
        selffit_preds)
            translate "$g" "$p" "$CSL_FT" ./data/CSL_Daily/labels.train "$T/selffit_preds" ;;
        fold?)
            local k=${name#fold} dir=$T/$name
            { trained "$dir" || train "$g" "$p" "$dir" empty --train_label "$T/folds/labels.train.fold$k"; } \
                && translate "$g" "$p" "$dir/best_checkpoint.pth" "$T/folds/labels.heldout.fold$k" "$T/${name}_heldout" ;;
        CSL_twopass_selffit)
            rasft "$g" "$p" "$name" selffit "$T/selffit_preds/test_tmp_pres.txt" ;;
        CSL_twopass_xfit)
            cat "$T"/fold?_heldout/test_tmp_pres.txt > "$T/xfit_train_preds.txt" \
                && rasft "$g" "$p" "$name" xfit "$T/xfit_train_preds.txt" ;;
    esac || { echo "[FAIL] $name $(date)"; return 1; }
    echo "[done] $name $(date)"
}

gpu_ok() {      # gpu need_mb
    local pid=${RUNNING[$1]:-} free
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && return 1
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" \
           | awk -F', ' '{print $1 - $2}')
    (( free >= $2 ))
}

queue() {
    declare -gA RUNNING=()
    prep_folds
    local pending=("${JOBS[@]}") port=29621
    echo "queue start $(date): ${#pending[@]} jobs"
    while ((${#pending[@]})); do
        local rest=() job
        for job in "${pending[@]}"; do
            IFS='|' read -r name need deps <<< "$job"
            if done_job "$name"; then echo "[skip] $name"; continue; fi
            if ! ready "$deps"; then rest+=("$job"); continue; fi
            local g started=0
            for g in $GPUS; do
                if gpu_ok "$g" "$need"; then
                    launch "$g" $port "$name" &
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
        if ((${#pending[@]})); then
            # nothing running and no pending job ready: the missing inputs come from failed jobs
            local alive=0 runnable=0 pid
            for pid in "${RUNNING[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
            if ((!alive)); then
                for job in "${pending[@]}"; do IFS='|' read -r _ _ deps <<< "$job"; ready "$deps" && runnable=1; done
                ((runnable)) || { echo "[stop] inputs missing and nothing running: ${pending[*]%%|*} $(date)"; break; }
            fi
            sleep 60
        fi
    done
    wait
    echo "queue end $(date)"
}

summary() {
    echo
    echo "hints (top-3 distinct TF-IDF on the draft; retrieval BLEU = top-1 scored as a translation):"
    $PY - "$T" <<'EOF'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "metrics_*.json"))):
    m = json.load(open(f))
    tag = os.path.basename(f)[len("metrics_"):-len(".json")]
    print(f"  {tag:22s} draft B4 {m['draft_bleu4']:6.2f}   retrieval B4 {m['bleu4']:6.2f}   F1@3 {m['f1@3']:6.2f}")
EOF
    for k in $(seq 0 $((FOLDS - 1))); do
        local f=$T/fold${k}_heldout/eval.out
        [[ -f $f ]] && echo "  fold$k translator: dev / held-out B4 $(grep -o "'bleu4': [0-9.]*" "$f" | cut -d' ' -f2 | xargs)"
    done
    echo
    echo "RA-SFT (best dev epoch; test at that epoch):"
    printf "  %-20s %6s %5s %8s %28s %7s\n" run epochs best dev_B4 "test B1/B2/B3/B4" test_RL
    local run
    for run in CSL_empty CSL_real_d CSL_oracle_d CSL_twopass_syn CSL_twopass_xfit CSL_twopass_selffit; do
        local f=$OUT/ra_sft/$run/log.txt
        [[ -s $f ]] || { printf "  %-20s %6s\n" "$run" "-"; continue; }
        $PY - "$f" "$run" <<'EOF'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
b = max(L, key=lambda d: d["dev_bleu4"])
t = " / ".join(f"{b[f'test_bleu{i}']:.2f}" for i in range(1, 5))
print(f"  {sys.argv[2]:20s} {len(L):6d} {b['epoch']:5d} {b['dev_bleu4']:8.2f} {t:>28s} {b['test_rouge']:7.2f}")
EOF
    done
}

case ${1:-all} in
    all) queue; summary ;;
    summary) summary ;;
    *) echo "usage: $0 [all|summary]"; exit 1 ;;
esac
