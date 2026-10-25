#!/usr/bin/env bash
# SignRAG-style retriever experiment matrix (doc/signrag_style_retriever_plan.md §6).
# Every stage is skipped when its output already exists. A failing stage is
# recorded in $OUT/failed.txt and the remaining stages still run (stages that
# depend on it fail fast and are recorded too). A summary table is printed at
# the end (and written to $OUT/summary.tsv / summary_pair_recall.tsv).
#
#   bash script/signrag_retrieval.sh           # P0: E1-H2S Step 1, S2-a/S2-s/S2-b, Step 3
#   bash script/signrag_retrieval.sh p1        # P0 + E2-H2S/E3-OpenASL Step 1, tau/theta ablations, S2-c/d/dn/e/ds
#   bash script/signrag_retrieval.sh encoder   # everything: P1 + S2-fd (fine + --train_encoder) with re-extraction
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

level=${1:-p0}
PY=${PY:-/home/student/hwu/miniconda3/envs/Uni-Sign/bin/python}
OUT=${OUT:-/data/hwu/Uni_sign_repo/out/signrag_retrieval}
CK=/data/hwu/Uni_sign_repo/out
E1=$CK/how2sign_pose_only/how2sign_pose_only_slt.pth      # released Uni-Sign, How2Sign
E2=$CK/openasl_pose_pretrain/openasl_pose_only_slt.pth    # released Uni-Sign, OpenASL
POS_TH=${POS_TH:-0.85}                                     # semantic-positive threshold
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
rm -f "$OUT/failed.txt"

run_stage() { # name cmd...  -> runs cmd; on failure records it and returns 1
    local name=$1 rc; shift
    "$@"
    rc=$?
    [[ $rc == 0 ]] && return 0
    echo "$(date '+%F %T')  $name  (exit $rc)" >> "$OUT/failed.txt"
    echo "[FAIL] $name (exit $rc)"
    return 1
}

extract() {   # feat_tag dataset checkpoint split...
    local tag=$1 ds=$2 ckpt=$3; shift 3
    for sp in "$@"; do
        if [[ -f $OUT/feats/$tag/$sp/meta.json ]]; then echo "[skip] extract $tag/$sp"; continue; fi
        run_stage "extract $tag/$sp" $PY signrag_retrieval.py extract --dataset "$ds" --finetune "$ckpt" \
            --splits "$sp" --feat_dir "$OUT/feats/$tag" || return 1
    done
}

retrieve() {  # run feat_tag [retrieve args...]   (Step 1 and Step 3)
    local run=$1 tag=$2; shift 2
    if [[ -f $OUT/runs/$run/metrics.json ]]; then echo "[skip] retrieve $run"; return 0; fi
    run_stage "retrieve $run" $PY signrag_retrieval.py retrieve --feat_dir "$OUT/feats/$tag" \
        --run_dir "$OUT/runs/$run" "$@" || return 1
    run_stage "evaluate $run" $PY signrag_retrieval.py evaluate --run_dir "$OUT/runs/$run" > /dev/null
}

train() {     # run feat_tag [contrastive args...]  (Step 2)
    local run=$1 tag=$2; shift 2
    if [[ -f $OUT/step2/$run/done.json ]]; then echo "[skip] train $run"; return 0; fi
    run_stage "train $run" $PY signrag_contrastive.py --feat_dir "$OUT/feats/$tag" \
        --run_dir "$OUT/step2/$run" "$@"
}

step23() {    # run feat_tag [contrastive args...] -> Step 2 then its Step 3 retrieval
    local run=$1 tag=$2; shift 2
    train "$run" "$tag" "$@" || return 1
    retrieve "${run/#S2/S3}" "$tag" --phi "$OUT/step2/$run/phi.pth" --score frame pool s2t t2s --pair_recall
}

# ---- P0 ------------------------------------------------------------------
extract E1_h2s How2Sign "$E1" train test
retrieve S1_E1_h2s E1_h2s --score frame pool
step23 S2a_E1_h2s E1_h2s                                                 # ours, exact positives
step23 S2s_E1_h2s E1_h2s --positives semantic --pos_threshold "$POS_TH"  # ours, semantic positives
step23 S2b_E1_h2s E1_h2s --positives semantic --pos_threshold "$POS_TH" --init random

# ---- P1 ------------------------------------------------------------------
if [[ $level == p1 || $level == encoder ]]; then
    retrieve S1_E1_h2s_tau005 E1_h2s --score frame --frame_tau 0.05 --no_baselines
    extract E2_h2s How2Sign "$E2" train test && retrieve S1_E2_h2s E2_h2s --score frame pool
    if extract E3_openasl OpenASL "$E2" train dev test; then
        retrieve S1_E3_openasl E3_openasl --score frame pool
        retrieve S1_E3_openasl_samevideo E3_openasl --score frame pool --same_video on
    fi
    step23 S2s09_E1_h2s E1_h2s --positives semantic --pos_threshold 0.9      # threshold ablation
    step23 S2c_E1_h2s E1_h2s --positives semantic --pos_threshold "$POS_TH" --phi_mlp_hidden 1024
    step23 S2d_E1_h2s E1_h2s --objective fine --slt_weight 1.0                # SignRAG Stage I
    step23 S2dn_E1_h2s E1_h2s --objective fine                                # ... without L_slt
    step23 S2e_E1_h2s E1_h2s --objective fine --slt_weight 1.0 --batch_size 8 # ... at batch 8
    step23 S2ds_E1_h2s E1_h2s --objective fine --positives semantic --pos_threshold "$POS_TH"
fi

# ---- encoder -------------------------------------------------------------
if [[ $level == encoder ]]; then
    # S2-dn (fine objective) with the ST-GCN trained too; replaced the pooled S2f on 2026-10-22
    if train S2fd_E1_h2s E1_h2s --objective fine --train_encoder --epochs 15 --patience 4 \
        && extract S2fd_E1_h2s How2Sign "$OUT/step2/S2fd_E1_h2s/encoder.pth" train test; then
        retrieve S3fd_E1_h2s S2fd_E1_h2s --phi "$OUT/step2/S2fd_E1_h2s/phi.pth" \
            --score s2t t2s --pair_recall
    fi
fi

# ---- summary -------------------------------------------------------------
summarize() { # file_in_run_dir summary_path
    local name=$1 out=$2 first=1
    rm -f "$out"
    for f in "$OUT"/runs/*/"$name"; do
        [[ -f $f ]] || continue
        local run; run=$(basename "$(dirname "$f")")
        if [[ $first == 1 ]]; then sed -n '1s/^/run\t/p' "$f" > "$out"; first=0; fi
        tail -n +2 "$f" | sed "s/^/$run\t/" >> "$out"
    done
    if [[ -f $out ]]; then echo; column -t -s $'\t' "$out"; fi
}
summarize metrics.tsv "$OUT/summary.tsv"
summarize pair_recall.tsv "$OUT/summary_pair_recall.tsv"   # V2T / T2V recall of trained retrievers
if [[ -f $OUT/failed.txt ]]; then echo; echo "FAILED stages:"; cat "$OUT/failed.txt"; else echo; echo "all stages done"; fi
