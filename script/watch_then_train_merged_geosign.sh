#!/usr/bin/env bash
# Wait for the current How2Sign Stage 3 run to finish, then launch one
# OpenASL+How2Sign Geo-Sign joint run once physical GPUs 1,2,3 are idle.
set -uo pipefail

repo_root=/home/student/hwu/Workplace/Uni-Sign
baseline_out=/data/hwu/Uni_sign_repo/out/unified_stage3_h2s
baseline_log="$baseline_out/log.txt"
source_checkpoint=/data/hwu/Uni_sign_repo/out/unified_stage1_openasl/best_checkpoint.pth
merged_out=/data/hwu/Uni_sign_repo/out/unified_stage3_merged_geosign
watch_log=${WATCH_LOG:-/tmp/unified_stage3_merged_geosign_watch.log}
train_log=${TRAIN_LOG:-/tmp/unified_stage3_merged_geosign_train.log}
poll_seconds=${POLL_SECONDS:-1800}

log() {
    printf '[%s UTC] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" >> "$watch_log"
}

last_epoch() {
    [[ -s "$baseline_log" ]] || return 0
    tail -n 1 "$baseline_log" | grep -oE '"epoch"[[:space:]]*:[[:space:]]*[0-9]+' \
        | grep -oE '[0-9]+' | tail -n 1
}

baseline_active() {
    local file cmd
    for file in /proc/[0-9]*/cmdline; do
        [[ -r "$file" ]] || continue
        cmd=$(tr '\0' ' ' < "$file" 2>/dev/null) || continue
        if [[ "$cmd" == *train_unified.py* && "$cmd" == *"$baseline_out"* ]]; then
            return 0
        fi
    done
    return 1
}

baseline_finished() {
    [[ "$(last_epoch)" == 19 ]]
}

gpus_idle() {
    local gpu row memory utilization
    for gpu in 1 2 3; do
        row=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null) || return 1
        IFS=, read -r memory utilization <<< "$row"
        memory=${memory//[[:space:]]/}
        utilization=${utilization//[[:space:]]/}
        [[ "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
        if (( memory >= 2048 || utilization >= 10 )); then
            log "GPU $gpu still busy (${memory} MiB, ${utilization}%); waiting."
            return 1
        fi
    done
    return 0
}

log "Watcher active; checks every ${poll_seconds}s. Waiting for Stage 3 epoch 19 and idle GPUs 1,2,3."
log "Baseline log: $baseline_log; merged output: $merged_out"

while true; do
    completed_epoch=$(last_epoch || true)
    if baseline_active; then
        log "Baseline Stage 3 is still running; last completed epoch: ${completed_epoch:-unknown}."
    elif ! baseline_finished; then
        log "Baseline process is absent but epoch 19 is not recorded (last: ${completed_epoch:-unknown}); will not start yet."
    elif ! gpus_idle; then
        :
    else
        # Recheck the baseline immediately before starting to avoid racing its exit.
        if baseline_active; then
            log "Baseline process reappeared; waiting."
        elif [[ -e "$merged_out" ]]; then
            log "Refusing to overwrite existing output directory: $merged_out"
            exit 3
        elif [[ ! -s "$source_checkpoint" ]]; then
            log "Required Stage 1 checkpoint is missing: $source_checkpoint"
            exit 4
        else
            log "Stage 3 completed and GPUs 1,2,3 are idle. Starting merged Geo-Sign training."
            log "Config: Stage 3, OpenASL+How2Sign concatenated sampling, init=$source_checkpoint, batch=4, accumulation=16."
            if GEOSIGN_GPUS=1,2,3 BATCH_SIZE=4 ACCUM_STEPS=16 \
                INIT_CHECKPOINT="$source_checkpoint" OUTPUT_DIR="$merged_out" \
                bash "$repo_root/script/train_unified_merged_geosign.sh" \
                >> "$train_log" 2>&1; then
                log "Merged Geo-Sign training exited successfully. Training log: $train_log"
                exit 0
            else
                status=$?
                log "Merged Geo-Sign training exited with status $status. Training log: $train_log"
                exit "$status"
            fi
        fi
    fi
    sleep "$poll_seconds"
done
