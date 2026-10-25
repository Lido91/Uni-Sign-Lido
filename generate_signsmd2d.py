"""Bulk SignSMD-2D generation over the Uni-Sign RTMPose corpora.

Compiles every clip of a dataset split into structured description text.
The parts variant writes one standalone text per pose part; all other variants
write one text per clip.  All variants share one step-aligned timeline (the
compiler resamples with the trainer's linear rule first, so
[step a-b] spans index the model's pose sequence directly):

    --variant coarse   interval prose        (contract signsmd2d_v0_2)
    --variant stats    per-channel numeric   (contract signsmd2d_stats_v0_1)
                       timelines, motion-smd angle_text_v5 style
    --variant fingers  compact left/right per-finger curl events only
                       (contract signsmd2d_fingers_v0_1)
    --variant parts    separate body/left/right/face text files,
                       self-contained events, part-local segmentation, plus an
                       events/<clip>.json span contract for time-aligned
                       text->pose injection (contract signsmd2d_parts_v0_2,
                       doc/signsmd_part_paired_fusion_plan.md)

IDs match the Uni-Sign labels exactly (video_path minus .mp4), so the fused
training run has zero coverage gaps.  --max_length MUST equal the trainer's
--max_length (both default 256); it is recorded in the manifest.

Intended pipeline (OpenASL pretrain -> How2Sign finetune needs BOTH):

    # Compact fingers-only contract used by the dual-stream experiment
    python generate_signsmd2d.py --dataset OpenASL --splits train dev test --variant fingers --max_length 256

    # Generate the same contract for downstream How2Sign finetuning
    python generate_signsmd2d.py --dataset How2Sign --splits train test --variant fingers --max_length 256

    # quick smoke: 10 samples per variant, inspect the texts
    python generate_signsmd2d.py --dataset How2Sign --splits test --limit 10
    python generate_signsmd2d.py --dataset How2Sign --splits test --limit 10 --variant stats
    python generate_signsmd2d.py --dataset How2Sign --splits test --limit 10 --variant fingers

Output layout (default out_root data/<dataset>/<contract-version>):

    <out_root>/texts/<clip-name>.txt        (non-parts variants)
    <out_root>/texts/<part>/<clip-name>.txt (parts variant; four files)
    <out_root>/events/<clip-name>.json     (parts variant only)
    <out_root>/manifest_<dataset>_<split>[_shardK-N].json

Resume is the default: a parts clip is skipped only when all four .txt files
and its event JSON exist; other variants require their single .txt file.
Use --overwrite to regenerate complete outputs.
Sharding: --num_shards N --shard K processes every N-th sample (offset K),
so several processes can run in parallel on disjoint subsets.

This tool is numpy-only (no torch); labels are read directly as gzip
pickles.  The compiler never reads the caption text, and every generated
file passed a banned-vocabulary assertion (no depth/contact claims).
"""

import argparse
import gzip
import json
import os
import pickle
import sys
import time

from signsmd2d import (
    COARSE_VERSION,
    FINGERS_VERSION,
    PARTS_VERSION,
    STATS_VERSION,
)
from signsmd2d.compiler import (
    VARIANTS,
    compile_clip,
    load_thresholds,
    thresholds_sha256,
)
from signsmd2d.realization_parts import PARTS

VARIANT_INFO = {
    "coarse": {
        "version": COARSE_VERSION,
        "thresholds": "configs/signsmd2d_thresholds.yaml",
    },
    "stats": {
        "version": STATS_VERSION,
        "thresholds": "configs/signsmd2d_stats_thresholds.yaml",
    },
    "fingers": {
        "version": FINGERS_VERSION,
        "thresholds": "configs/signsmd2d_fingers_thresholds.yaml",
    },
    "parts": {
        "version": PARTS_VERSION,
        "thresholds": "configs/signsmd2d_parts_thresholds.yaml",
    },
}

LABEL_PATHS = {
    "How2Sign": {
        "train": "./data/How2Sign/labels.train",
        "test": "./data/How2Sign/labels.test",
    },
    "OpenASL": {
        "train": "./data/OpenASL/labels.train",
        "dev": "./data/OpenASL/labels.dev",
        "test": "./data/OpenASL/labels.test",
    },
}
POSE_DIRS = {
    # How2Sign pose dirs are per split; OpenASL is one flat directory.
    "How2Sign": "./data/How2Sign/{split}-pose-rtmpose-192",
    "OpenASL": "./data/OpenASL/pose-rtmpose-192",
}


def load_labels(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        "SignSMD-2D bulk generator", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        choices=sorted(LABEL_PATHS.keys()))
    parser.add_argument("--splits", nargs="+", required=True,
                        help="e.g. train test (How2Sign) or "
                             "train dev test (OpenASL)")
    parser.add_argument("--variant", default="coarse",
                        choices=sorted(VARIANTS),
                        help="coarse: interval prose (signsmd2d_v0_2); "
                             "stats: per-channel numeric timelines "
                             "(signsmd2d_stats_v0_1); fingers: compact "
                             "per-finger curl events only "
                             "(signsmd2d_fingers_v0_1); parts: per-part "
                             "self-contained events + events/<clip>.json "
                             "spans for time-aligned injection "
                             "(signsmd2d_parts_v0_2)")
    parser.add_argument("--out_root", default=None,
                        help="default: data/<dataset>/<contract-version>")
    parser.add_argument("--thresholds", default=None,
                        help="default: the variant's yaml under configs/")
    parser.add_argument("--max_length", type=int, default=256,
                        help="MUST equal the trainer's --max_length; the "
                             "step timeline is the linear downsampling of "
                             "each clip to at most this many steps")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="assumed FPS, recorded in the manifest")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N samples per split (0 = all)")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail_fast", action="store_true",
                        help="stop on the first failing sample")
    return parser.parse_args()


def process_split(args, cfg, split):
    label_path = LABEL_PATHS[args.dataset].get(split)
    if not label_path:
        raise ValueError(f"{args.dataset} has no split {split!r}")
    pose_dir = POSE_DIRS[args.dataset].format(split=split)

    version = VARIANT_INFO[args.variant]["version"]
    out_root = args.out_root or f"data/{args.dataset}/{version}"
    text_dir = os.path.join(out_root, "texts")
    os.makedirs(text_dir, exist_ok=True)
    events_dir = None
    part_text_dirs = None
    if args.variant == "parts":
        events_dir = os.path.join(out_root, "events")
        os.makedirs(events_dir, exist_ok=True)
        part_text_dirs = {
            part: os.path.join(text_dir, part) for part in PARTS
        }
        for part_dir in part_text_dirs.values():
            os.makedirs(part_dir, exist_ok=True)

    labels = load_labels(label_path)
    ids = sorted(labels.keys())
    if args.num_shards > 1:
        ids = ids[args.shard::args.num_shards]

    done = skipped = failed = 0
    failures = []
    ivp_sums = {}
    ivp_clips = 0
    start_time = time.time()

    for count, key in enumerate(ids):
        if args.limit and done + skipped + failed >= args.limit:
            break
        video_path = labels[key]["video_path"]
        clip = video_path[:-4] if video_path.endswith(".mp4") else video_path
        out_path = (
            None if part_text_dirs is not None
            else os.path.join(text_dir, clip + ".txt")
        )
        part_out_paths = (
            {
                part: os.path.join(part_dir, clip + ".txt")
                for part, part_dir in part_text_dirs.items()
            }
            if part_text_dirs is not None else None
        )
        event_path = (
            os.path.join(events_dir, clip + ".json") if events_dir else None
        )
        text_exists = (
            all(os.path.exists(path) for path in part_out_paths.values())
            if part_out_paths is not None else os.path.exists(out_path)
        )
        exists = text_exists and (
            event_path is None or os.path.exists(event_path)
        )
        if exists and not args.overwrite:
            skipped += 1
            continue

        pkl_path = os.path.join(pose_dir, clip + ".pkl")
        try:
            with open(pkl_path, "rb") as f:
                pose = pickle.load(f)
            result = compile_clip(
                pose, cfg, args.fps,
                variant=args.variant, max_length=args.max_length,
            )
            if part_out_paths is not None:
                part_texts = result.get("part_texts") or {}
                if set(part_texts) != set(PARTS):
                    raise AssertionError(
                        f"parts text keys {sorted(part_texts)} do not match "
                        f"contract parts {list(PARTS)}"
                    )
                for part in PARTS:
                    part_path = part_out_paths[part]
                    tmp_path = part_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(part_texts[part] + "\n")
                    os.replace(tmp_path, part_path)
            else:
                tmp_path = out_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(result["text"] + "\n")
                os.replace(tmp_path, out_path)
            if event_path is not None:
                payload = {
                    "contract": version,
                    "clip": clip,
                    "num_steps": result["num_steps"],
                    "num_frames_original": result["num_frames"],
                    "events": result["events"],
                }
                tmp_event = event_path + ".tmp"
                with open(tmp_event, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=1)
                os.replace(tmp_event, event_path)
            if result.get("intervals_per_part"):
                ivp_clips += 1
                for part, n in result["intervals_per_part"].items():
                    ivp_sums[part] = ivp_sums.get(part, 0) + n
            done += 1
        except Exception as exc:  # noqa: BLE001 - per-sample robustness
            failed += 1
            failures.append({"id": clip, "error": repr(exc)})
            if args.fail_fast:
                raise
        if (count + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{args.dataset}/{split}] {count + 1}/{len(ids)} "
                f"done={done} skipped={skipped} failed={failed} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    manifest = {
        "version": version,
        "variant": args.variant,
        "dataset": args.dataset,
        "split": split,
        "label_path": label_path,
        "pose_dir": pose_dir,
        "max_length": args.max_length,
        "fps": args.fps,
        "fps_provenance": "assumed",
        "thresholds": args.thresholds,
        "thresholds_sha256": thresholds_sha256(args.thresholds),
        "num_ids": len(ids),
        "completed": done,
        "skipped_existing": skipped,
        "failed": failed,
        "failures": failures[:200],
        "num_shards": args.num_shards,
        "shard": args.shard,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    if args.variant == "parts":
        manifest["text_layout"] = "texts/<part>/<clip>.txt"
        manifest["parts"] = list(PARTS)
        manifest["text_files_per_clip"] = len(PARTS)
    if ivp_clips:
        # Segmentation-granularity diagnostic (interval-collapse check,
        # doc/signsmd_part_paired_fusion_plan.md section 2.1): hand parts
        # averaging ~1 interval on multi-second clips means the spans carry
        # no temporal resolution and thresholds must be revisited.
        manifest["mean_intervals_per_part"] = {
            part: round(total / ivp_clips, 2)
            for part, total in sorted(ivp_sums.items())
        }
    shard_tag = (
        f"_shard{args.shard}-{args.num_shards}"
        if args.num_shards > 1 else ""
    )
    manifest_path = os.path.join(
        out_root, f"manifest_{args.dataset}_{split}{shard_tag}.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[{args.dataset}/{split}] finished: done={done} "
        f"skipped={skipped} failed={failed} -> {manifest_path}",
        flush=True,
    )
    return failed


def main():
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("--shard must be in [0, --num_shards)")
    if args.max_length < 1:
        raise ValueError("--max_length must be >= 1")
    if args.thresholds is None:
        args.thresholds = VARIANT_INFO[args.variant]["thresholds"]
    cfg = load_thresholds(
        args.thresholds,
        expected_version=VARIANT_INFO[args.variant]["version"],
    )
    total_failed = 0
    for split in args.splits:
        total_failed += process_split(args, cfg, split)
    sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
