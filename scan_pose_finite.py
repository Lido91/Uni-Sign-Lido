"""Scan pose pickles for NaN/inf values that would poison the training loss.

The in-training guard (fine_tuning.py train_one_epoch) can only report the
batch it died on; this script sweeps a whole split offline so a single bad
pickle can be found and fixed/excluded before relaunching.

Usage (CPU only, no torch):

    python scan_pose_finite.py --dataset OpenASL --splits train dev test
    python scan_pose_finite.py --dataset How2Sign --splits train test
"""

import argparse
import gzip
import os
import pickle

import numpy as np

from config import (
    dev_label_paths,
    pose_dirs,
    test_label_paths,
    train_label_paths,
)

LABEL_PATHS = {
    "train": train_label_paths,
    "dev": dev_label_paths,
    "test": test_label_paths,
}


def load_labels(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def check_sample(pose_path):
    """Return a list of problem strings for one pose pickle ('' = fine)."""
    problems = []
    with open(pose_path, "rb") as f:
        pose = pickle.load(f)

    for field in ("keypoints", "scores"):
        if field not in pose:
            problems.append(f"missing field {field!r}")
            continue
        try:
            arr = np.asarray(pose[field], dtype=np.float64)
        except Exception:
            # ragged per-frame arrays: fall back to per-frame checks
            bad_frames = [
                i for i, frame in enumerate(pose[field])
                if not np.isfinite(np.asarray(frame, dtype=np.float64)).all()
            ]
            if bad_frames:
                problems.append(f"{field}: non-finite in frames {bad_frames[:10]}")
            continue
        if arr.size == 0:
            problems.append(f"{field}: empty")
        elif not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            problems.append(f"{field}: {n_bad} non-finite values")

    if "start" in pose:
        if not pose["start"] < pose["end"]:
            problems.append(f"start={pose['start']} !< end={pose['end']}")
    else:
        if len(pose.get("scores", [])) == 0:
            problems.append("zero-length clip (no scores)")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=["OpenASL", "How2Sign"])
    parser.add_argument("--splits", nargs="+", default=["train"],
                        choices=["train", "dev", "test"])
    args = parser.parse_args()

    total_bad = 0
    for split in args.splits:
        label_path = LABEL_PATHS[split][args.dataset]
        if not label_path:
            print(f"[{split}] no label file for {args.dataset}, skipping")
            continue
        pose_dir = pose_dirs[args.dataset].format(split)

        raw = load_labels(label_path)
        print(f"[{split}] {len(raw)} samples, pose dir: {pose_dir}")

        n_missing = n_bad = 0
        for i, (key, sample) in enumerate(raw.items()):
            rel = sample["video_path"].replace(".mp4", ".pkl")
            pose_path = os.path.join(pose_dir, rel)
            if not os.path.isfile(pose_path):
                n_missing += 1
                print(f"  MISSING {rel}")
                continue
            try:
                problems = check_sample(pose_path)
            except Exception as e:
                problems = [f"unreadable ({type(e).__name__}: {e})"]
            if problems:
                n_bad += 1
                print(f"  BAD {rel}: {'; '.join(problems)}")
            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1}/{len(raw)} scanned "
                      f"({n_bad} bad, {n_missing} missing so far)")

        print(f"[{split}] done: {n_bad} bad, {n_missing} missing "
              f"out of {len(raw)}")
        total_bad += n_bad + n_missing

    if total_bad == 0:
        print("All pose files finite and present — data is clean; "
              "look at the optimizer side instead.")
    else:
        print(f"TOTAL problems: {total_bad}")


if __name__ == "__main__":
    main()
