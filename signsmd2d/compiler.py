"""Per-clip SignSMD-2D compilation: RTMPose pickle dict -> training text.

Step-aligned timeline: before computing any evidence the clip is resampled
with EXACTLY the trainer's rule (datasets_signsmd.load_pose under the
default ``--frame_sampling linear``), so every ``[step a-b]`` span rendered
by either variant indexes the model's pose sequence directly and all rates
are per-step — what the model actually sees at its input.

Leakage rules enforced here:
- the compiler never receives caption/gloss text, only keypoints;
- assert_clean() rejects banned depth/contact vocabulary so the realization
  can never silently drift into claims 2D geometry cannot support.
"""

import hashlib
import re

import numpy as np
import yaml

from .evidence import compute_evidence
from .realization import realize
from .realization_fingers import realize_fingers
from .realization_parts import realize_parts
from .realization_stats import realize_stats
from .segmentation import segment

VARIANTS = ("coarse", "fingers", "stats", "parts")

# Depth or contact claims that 2D image geometry cannot support.
BANNED_PATTERN = re.compile(
    r"\b(forward|backward|contact|toward the body|away from the body)\b",
    re.IGNORECASE,
)


def load_thresholds(path, expected_version):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("version") != expected_version:
        raise ValueError(
            f"threshold manifest {path} has version "
            f"{cfg.get('version')!r}, expected {expected_version!r}"
        )
    return cfg


def thresholds_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def assert_clean(text):
    match = BANNED_PATTERN.search(text)
    if match:
        raise AssertionError(
            f"banned vocabulary {match.group(0)!r} in generated text"
        )


def resample_like_trainer(num_frames, max_length):
    """EXACT mirror of datasets_signsmd.S2T_Dataset_SignSMD.load_pose
    (``--frame_sampling linear``): clips longer than max_length are
    linearly downsampled, shorter clips are kept whole."""
    if num_frames > max_length:
        return np.round(
            np.linspace(0, num_frames - 1, max_length)
        ).astype(np.int64)
    return np.arange(num_frames)


def compile_clip(pose, cfg, fps, variant="coarse", max_length=256):
    """pose: dict with keypoints [T,1,133,2] and scores [T,1,133]."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    keypoints = np.asarray(pose["keypoints"], dtype=np.float64)
    scores = np.asarray(pose["scores"], dtype=np.float64)
    if keypoints.ndim == 4:
        keypoints = keypoints[:, 0]
        scores = scores[:, 0]
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 2):
        raise ValueError(f"unexpected keypoint shape {keypoints.shape}")

    # Same [start, end) window as the trainer, applied before resampling.
    if "start" in pose:
        a, b = int(pose["start"]), int(pose["end"])
        keypoints = keypoints[a:b]
        scores = scores[a:b]
    orig_frames = len(keypoints)
    if orig_frames == 0:
        raise ValueError("clip has zero frames")

    idx = resample_like_trainer(orig_frames, max_length)
    keypoints = keypoints[idx]
    scores = scores[idx]

    evidence = compute_evidence(keypoints, scores, cfg)
    events = None
    intervals_per_part = None
    part_texts = None
    if variant == "stats":
        text = realize_stats(evidence, cfg, fps, orig_num_frames=orig_frames)
        num_intervals = None
    elif variant == "fingers":
        text = realize_fingers(
            evidence, cfg, fps, orig_num_frames=orig_frames,
        )
        num_intervals = None
    elif variant == "parts":
        part_texts, events, intervals_per_part = realize_parts(
            evidence, cfg, fps, orig_num_frames=orig_frames,
        )
        text = None
        num_intervals = None
        num_steps = int(len(idx))
        for event in events:
            if not 0 <= event["start"] <= event["end"] < num_steps:
                raise AssertionError(
                    f"event span [{event['start']}, {event['end']}] outside "
                    f"[0, {num_steps}) for part {event['part']!r}"
                )
    else:
        intervals, signals = segment(evidence, cfg)
        text = realize(
            evidence, intervals, signals, cfg, fps,
            orig_num_frames=orig_frames,
        )
        num_intervals = len(intervals)
    if part_texts is not None:
        for part, part_text in part_texts.items():
            try:
                assert_clean(part_text)
            except AssertionError as exc:
                raise AssertionError(f"part {part!r}: {exc}") from exc
    else:
        assert_clean(text)
    return {
        "text": text,
        "num_frames": orig_frames,
        "num_steps": int(len(idx)),
        "num_intervals": num_intervals,
        "events": events,
        "intervals_per_part": intervals_per_part,
        "part_texts": part_texts,
    }
