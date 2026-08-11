"""Activity-driven temporal segmentation over validity-masked 2D evidence.

Per-frame activity is the max over threshold-normalized rates of every
sign-relevant articulator (per-side wrist speed, handshape/orientation/arm
rates, bimanual separation rate, non-manual rates).  Invalid frames
contribute nothing (NaN-aware); a missing hand can therefore never force a
"hold" label.  Every frame belongs to exactly one interval; intervals
shorter than min_interval_frames are merged one at a time.
"""

import numpy as np


def _rate(series):
    """|d/dt| with NaN propagation, aligned to frames (first frame 0)."""
    diff = np.abs(np.diff(series, axis=0))
    pad = np.zeros((1,) + series.shape[1:])
    return np.concatenate([pad, diff], axis=0)


def _nanmax_stack(components):
    stacked = np.stack(components, axis=-1)
    all_nan = np.isnan(stacked).all(-1)
    with np.errstate(all="ignore"):
        out = np.nanmax(stacked, axis=-1)
    out[all_nan] = 0.0
    return np.nan_to_num(out, nan=0.0)


def _smooth(series, window):
    if window <= 1:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="same")


def per_frame_signals(evidence, cfg):
    seg = cfg["segmentation"]
    signals = {}

    for side in ("left", "right"):
        arm = evidence["arms"][side]
        hand = evidence["hands"][side]

        speed = np.linalg.norm(_rate(arm["wrist_body"]), axis=-1)
        hs_rate = np.nansum(_rate(hand["curls"]), axis=-1)
        hs_rate = np.where(
            np.isnan(hand["curls"]).all(-1), np.nan, hs_rate
        )
        orient_rate = _rate(hand["palm_cross"])
        forearm_angle = np.arctan2(
            arm["forearm_dir"][:, 1], arm["forearm_dir"][:, 0]
        )
        # unwrap-free angular rate (shortest arc)
        da = np.abs(np.diff(forearm_angle))
        da = np.minimum(da, 2 * np.pi - da)
        arm_rate = np.concatenate([[0.0], da]) + _rate(arm["elbow_bend"])

        signals[f"{side}_speed"] = speed
        signals[f"{side}_activity"] = _nanmax_stack([
            speed / seg["wrist_speed"],
            hs_rate / seg["handshape_rate"],
            orient_rate / seg["orientation_rate"],
            arm_rate / seg["arm_rate"],
        ])
        signals[f"{side}_config_rate"] = _nanmax_stack([
            hs_rate / seg["handshape_rate"],
            orient_rate / seg["orientation_rate"],
        ])
        signals[f"{side}_position_rate"] = _nanmax_stack([
            speed / seg["wrist_speed"],
            arm_rate / seg["arm_rate"],
        ])

    lw = evidence["arms"]["left"]["wrist_body"]
    rw = evidence["arms"]["right"]["wrist_body"]
    separation = np.linalg.norm(lw - rw, axis=-1)
    signals["separation"] = separation
    signals["bimanual"] = _nanmax_stack(
        [_rate(separation) / seg["bimanual_rate"]]
    )

    face = evidence["face"]
    signals["nonmanual"] = _nanmax_stack([
        _rate(face["brow"]) / seg["brow_rate"],
        _rate(face["mouth_open"]) / seg["mouth_rate"],
        _rate(face["tilt"]) / seg["tilt_rate"],
    ])

    signals["sequence"] = _nanmax_stack([
        signals["left_activity"],
        signals["right_activity"],
        signals["bimanual"],
        signals["nonmanual"],
    ])
    return signals


def segment(evidence, cfg):
    """Return list of intervals: dict(start, end_exclusive, label)."""
    seg = cfg["segmentation"]
    num_frames = evidence["num_frames"]
    signals = per_frame_signals(evidence, cfg)
    active = _smooth(signals["sequence"], int(seg["smooth_window"])) > 1.0

    # runs of equal active-state
    bounds = [0]
    for t in range(1, num_frames):
        if active[t] != active[t - 1]:
            bounds.append(t)
    bounds.append(num_frames)

    intervals = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        intervals.append({
            "start": a,
            "end": b,
            "label": _label_interval(signals, a, b, active[a]),
        })

    min_len = int(seg["min_interval_frames"])
    while len(intervals) > 1:
        lengths = [iv["end"] - iv["start"] for iv in intervals]
        shortest = int(np.argmin(lengths))
        if lengths[shortest] >= min_len:
            break
        # merge into the longer neighbor
        if shortest == 0:
            into = 1
        elif shortest == len(intervals) - 1:
            into = shortest - 1
        else:
            left_len = lengths[shortest - 1]
            right_len = lengths[shortest + 1]
            into = shortest - 1 if left_len >= right_len else shortest + 1
        lo, hi = min(shortest, into), max(shortest, into)
        merged = {
            "start": intervals[lo]["start"],
            "end": intervals[hi]["end"],
            "label": intervals[into]["label"],
        }
        intervals[lo:hi + 1] = [merged]

    return intervals, signals


def _label_interval(signals, a, b, is_active):
    if not is_active:
        return "hold"
    pos = max(
        float(np.mean(signals["left_position_rate"][a:b])),
        float(np.mean(signals["right_position_rate"][a:b])),
    )
    config = max(
        float(np.mean(signals["left_config_rate"][a:b])),
        float(np.mean(signals["right_config_rate"][a:b])),
    )
    if pos >= 1.0 or pos >= config:
        return "movement"
    if config >= 1.0:
        return "transition"
    return "mixed"
