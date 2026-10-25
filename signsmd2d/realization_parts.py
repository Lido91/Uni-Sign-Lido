"""Per-part coarse-style events with part-local temporal segmentation.

Each part of the Uni-Sign pose encoder (body, left hand, right hand, face)
is segmented on its OWN activity channels — not the global max used by the
coarse variant, which is known to collapse whole clips into one interval —
and every (part, interval) is realized as one SELF-CONTAINED categorical
phrase with an inclusive [step a-b] span.

This is the corpus for the time-aligned text->pose injection experiment
(doc/signsmd_part_paired_fusion_plan.md): each phrase is embedded
independently and scattered onto pose steps a..b of the matching part
stream. Self-containment is therefore a hard requirement — unlike the
coarse variant there are no cross-interval deltas ("handshape changes to
...") and every hand phrase names its own hand.

The machine event contract is written to events/<clip>.json by
generate_signsmd2d.py.  Text is rendered separately for each part under
texts/<part>/<clip>.txt so every part has its own tokenizer/max-length budget.
"""

import numpy as np

from .realization import (
    _HandState,
    _coordination_lines,
    _movement_summary,
    _nanmean,
    _roles_line,
    _valid_fraction,
    arm_phrase,
    motion_header,
    zone_phrase,
)
from .segmentation import _nanmax_stack, _smooth, per_frame_signals

PARTS = ("body", "left", "right", "face_all")
SECTION_TITLES = {
    "body": "[Body]",
    "left": "[Left Hand]",
    "right": "[Right Hand]",
    "face_all": "[Face]",
}


def _part_activity(signals, cfg):
    """Threshold-normalized activity per part, matching what each part's
    events describe (and what each pose stream encodes): hands react to
    handshape/orientation change only; wrist position and arm motion
    belong to the body part (the hand streams receive location via the
    body-anchor addition in the pose encoder)."""
    return {
        "left": signals["left_config_rate"],
        "right": signals["right_config_rate"],
        "body": _nanmax_stack([
            signals["bimanual"],
            signals["left_position_rate"],
            signals["right_position_rate"],
        ]),
        "face_all": signals["nonmanual"],
    }


def _segment_activity(activity, cfg):
    """Mirror of segmentation.segment's run/merge logic on ONE activity
    series. Returns intervals: dict(start, end_exclusive, active).

    With parts_segmentation.adaptive_quantile > 0 the active threshold is
    max(1.0, per-clip quantile of the smoothed activity) — the escape
    hatch for the long-interval collapse (whole clip judged "active"):
    the quantile splits it by relative activity instead.
    """
    seg = cfg["segmentation"]
    pcfg = cfg.get("parts_segmentation") or {}
    num_frames = len(activity)
    smoothed = _smooth(activity, int(seg["smooth_window"]))
    threshold = 1.0
    quantile = float(pcfg.get("adaptive_quantile") or 0.0)
    if quantile > 0.0:
        threshold = max(threshold, float(np.quantile(smoothed, quantile)))
    active = smoothed > threshold

    bounds = [0]
    for t in range(1, num_frames):
        if active[t] != active[t - 1]:
            bounds.append(t)
    bounds.append(num_frames)
    intervals = [
        {"start": a, "end": b, "active": bool(active[a])}
        for a, b in zip(bounds[:-1], bounds[1:])
    ]

    min_len = int(seg["min_interval_frames"])
    while len(intervals) > 1:
        lengths = [iv["end"] - iv["start"] for iv in intervals]
        shortest = int(np.argmin(lengths))
        if lengths[shortest] >= min_len:
            break
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
            "active": intervals[into]["active"],
        }
        intervals[lo:hi + 1] = [merged]
    return intervals


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def _finger_list(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _finger_word(name):
    return name if name == "thumb" else f"{name} finger"


def _finger_state_clause(curl_means, cfg):
    """Per-finger extended/partially-curled/closed grouping from mean curls
    within the interval. Used when the handshape category leaves the finger
    configuration underdetermined."""
    hands_cfg = cfg["hands"]
    groups = (("extended", []), ("partially curled", []), ("closed", []))
    for i, name in enumerate(FINGER_NAMES):
        value = curl_means[i]
        if not np.isfinite(value):
            continue
        if value < hands_cfg["curl_extended_max"]:
            groups[0][1].append(name)
        elif value > hands_cfg["curl_closed_min"]:
            groups[2][1].append(name)
        else:
            groups[1][1].append(name)
    bits = [
        f"{_finger_list(names)} {state}" for state, names in groups if names
    ]
    if not bits:
        return None
    return "fingers: " + ", ".join(bits)


def _finger_motion_bits(curls_slice, cfg):
    """Strongest per-finger net curl changes within the interval, fingers
    variant style: |net| >= finger_curl_change_min, keep the
    finger_events_max largest, rendered in anatomical finger order."""
    hands_cfg = cfg["hands"]
    threshold = float(hands_cfg.get("finger_curl_change_min", 0.7))
    max_keep = int(hands_cfg.get("finger_events_max", 2))
    candidates = []
    for i, name in enumerate(FINGER_NAMES):
        series = curls_slice[:, i]
        finite = series[np.isfinite(series)]
        if len(finite) < 3:
            continue
        net = float(finite[-1] - finite[0])
        if abs(net) < threshold:
            continue
        verb = "curls" if net > 0 else "extends"
        candidates.append((abs(net), i, f"the {_finger_word(name)} {verb}"))
    candidates.sort(key=lambda c: -c[0])
    kept = sorted(candidates[:max_keep], key=lambda c: c[1])
    return [c[2] for c in kept]


def _hand_event_phrase(evidence, side, a, b, cfg):
    """Finger-only self-contained state for one hand interval: handshape
    category, per-finger detail, palm orientation, finger direction.
    Wrist location, arm bend and wrist motion deliberately EXCLUDED —
    those belong to body events, matching the pose encoder where hand
    streams carry articulation and location arrives via the body anchor.
    Returns None when the hand is not reliably detected."""
    state = _HandState(evidence, side, a, b, cfg)
    if not state.detected:
        return None

    curls_slice = evidence["hands"][side]["curls"][a:b]
    curl_means = np.array([
        _nanmean(curls_slice[:, i]) for i in range(5)
    ])
    # Finger-state clause only where the handshape category leaves the
    # configuration underdetermined (absent / curved / pinching); the
    # explicit categories (fist, open flat, index extended, ...) already
    # pin down every finger.
    finger_state = None
    if state.handshape is None or state.handshape.startswith(
        ("curved", "pinching")
    ):
        finger_state = _finger_state_clause(curl_means, cfg)
    finger_motions = _finger_motion_bits(curls_slice, cfg)
    # per-finger motions subsume the aggregate "fingers curling overall"
    curl_dynamics = None if finger_motions else state.curl_dynamics

    bits = [p for p in (
        state.handshape,
        finger_state,
        curl_dynamics,
        *finger_motions,
        state.palm,
        f"fingers directed {state.fingers}" if state.fingers else None,
    ) if p]
    if not bits:
        return None
    return f"{side} hand: " + "; ".join(bits)


def _arm_event(evidence, side, a, b, cfg):
    """One arm's wrist location + elbow bend + wrist motion as a body
    event. Returns (kind, phrase) or None."""
    arm = evidence["arms"][side]
    if _valid_fraction(arm["wrist_valid"][a:b]) < 0.3:
        return None
    wrist_mean = np.array([
        _nanmean(arm["wrist_body"][a:b, 0]),
        _nanmean(arm["wrist_body"][a:b, 1]),
    ])
    nose = np.array([
        _nanmean(evidence["face"]["nose_body"][a:b, 0]),
        _nanmean(evidence["face"]["nose_body"][a:b, 1]),
    ])
    zone = zone_phrase(wrist_mean, nose, cfg)
    bend = arm_phrase(_nanmean(arm["elbow_bend"][a:b]), cfg)
    move = _movement_summary(
        arm["wrist_body"][a:b], arm["wrist_valid"][a:b], cfg
    )
    kind = "hold"
    move_bit = None
    if move is not None:
        if move["kind"] == "hold":
            move_bit = "holds its position"
        else:
            kind = "movement"
            move_bit = "moves"
            if move["direction"]:
                move_bit += f" {move['direction']}"
            move_bit += (
                f" along a {move['shape']} path at {move['speed_word']} speed"
            )
            if move["repeated"]:
                move_bit += ", with repeated direction reversals"
    bits = [p for p in (
        f"wrist {zone}" if zone else None,
        bend,
        move_bit,
    ) if p]
    if not bits:
        return None
    return kind, f"{side} arm: " + "; ".join(bits)


def _body_interval_events(evidence, signals, a, b, active, cfg):
    """Body events for one interval: one per tracked arm (wrist zone +
    elbow bend + wrist motion) and one bimanual-coordination event."""
    out = []
    for side in ("left", "right"):
        arm_event = _arm_event(evidence, side, a, b, cfg)
        if arm_event is not None:
            out.append(arm_event)
    states = {
        side: _HandState(evidence, side, a, b, cfg)
        for side in ("left", "right")
    }
    lines = _coordination_lines(evidence, signals, a, b, states, cfg)
    if lines:
        out.append(("movement" if active else "hold", "; ".join(lines)))
    return out


def _face_event_phrase(evidence, a, b, cfg):
    """Absolute non-manual state for one interval (no 'Non-manual:'
    prefix, no neutral-delta wording that needs surrounding context)."""
    face = evidence["face"]
    if _valid_fraction(face["valid"][a:b]) < 0.3:
        return None
    nm_cfg = cfg["nonmanual"]
    bits = []
    brow = _nanmean(face["brow"][a:b])
    if np.isfinite(brow) and np.isfinite(face["brow_neutral"]):
        delta = brow - face["brow_neutral"]
        if delta > nm_cfg["brow_raised_min"]:
            bits.append("eyebrows raised")
        elif delta < -nm_cfg["brow_raised_min"]:
            bits.append("eyebrows lowered")
    mouth = _nanmean(face["mouth_open"][a:b])
    if np.isfinite(mouth) and np.isfinite(face["mouth_open_neutral"]):
        if mouth - face["mouth_open_neutral"] > nm_cfg["mouth_open_min"]:
            bits.append("mouth open")
    tilt = _nanmean(face["tilt"][a:b])
    if np.isfinite(tilt) and tilt > nm_cfg["tilt_min"]:
        sign = _nanmean(face["tilt_sign"][a:b])
        towards = "signer-left" if sign >= 0 else "signer-right"
        bits.append(f"head tilted toward {towards}")
    if not bits:
        return "facial expression near neutral"
    return "; ".join(bits)


def realize_parts(evidence, cfg, fps, orig_num_frames=None):
    """Returns (part_texts, events, intervals_per_part).

    part_texts: dict part -> standalone text containing only that part's
    events; no text file contains events from another part.
    events: list of dicts {part, kind, phrase, start, end} with INCLUSIVE
    trainer-aligned step spans; intervals_per_part: dict part -> interval
    count (segmentation granularity diagnostic, aggregated into the
    generation manifest to detect interval collapse).
    """
    num_frames = evidence["num_frames"]
    header = motion_header(num_frames, orig_num_frames, fps)

    if _valid_fraction(evidence["frame"]["frame_valid"]) < 0.2:
        part_texts = {
            part: f"{header}\n\n{SECTION_TITLES[part]}\n  none"
            for part in PARTS
        }
        return part_texts, [], {part: 0 for part in PARTS}

    signals = per_frame_signals(evidence, cfg)
    activity = _part_activity(signals, cfg)

    events = []
    intervals_per_part = {}
    for part in PARTS:
        intervals = _segment_activity(activity[part], cfg)
        intervals_per_part[part] = len(intervals)
        for iv in intervals:
            a, b = iv["start"], iv["end"]
            if part in ("left", "right"):
                kind = "transition" if iv["active"] else "hold"
                phrase = _hand_event_phrase(evidence, part, a, b, cfg)
                interval_events = [(kind, phrase)] if phrase else []
            elif part == "body":
                interval_events = _body_interval_events(
                    evidence, signals, a, b, iv["active"], cfg
                )
            else:
                kind = "movement" if iv["active"] else "hold"
                phrase = _face_event_phrase(evidence, a, b, cfg)
                interval_events = [(kind, phrase)] if phrase else []
            for kind, phrase in interval_events:
                events.append({
                    "part": part,
                    "kind": kind,
                    "phrase": phrase,
                    "start": int(a),
                    "end": int(b - 1),
                })

    role = _roles_line(evidence, cfg)
    if role:
        events.append({
            "part": "body",
            "kind": "role",
            "phrase": role.rstrip("."),
            "start": 0,
            "end": int(num_frames - 1),
        })

    part_texts = {}
    for part in PARTS:
        lines = [header, "", SECTION_TITLES[part]]
        part_events = sorted(
            (e for e in events if e["part"] == part),
            key=lambda e: (e["start"], e["end"]),
        )
        if not part_events:
            lines.append("  none")
        else:
            for event in part_events:
                lines.append(
                    f"  {event['phrase']} "
                    f"[step {event['start']}-{event['end']}]"
                )
        part_texts[part] = "\n".join(lines).rstrip()
    return part_texts, events, intervals_per_part
