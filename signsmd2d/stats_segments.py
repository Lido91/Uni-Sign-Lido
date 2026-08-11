"""NaN-aware peak/valley + static segmentation for the stats variant.

Port of motion-smd ``scripts/utils/test_peak_valley.py``
(``find_segments_with_static`` / ``detect_cyclic``), adapted to the
validity-masked 2D sign evidence:

- segmentation runs independently inside every contiguous finite run of the
  channel (motion-smd assumes dense signals; our channels have NaN where the
  detector failed);
- missing runs surface as explicit ``gap`` records ("not detected") instead
  of being interpolated over;
- no angle unwrapping — no 2D channel here wraps at +-180 degrees.

All indices are steps of the trainer-aligned pose timeline.  numpy-only.
"""

import numpy as np


def _smooth(values, window):
    if window <= 1 or len(values) <= window:
        return values.copy()
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    half = window // 2
    for i in range(half):
        smoothed[i] = np.mean(values[: i + half + 1])
        smoothed[-(i + 1)] = np.mean(values[-(i + half + 1):])
    return smoothed


def _local_std(values, window):
    n = len(values)
    activity = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        activity[i] = np.std(values[lo:hi])
    return activity


def _find_extrema(smoothed, start, end, min_change):
    """Peak/valley step indices within [start, end) of a smoothed signal."""
    raw = [start]
    for i in range(start + 1, end - 1):
        if (smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]
                and smoothed[i] > smoothed[i - 1]):
            raw.append(i)
        elif (smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]
                and smoothed[i] < smoothed[i - 1]):
            raw.append(i)
    raw.append(end - 1)
    raw = sorted(set(raw))

    filtered = [raw[0]]
    for idx in raw[1:]:
        if abs(smoothed[idx] - smoothed[filtered[-1]]) >= min_change:
            filtered.append(idx)
    if filtered[-1] != end - 1:
        if abs(smoothed[end - 1] - smoothed[filtered[-1]]) >= min_change:
            filtered.append(end - 1)
        else:
            filtered[-1] = end - 1
    return filtered


def _segments_in_run(values, offset, min_change, static_threshold,
                     smooth_window, static_min_steps, activity_window):
    """Segment one finite run; returned step indices are offset to absolute.

    Mirrors motion-smd find_segments_with_static: smooth -> sliding-window
    std activity -> active/static classification -> peak/valley split of
    active regions.  ``end`` is exclusive.
    """
    n = len(values)
    if n < 3:
        return []

    smoothed = _smooth(values, smooth_window)
    activity = _local_std(smoothed, activity_window)
    is_static = activity < static_threshold

    regions = []
    i = 0
    while i < n:
        j = i
        while j < n and is_static[j] == is_static[i]:
            j += 1
        if is_static[i] and j - i >= static_min_steps:
            regions.append(("static", i, j))
        else:
            regions.append(("active", i, j))
        i = j

    merged = []
    for r in regions:
        if merged and merged[-1][0] == "active" and r[0] == "active":
            merged[-1] = ("active", merged[-1][1], r[2])
        else:
            merged.append(r)

    output = []
    for rtype, start, end in merged:
        if rtype == "static":
            avg = float(np.mean(smoothed[start:end]))
            output.append({
                "type": "static",
                "start": start + offset, "end": end + offset,
                "start_val": avg, "end_val": avg, "delta": 0.0,
            })
        else:
            extrema = _find_extrema(smoothed, start, end, min_change)
            for k in range(len(extrema) - 1):
                si, ei = extrema[k], extrema[k + 1]
                sv, ev = float(smoothed[si]), float(smoothed[ei])
                delta = ev - sv
                if abs(delta) >= min_change * 0.5:
                    output.append({
                        "type": "moving",
                        "start": si + offset, "end": ei + 1 + offset,
                        "start_val": sv, "end_val": ev, "delta": delta,
                    })
    return output


def detect_cyclic(segments, min_amp, cycle_threshold=0.6):
    """Port of motion-smd detect_cyclic with a channel-relative amplitude
    floor.  Returns (is_cyclic, n_cycles, mean_amplitude)."""
    moving = [s for s in segments if s["type"] == "moving"]
    if len(moving) < 4:
        return False, 0, 0.0
    directions = [1 if s["delta"] > 0 else -1 for s in moving]
    if any(directions[i] == directions[i + 1]
           for i in range(len(directions) - 1)):
        return False, 0, 0.0
    amplitudes = [abs(s["delta"]) for s in moving]
    mean_amp = float(np.mean(amplitudes))
    if mean_amp < min_amp:
        return False, 0, 0.0
    consistent = sum(
        1 for a in amplitudes if abs(a - mean_amp) / mean_amp < cycle_threshold
    )
    if consistent >= len(amplitudes) * 0.6:
        return True, len(moving) // 2, mean_amp
    return False, 0, 0.0


def analyze_channel(values, min_change, static_threshold, smooth_window,
                    static_min_steps, min_valid_run, gap_report_min,
                    activity_window):
    """Full NaN-aware analysis of one channel over the step timeline.

    Returns dict:
      status: "missing" (no usable finite run) | "static" | "segmented"
      mean: finite mean (nan if none)
      runs: list of {"segments": [...], "cyclic": (n, lo, hi, a, b) or None}
      gaps: [(start, end_exclusive), ...] missing spans >= gap_report_min
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    n = len(values)

    gaps = []
    runs_idx = []
    i = 0
    while i < n:
        j = i
        while j < n and finite[j] == finite[i]:
            j += 1
        if finite[i]:
            if j - i >= min_valid_run:
                runs_idx.append((i, j))
        elif j - i >= gap_report_min:
            gaps.append((i, j))
        i = j

    if not runs_idx:
        return {"status": "missing", "mean": float("nan"),
                "runs": [], "gaps": gaps}

    finite_vals = values[finite]
    mean = float(np.mean(finite_vals))
    if float(np.max(finite_vals) - np.min(finite_vals)) < min_change:
        return {"status": "static", "mean": mean, "runs": [], "gaps": gaps}

    runs = []
    for a, b in runs_idx:
        segments = _segments_in_run(
            values[a:b], a, min_change, static_threshold,
            smooth_window, static_min_steps, activity_window,
        )
        is_cyc, n_cycles, _ = detect_cyclic(segments, min_amp=min_change)
        cyclic = None
        if is_cyc and n_cycles >= 2:
            moving = [s for s in segments if s["type"] == "moving"]
            vals = ([s["start_val"] for s in moving]
                    + [s["end_val"] for s in moving])
            cyclic = (n_cycles, min(vals), max(vals),
                      moving[0]["start"], moving[-1]["end"])
        runs.append({"segments": segments, "cyclic": cyclic})

    return {"status": "segmented", "mean": mean, "runs": runs, "gaps": gaps}
