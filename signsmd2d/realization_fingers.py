"""Compact SignSMD realization containing only per-finger curl channels.

The geometry and segmentation are identical to the stats variant, but this
contract drops body, wrist, hand-summary, and non-manual channels.  It also
keeps only the strongest few non-static events per finger so the auxiliary
text remains short enough to survive the model-side token cap.
"""

import numpy as np

from .realization import _valid_fraction, motion_header
from .realization_stats import _fmt, _span
from .stats_segments import analyze_channel


def _finger_sections(evidence):
    finger_names = (
        "Thumb", "Index finger", "Middle finger", "Ring finger", "Pinky",
    )
    sections = []
    for side in ("left", "right"):
        curls_deg = np.degrees(evidence["hands"][side]["curls"])
        channels = [
            {
                "key": "finger_curl",
                "label": f"{name} curl",
                "values": curls_deg[:, index],
                "unit": "°",
                "decimals": 0,
                "pos_verb": "curls",
                "neg_verb": "extends",
            }
            for index, name in enumerate(finger_names)
        ]
        sections.append((f"[{side.capitalize()} Fingers]", channels))
    return sections


def _compact_channel_lines(spec, result, max_events):
    unit, decimals = spec["unit"], spec["decimals"]
    lines = [f"  {spec['label']}:"]

    if result["status"] == "missing":
        return lines + ["    not detected throughout"]
    if result["status"] == "static":
        value = _fmt(result["mean"], decimals, unit)
        return lines + [f"    static at {value} throughout"]

    events = []
    for run in result["runs"]:
        if run["cyclic"] is not None:
            cycles, lo, hi, start, end = run["cyclic"]
            events.append((
                float(hi - lo), start,
                f"    repeats {cycles} cycles "
                f"({_fmt(lo, decimals, unit)} to "
                f"{_fmt(hi, decimals, unit)}) {_span(start, end)}",
            ))
            continue
        for segment in run["segments"]:
            if segment["type"] == "static":
                continue
            start, end = segment["start"], segment["end"]
            verb = (
                spec["pos_verb"] if segment["delta"] > 0
                else spec["neg_verb"]
            )
            events.append((
                abs(float(segment["delta"])), start,
                f"    {verb}: "
                f"{_fmt(segment['start_val'], decimals, unit)} → "
                f"{_fmt(segment['end_val'], decimals, unit)} "
                f"{_span(start, end)}",
            ))

    strongest = sorted(events, key=lambda event: event[0], reverse=True)
    strongest = strongest[:max_events]
    for _, _, line in sorted(strongest, key=lambda event: event[1]):
        lines.append(line)
    if len(lines) == 1:
        value = _fmt(result["mean"], decimals, unit)
        lines.append(f"    no substantial change (~{value})")
    return lines


def realize_fingers(evidence, cfg, fps, orig_num_frames=None):
    stats_cfg = cfg["stats"]
    num_steps = evidence["num_frames"]
    header = motion_header(num_steps, orig_num_frames, fps)

    if _valid_fraction(evidence["frame"]["frame_valid"]) < 0.2:
        return (
            f"{header}\n\n"
            "Finger keypoints were not reliably detected; no finger "
            "articulation description is available for this clip."
        )

    common = dict(
        smooth_window=int(stats_cfg["smooth_window"]),
        static_min_steps=int(stats_cfg["static_min_steps"]),
        min_valid_run=int(stats_cfg["min_valid_run"]),
        gap_report_min=int(stats_cfg["gap_report_min"]),
        activity_window=max(10, int(stats_cfg["smooth_window"]) * 2),
    )
    finger_cfg = stats_cfg["channels"]["finger_curl"]
    max_events = int(stats_cfg.get("max_events_per_finger", 2))
    if max_events < 1:
        raise ValueError("stats.max_events_per_finger must be >= 1")

    lines = [header, ""]
    for section, channels in _finger_sections(evidence):
        lines.append(section)
        for spec in channels:
            result = analyze_channel(
                np.asarray(spec["values"], dtype=np.float64),
                min_change=float(finger_cfg["min_change"]),
                static_threshold=float(finger_cfg["static_threshold"]),
                **common,
            )
            lines.extend(_compact_channel_lines(spec, result, max_events))
        lines.append("")
    return "\n".join(lines).strip()
