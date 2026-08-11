"""Stats-variant realization: per-channel segmented numeric timelines in the
style of motion-smd ``angle_text_v5`` (holds / moves / repeats-N-cycles with
values), over the step-aligned 2D sign evidence.

Every span is an inclusive ``[step a-b]`` into the trainer-aligned pose
timeline; all channels come from the existing evidence dict (no new
geometry).  Vocabulary keeps the 2D contract: camera-relative palm proxy,
signer-relative directions, no depth/contact claims
(compiler.assert_clean applies to this variant too).
"""

import numpy as np

from .realization import motion_header, _valid_fraction
from .stats_segments import analyze_channel

LEGEND = (
    "Units: sw = shoulder width; ° = degrees; face measures in "
    "interocular units; steps index the model pose sequence (inclusive)."
)


def _fmt(value, decimals, unit):
    text = f"{value:.{decimals}f}"
    return text + unit


def _span(start, end_exclusive):
    return f"[step {start}–{end_exclusive - 1}]"


def _channel_series(evidence):
    """Ordered (section, channels) list; each channel is a spec dict."""
    arms = evidence["arms"]
    hands = evidence["hands"]
    face = evidence["face"]

    def ch(key, label, values, unit, decimals, pos_verb, neg_verb):
        return {
            "key": key, "label": label,
            "values": np.asarray(values, dtype=np.float64),
            "unit": unit, "decimals": decimals,
            "pos_verb": pos_verb, "neg_verb": neg_verb,
        }

    lw = arms["left"]["wrist_body"]
    rw = arms["right"]["wrist_body"]
    separation = np.linalg.norm(lw - rw, axis=-1)

    sections = [(
        "Bimanual:",
        [ch("separation", "Hand separation", separation, " sw", 2,
            "increases", "decreases")],
    )]

    finger_names = ("Thumb", "Index finger", "Middle finger",
                    "Ring finger", "Pinky")

    for side in ("left", "right"):
        title = side.capitalize()
        wrist = arms[side]["wrist_body"]
        hand = hands[side]
        curls_deg = np.degrees(hand["curls"])
        with np.errstate(all="ignore"):
            # hand-level summary: 5 x per-finger mean keeps the scale
            # stable when a subset of fingers drops out of detection
            total_curl = 5.0 * np.nanmean(curls_deg, axis=1)

        finger_channels = [
            # per-finger curl = summed 3-joint bend; ~<51° extended,
            # ~>126° closed (same convention as the coarse thresholds)
            ch("finger_curl", f"{name} curl", curls_deg[:, i], "°", 0,
               "curls", "extends")
            for i, name in enumerate(finger_names)
        ]

        sections.extend([
            (f"[{title} Wrist]", [
                ch("wrist_x", "Horizontal position (+ = signer-right)",
                   wrist[:, 0], " sw", 2,
                   "moves toward signer-right", "moves toward signer-left"),
                ch("wrist_y", "Vertical position (+ = up)",
                   wrist[:, 1], " sw", 2, "rises", "lowers"),
            ]),
            (f"[{title} Arm]", [
                ch("elbow_bend", "Elbow bend",
                   np.degrees(arms[side]["elbow_bend"]), "°", 0,
                   "bends", "straightens"),
            ]),
            (f"[{title} Hand]", [
                ch("total_curl", "Total finger curl (5 fingers)",
                   total_curl, "°", 0, "curls", "extends"),
                ch("spread", "Finger spread",
                   np.degrees(hand["spread"]), "°", 0,
                   "spreads", "narrows"),
                ch("opposition", "Thumb opposition (palm-scale units)",
                   hand["opposition"], "", 2, "opens", "closes"),
                ch("palm_cross",
                   "Palm orientation proxy "
                   "(+ = appears to face away from the camera)",
                   hand["palm_cross"], "", 2, "increases", "decreases"),
            ]),
            (f"[{title} Fingers]", finger_channels),
        ])

    tilt_signed = np.degrees(face["tilt"]) * face["tilt_sign"]
    sections.append((
        "Non-manual:",
        [
            ch("brow", "Brow height", face["brow"], "", 2,
               "rises", "lowers"),
            ch("mouth_open", "Mouth opening", face["mouth_open"], "", 2,
               "opens", "closes"),
            ch("tilt", "Head tilt (+ = toward signer-left)",
               tilt_signed, "°", 0,
               "tilts toward signer-left", "tilts toward signer-right"),
        ],
    ))
    return sections


def _channel_lines(spec, result, stats_cfg):
    unit, dec = spec["unit"], spec["decimals"]
    lines = [f"  {spec['label']}:"]

    if result["status"] == "missing":
        lines.append("    not detected throughout")
        return lines

    items = [(a, f"    not detected {_span(a, b)}")
             for a, b in result["gaps"]]

    if result["status"] == "static":
        lines.append(
            f"    static at {_fmt(result['mean'], dec, unit)} throughout"
        )
        for _, line in sorted(items):
            lines.append(line)
        return lines

    for run in result["runs"]:
        if run["cyclic"] is not None:
            n_cycles, lo, hi, a, b = run["cyclic"]
            items.append((a, (
                f"    repeats {n_cycles} cycles "
                f"({_fmt(lo, dec, unit)} to {_fmt(hi, dec, unit)}) "
                f"{_span(a, b)}"
            )))
            continue
        for seg in run["segments"]:
            a, b = seg["start"], seg["end"]
            if seg["type"] == "static":
                items.append((a, (
                    f"    holds at {_fmt(seg['start_val'], dec, unit)} "
                    f"{_span(a, b)} ({b - a} steps)"
                )))
            else:
                verb = (spec["pos_verb"] if seg["delta"] > 0
                        else spec["neg_verb"])
                items.append((a, (
                    f"    {verb}: {_fmt(seg['start_val'], dec, unit)} "
                    f"→ {_fmt(seg['end_val'], dec, unit)} {_span(a, b)}"
                )))

    for _, line in sorted(items, key=lambda item: item[0]):
        lines.append(line)
    if len(lines) == 1:
        lines.append(
            f"    ~{_fmt(result['mean'], dec, unit)} "
            "(active, but no segment above threshold)"
        )
    return lines


def realize_stats(evidence, cfg, fps, orig_num_frames=None):
    stats_cfg = cfg["stats"]
    num_steps = evidence["num_frames"]
    header = motion_header(num_steps, orig_num_frames, fps)

    if _valid_fraction(evidence["frame"]["frame_valid"]) < 0.2:
        return (
            f"{header}\n\n"
            "Signer keypoints were not reliably detected; no articulation "
            "description is available for this clip."
        )

    channel_cfg = stats_cfg["channels"]
    common = dict(
        smooth_window=int(stats_cfg["smooth_window"]),
        static_min_steps=int(stats_cfg["static_min_steps"]),
        min_valid_run=int(stats_cfg["min_valid_run"]),
        gap_report_min=int(stats_cfg["gap_report_min"]),
        activity_window=max(10, int(stats_cfg["smooth_window"]) * 2),
    )

    sections = _channel_series(evidence)

    analyzed = []
    for section, channels in sections:
        entries = []
        for spec in channels:
            ch_cfg = channel_cfg[spec["key"]]
            result = analyze_channel(
                spec["values"],
                min_change=float(ch_cfg["min_change"]),
                static_threshold=float(ch_cfg["static_threshold"]),
                **common,
            )
            finite = spec["values"][np.isfinite(spec["values"])]
            score = (
                float(np.max(finite) - np.min(finite))
                / float(ch_cfg["min_change"])
                if len(finite) else 0.0
            )
            entries.append((spec, result, score))
        analyzed.append((section, entries))

    top_k = int(stats_cfg.get("top_k", 0))
    if top_k > 0:
        scored = sorted(
            (entry for _, entries in analyzed for entry in entries
             if entry[0]["key"] != "separation"),
            key=lambda entry: entry[2], reverse=True,
        )
        keep = {id(entry) for entry in scored[:top_k]}
        analyzed = [
            (section, [e for e in entries
                       if e[0]["key"] == "separation" or id(e) in keep])
            for section, entries in analyzed
        ]

    lines = [header, LEGEND, ""]
    for section, entries in analyzed:
        if not entries:
            continue
        lines.append(section)
        for spec, result, _ in entries:
            lines.extend(_channel_lines(spec, result, stats_cfg))
        lines.append("")
    return "\n".join(lines).strip()
