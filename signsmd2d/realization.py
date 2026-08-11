"""Compact target-free training text from evidence + intervals.

Vocabulary rules (enforced by compiler.assert_clean):
- orientation is camera-relative ("palm appears to face the camera"),
  never body-relative depth ("forward"/"backward" are banned words);
- inter-hand proximity is "overlap in the image", never "contact";
- a hand missing from detection is reported under Visibility and is never
  described as holding its configuration.

Interval spans are rendered as inclusive ``[step a-b]`` indices into the
trainer-aligned pose timeline (the compiler resamples with the trainer's
rule before any evidence is computed), so step i of the text IS pose token
i of the model input.
"""

import numpy as np


def motion_header(num_steps, orig_num_frames, fps):
    """Shared header for both variants; steps = model pose-sequence length."""
    orig = orig_num_frames if orig_num_frames is not None else num_steps
    if orig != num_steps:
        header = (
            f"Motion: {num_steps} steps "
            f"(linearly downsampled from {orig} frames"
        )
    else:
        header = f"Motion: {num_steps} steps ({orig} frames"
    if fps:
        header += f", ~{orig / fps:.1f}s at {fps:g} FPS"
    return header + ")"


def _nanmean(a):
    with np.errstate(all="ignore"):
        v = np.nanmean(a)
    return float(v) if np.isfinite(v) else np.nan


def _valid_fraction(mask):
    return float(mask.mean()) if len(mask) else 0.0


def handshape_phrase(curls, spread, opposition, cfg):
    hands_cfg = cfg["hands"]
    if np.isnan(curls).all():
        return None
    ext = curls < hands_cfg["curl_extended_max"]
    closed = curls > hands_cfg["curl_closed_min"]
    thumb_ext, fingers_ext = ext[0], ext[1:]
    fingers_closed = closed[1:]

    if fingers_ext.all():
        phrase = "open flat handshape"
        if np.isfinite(spread) and spread > hands_cfg["spread_open_min"]:
            phrase += " with fingers spread"
        return phrase
    if fingers_closed.all() and not thumb_ext:
        return "fist"
    if (
        ext[1]
        and fingers_closed[1:].all()
    ):
        return "fist with the index finger extended"
    if ext[1] and ext[2] and fingers_closed[2:].all():
        return "handshape with index and middle fingers extended"
    if (
        np.isfinite(opposition)
        and opposition < hands_cfg["opposition_pinch_max"]
        and not fingers_closed.any()
    ):
        return "pinching or rounded handshape"
    if not fingers_ext.any() and not fingers_closed.any():
        return "curved or cupped handshape"
    # No confident category: say nothing rather than emit a junk label
    # ("mixed handshape" would teach the model that uncertainty is a
    # handshape); interval dynamics are reported by curl_dynamics instead.
    return None


def palm_phrase(cross, span, cfg):
    if not np.isfinite(cross) or not np.isfinite(span):
        return None
    if span < cfg["hands"]["edge_on_span_max"]:
        return "palm edge-on to the camera"
    if cross < -0.15:
        return "palm appears to face the camera"
    if cross > 0.15:
        return "palm appears to face away from the camera"
    return None


def direction_phrase(vec):
    if vec is None or not np.all(np.isfinite(vec)):
        return None
    x, y = float(vec[0]), float(vec[1])
    if x == 0 and y == 0:
        return None
    words = []
    if abs(y) >= 0.4 * abs(x):
        words.append("upward" if y > 0 else "downward")
    if abs(x) >= 0.4 * abs(y):
        words.append(
            "toward signer-right" if x > 0 else "toward signer-left"
        )
    return " and ".join(words) if words else None


def zone_phrase(center, nose, cfg):
    if center is None or not np.all(np.isfinite(center)):
        return None
    x, y = float(center[0]), float(center[1])
    space = cfg["space"]
    if y > space["upper_y"]:
        vertical = "at head height"
    elif y < space["lower_y"]:
        vertical = "in lower signing space"
    else:
        vertical = "at chest height"
    if x > space["side_x"]:
        lateral = "on the signer-right side"
    elif x < -space["side_x"]:
        lateral = "on the signer-left side"
    else:
        lateral = "near the body midline"
    phrase = f"{vertical} {lateral}"
    if nose is not None and np.all(np.isfinite(nose)):
        if np.linalg.norm(center - nose) < space["near_face_dist"]:
            phrase += ", close to the face"
    return phrase


def curl_dynamics_phrase(curls, cfg):
    """Within-interval handshape DYNAMICS from the summed 5-finger curl
    trajectory [n,5] (radians): net curling/extending, or repeated change
    with no net drift.  Returns None for a stable hand."""
    with np.errstate(all="ignore"):
        total = 5.0 * np.nanmean(curls, axis=1)
    total = total[np.isfinite(total)]
    if len(total) < 3:
        return None
    thr = float(cfg["hands"].get("curl_change_min", 1.0))
    travel = float(np.abs(np.diff(total)).sum())
    net = float(total[-1] - total[0])
    if travel < thr:
        return None
    if abs(net) >= thr:
        return ("fingers curling overall" if net > 0
                else "fingers extending overall")
    if travel >= 2 * thr:
        return "handshape changing repeatedly"
    return None


def arm_phrase(bend, cfg):
    if not np.isfinite(bend):
        return None
    arm_cfg = cfg["arm"]
    if bend < arm_cfg["bend_slight_max"]:
        return "arm nearly extended"
    if bend < arm_cfg["bend_partial_max"]:
        return "arm partially bent"
    return "arm strongly bent"


def _movement_summary(wrist, valid, cfg):
    move_cfg = cfg["movement"]
    pts = wrist[valid]
    if len(pts) < 3:
        return None
    steps = np.diff(pts, axis=0)
    step_norm = np.linalg.norm(steps, axis=-1)
    path = float(step_norm.sum())
    frames = len(pts)
    speed = path / max(frames - 1, 1)
    if speed <= move_cfg["hold_speed_max"]:
        return {"kind": "hold", "path": path}
    net_vec = pts[-1] - pts[0]
    net = float(np.linalg.norm(net_vec))
    straight = net / max(path, 1e-8)
    main_dir = net_vec / max(net, 1e-8)
    proj = steps @ main_dir
    moving = step_norm > move_cfg["hold_speed_max"]
    signs = np.sign(proj[moving])
    reversals = int(np.abs(np.diff(signs)).sum() // 2) if signs.size else 0
    if speed >= move_cfg["fast_min"]:
        speed_word = "fast"
    elif speed >= move_cfg["slow_max"]:
        speed_word = "moderate"
    else:
        speed_word = "slow"
    return {
        "kind": "move",
        "path": path,
        "direction": direction_phrase(main_dir),
        "shape": (
            "straight" if straight >= move_cfg["straightness_min"]
            else "curved"
        ),
        "speed_word": speed_word,
        "repeated": reversals >= move_cfg["repetition_reversals"],
    }


class _HandState:
    """Categorical state of one hand within one interval."""

    def __init__(self, evidence, side, a, b, cfg):
        hand = evidence["hands"][side]
        arm = evidence["arms"][side]
        self.side = side
        self.valid_frac = _valid_fraction(hand["valid"][a:b])
        self.detected = self.valid_frac >= 0.3

        curls = np.array([
            _nanmean(hand["curls"][a:b, i]) for i in range(5)
        ])
        self.handshape = handshape_phrase(
            curls,
            _nanmean(hand["spread"][a:b]),
            _nanmean(hand["opposition"][a:b]),
            cfg,
        )
        self.curl_dynamics = curl_dynamics_phrase(hand["curls"][a:b], cfg)
        self.palm = palm_phrase(
            _nanmean(hand["palm_cross"][a:b]),
            _nanmean(hand["span_ratio"][a:b]),
            cfg,
        )
        finger_dir = hand["finger_dir"][a:b, 1:3].reshape(-1, 2)
        with np.errstate(all="ignore"):
            mean_dir = np.nanmean(finger_dir, axis=0)
        self.fingers = direction_phrase(mean_dir)

        center = np.array([
            _nanmean(hand["center_body"][a:b, 0]),
            _nanmean(hand["center_body"][a:b, 1]),
        ])
        nose = np.array([
            _nanmean(evidence["face"]["nose_body"][a:b, 0]),
            _nanmean(evidence["face"]["nose_body"][a:b, 1]),
        ])
        self.zone = zone_phrase(center, nose, cfg)
        self.arm = arm_phrase(_nanmean(arm["elbow_bend"][a:b]), cfg)
        self.movement = _movement_summary(
            arm["wrist_body"][a:b], arm["wrist_valid"][a:b], cfg
        )
        self.invalid_run = _longest_false_run(hand["valid"][a:b])

    def config_sentence(self):
        parts = [p for p in (
            self.handshape,
            self.curl_dynamics,
            self.zone,
            self.palm,
            f"fingers directed {self.fingers}" if self.fingers else None,
            self.arm,
        ) if p]
        return "; ".join(parts) if parts else None

    def changes_vs(self, prev):
        changes = []
        if self.handshape and prev.handshape and \
                self.handshape != prev.handshape:
            changes.append(f"handshape changes to {self.handshape}")
        if self.curl_dynamics and self.curl_dynamics != prev.curl_dynamics:
            changes.append(self.curl_dynamics)
        if self.palm and prev.palm and self.palm != prev.palm:
            changes.append(f"{self.palm}")
        if self.zone and prev.zone and self.zone != prev.zone:
            changes.append(f"moves to {self.zone}")
        return changes


def _longest_false_run(mask):
    longest = run = 0
    for ok in mask:
        run = 0 if ok else run + 1
        longest = max(longest, run)
    return longest


def _articulation_line(state):
    side_name = f"{state.side.capitalize()} hand"
    if not state.detected:
        return None
    m = state.movement
    if m is None:
        return None
    if m["kind"] == "hold":
        return f"{side_name} holds its position."
    sentence = f"{side_name} moves"
    if m["direction"]:
        sentence += f" {m['direction']}"
    sentence += f" along a {m['shape']} path at {m['speed_word']} speed"
    if m["repeated"]:
        sentence += ", with repeated direction reversals"
    return sentence + "."


def _coordination_lines(evidence, signals, a, b, states, cfg):
    lines = []
    move_cfg = cfg["movement"]
    sep = signals["separation"][a:b]
    sep_ok = np.isfinite(sep)
    if sep_ok.sum() >= 3:
        sep_valid = sep[sep_ok]
        delta = float(sep_valid[-1] - sep_valid[0])
        if delta > move_cfg["separation_change_min"]:
            lines.append("hand separation increases")
        elif delta < -move_cfg["separation_change_min"]:
            lines.append("hand separation decreases")
        if float(np.min(sep_valid)) < cfg["space"]["overlap_dist"]:
            lines.append("the hands overlap in the image")

    left, right = states["left"], states["right"]
    if left.detected and right.detected:
        lm, rm = left.movement, right.movement
        if lm and rm:
            if lm["kind"] == "move" and rm["kind"] == "hold":
                lines.append(
                    "the left hand articulates while the right hand "
                    "provides a base"
                )
            elif rm["kind"] == "move" and lm["kind"] == "hold":
                lines.append(
                    "the right hand articulates while the left hand "
                    "provides a base"
                )
        lx = evidence["hands"]["left"]["center_body"][a:b]
        rx = evidence["hands"]["right"]["center_body"][a:b]
        mirror_err = _nanmean(
            np.abs(lx[:, 0] + rx[:, 0]) + np.abs(lx[:, 1] - rx[:, 1])
        )
        if np.isfinite(mirror_err) and mirror_err < 0.3:
            lines.append(
                "approximately mirror-symmetric construction about the "
                "body midline"
            )
    return lines


def _nonmanual_line(evidence, a, b, cfg):
    face = evidence["face"]
    nm_cfg = cfg["nonmanual"]
    parts = []
    brow = _nanmean(face["brow"][a:b])
    if np.isfinite(brow) and np.isfinite(face["brow_neutral"]):
        delta = brow - face["brow_neutral"]
        if delta > nm_cfg["brow_raised_min"]:
            parts.append("eyebrows raised")
        elif delta < -nm_cfg["brow_raised_min"]:
            parts.append("eyebrows lowered")
    mouth = _nanmean(face["mouth_open"][a:b])
    if np.isfinite(mouth) and np.isfinite(face["mouth_open_neutral"]):
        if mouth - face["mouth_open_neutral"] > nm_cfg["mouth_open_min"]:
            parts.append("mouth open")
    tilt = _nanmean(face["tilt"][a:b])
    if np.isfinite(tilt) and tilt > nm_cfg["tilt_min"]:
        sign = _nanmean(face["tilt_sign"][a:b])
        towards = "signer-left" if sign >= 0 else "signer-right"
        parts.append(f"head tilted toward {towards}")
    if not parts:
        return None
    return "Non-manual: " + "; ".join(parts) + "."


def _visibility_line(states, cfg):
    min_frames = int(cfg["visibility"]["min_report_frames"])
    notes = []
    for side in ("left", "right"):
        state = states[side]
        if not state.detected:
            notes.append(f"{side} hand not detected")
        elif state.invalid_run >= min_frames:
            notes.append(
                f"{side} hand not detected for {state.invalid_run} frames"
            )
    if not notes:
        return None
    return "Visibility: " + "; ".join(notes) + "."


def realize(evidence, intervals, signals, cfg, fps, orig_num_frames=None):
    num_frames = evidence["num_frames"]
    header = motion_header(num_frames, orig_num_frames, fps)

    if _valid_fraction(evidence["frame"]["frame_valid"]) < 0.2:
        return (
            f"{header}\n\n"
            "Signer keypoints were not reliably detected; no articulation "
            "description is available for this clip."
        )

    labels = ", ".join(iv["label"] for iv in intervals)
    lines = [
        header,
        "",
        f"Temporal Structure: {len(intervals)} intervals ({labels})",
        "",
        "Sign Structure:",
    ]

    # global functional roles from whole-clip wrist path lengths
    role_line = _roles_line(evidence, cfg)

    prev_states = None
    for k, iv in enumerate(intervals):
        a, b = iv["start"], iv["end"]
        title = "Initial Phase" if k == 0 else f"Interval {k + 1}"
        lines.append(
            f" [{title}: step {a}–{b - 1} | {iv['label']}]"
        )

        states = {
            side: _HandState(evidence, side, a, b, cfg)
            for side in ("left", "right")
        }

        if k == 0:
            if role_line:
                lines.append(f"  Functional Roles: {role_line}")
            config_lines = []
            for side in ("left", "right"):
                if states[side].detected:
                    sentence = states[side].config_sentence()
                    if sentence:
                        config_lines.append(
                            f"   {side.capitalize()} Hand: {sentence}."
                        )
            if config_lines:
                lines.append("  Initial Configuration:")
                lines.extend(config_lines)
        else:
            change_lines = []
            for side in ("left", "right"):
                if states[side].detected and prev_states[side].detected:
                    changes = states[side].changes_vs(prev_states[side])
                    if changes:
                        change_lines.append(
                            f"   {side.capitalize()} Hand: "
                            + "; ".join(changes) + "."
                        )
            if change_lines:
                lines.append("  Configuration Changes:")
                lines.extend(change_lines)

        articulation = [
            _articulation_line(states[side]) for side in ("left", "right")
        ]
        articulation = [s for s in articulation if s]
        if articulation:
            lines.append("  Articulation:")
            lines.extend(f"   {s}" for s in articulation)

        coord = _coordination_lines(
            evidence, signals, a, b, states, cfg
        )
        if coord:
            lines.append("  Coordination: " + "; ".join(coord) + ".")

        nonmanual = _nonmanual_line(evidence, a, b, cfg)
        if nonmanual:
            lines.append("  " + nonmanual)

        visibility = _visibility_line(states, cfg)
        if visibility:
            lines.append("  " + visibility)

        prev_states = states
        if k != len(intervals) - 1:
            lines.append("")

    return "\n".join(lines)


def _roles_line(evidence, cfg):
    paths = {}
    for side in ("left", "right"):
        arm = evidence["arms"][side]
        wrist = arm["wrist_body"][arm["wrist_valid"]]
        if len(wrist) < 3:
            paths[side] = 0.0
        else:
            paths[side] = float(
                np.linalg.norm(np.diff(wrist, axis=0), axis=-1).sum()
            )
    ratio = float(cfg["roles"]["dominant_ratio"])
    left_path, right_path = paths["left"], paths["right"]
    if right_path > ratio * max(left_path, 1e-6):
        return (
            "the right hand acts as the dominant articulator; the left "
            "hand serves as the supporting or base hand."
        )
    if left_path > ratio * max(right_path, 1e-6):
        return (
            "the left hand acts as the dominant articulator; the right "
            "hand serves as the supporting or base hand."
        )
    if left_path > 0 and right_path > 0:
        return "both hands articulate actively."
    return None
