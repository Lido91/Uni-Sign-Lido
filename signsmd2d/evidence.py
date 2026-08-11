"""Frame-level continuous evidence from RTMPose 2D keypoints.

Everything here is image-plane geometry with explicit validity masks:
a point with detector score <= confidence_min is MISSING (NaN + mask False),
never zero.  Gaps up to max_interp_frames are linearly interpolated; longer
gaps stay missing and surface later as visibility codes.

Coordinate conventions:
- raw keypoints are image pixels, y grows DOWNWARD;
- the body frame is built from the shoulders only (no pelvis/spine in this
  representation): origin = shoulder center, +x = signer-right
  (right_shoulder - left_shoulder), +y = up, unit = shoulder width;
- hand-local features are normalized by palm scale so handshape evidence is
  invariant to the hand's distance from the camera;
- all angles are projected (image-plane) proxies: constant foreshortening
  cancels in the *rates* used for segmentation, but absolute values are
  apparent, not anatomical.
"""

import numpy as np

from . import layout


def _unit(v, axis=-1, eps=1e-8):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


def _angle_between(u, v, eps=1e-8):
    """Unsigned angle between 2D vectors along the last axis."""
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    dot = (u * v).sum(-1)
    cos = dot / np.maximum(nu * nv, eps)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def interpolate_gaps(points, valid, max_gap):
    """Linearly fill invalid runs of length <= max_gap per keypoint.

    points: [T,K,2], valid: [T,K] bool.  Returns filled copies.
    """
    points = points.copy()
    valid = valid.copy()
    num_frames, num_points = valid.shape
    for k in range(num_points):
        col = valid[:, k]
        if col.all() or not col.any():
            continue
        idx = np.flatnonzero(col)
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                t = (np.arange(a + 1, b) - a) / (b - a)
                points[a + 1:b, k] = (
                    (1 - t)[:, None] * points[a, k][None]
                    + t[:, None] * points[b, k][None]
                )
                valid[a + 1:b, k] = True
    return points, valid


def body_frame(points, valid):
    """Shoulder-based 2D body frame per frame.

    Returns dict with origin [T,2], lateral/vertical unit axes [T,2],
    scale [T], frame_valid [T].
    """
    num_frames = points.shape[0]
    ls = points[:, layout.LEFT_SHOULDER]
    rs = points[:, layout.RIGHT_SHOULDER]
    frame_valid = (
        valid[:, layout.LEFT_SHOULDER] & valid[:, layout.RIGHT_SHOULDER]
    )

    origin = 0.5 * (ls + rs)
    lateral = rs - ls
    scale = np.linalg.norm(lateral, axis=-1)
    frame_valid &= scale > 1e-6
    lateral = _unit(lateral)

    # +y must point up for the signer: pick the perpendicular whose image-y
    # component is negative (image y grows downward); if the nose is valid it
    # confirms/overrides the choice.
    vertical = np.stack([lateral[:, 1], -lateral[:, 0]], axis=-1)
    flip = vertical[:, 1] > 0
    nose_ok = valid[:, layout.NOSE]
    to_nose = points[:, layout.NOSE] - origin
    nose_dot = (to_nose * vertical).sum(-1)
    flip = np.where(nose_ok, nose_dot < 0, flip)
    vertical = np.where(flip[:, None], -vertical, vertical)

    return {
        "origin": origin,
        "lateral": lateral,
        "vertical": vertical,
        "scale": np.where(frame_valid, scale, np.nan),
        "frame_valid": frame_valid,
    }


def to_body(points, frame):
    """Project image points [T,...,2] into body coordinates."""
    rel = points - frame["origin"].reshape(
        (-1,) + (1,) * (points.ndim - 2) + (2,)
    )
    lat = frame["lateral"].reshape(
        (-1,) + (1,) * (points.ndim - 2) + (2,)
    )
    vert = frame["vertical"].reshape(
        (-1,) + (1,) * (points.ndim - 2) + (2,)
    )
    scale = frame["scale"].reshape((-1,) + (1,) * (points.ndim - 2))
    x = (rel * lat).sum(-1) / scale
    y = (rel * vert).sum(-1) / scale
    return np.stack([x, y], axis=-1)


def _rotate_dir(vec, frame):
    """Direction (no scale) into body axes; vec [T,...,2]."""
    lat = frame["lateral"].reshape((-1,) + (1,) * (vec.ndim - 2) + (2,))
    vert = frame["vertical"].reshape((-1,) + (1,) * (vec.ndim - 2) + (2,))
    return np.stack([(vec * lat).sum(-1), (vec * vert).sum(-1)], axis=-1)


def arm_evidence(points, valid, frame, side):
    sh = points[:, layout.BODY_SHOULDER[side]]
    el = points[:, layout.BODY_ELBOW[side]]
    wr = points[:, layout.BODY_WRIST[side]]
    ok = (
        frame["frame_valid"]
        & valid[:, layout.BODY_SHOULDER[side]]
        & valid[:, layout.BODY_ELBOW[side]]
        & valid[:, layout.BODY_WRIST[side]]
    )
    bend = np.pi - _angle_between(sh - el, wr - el)
    forearm = _rotate_dir(_unit(wr - el), frame)
    wrist_body = to_body(wr, frame)
    bend = np.where(ok, bend, np.nan)
    forearm = np.where(ok[:, None], forearm, np.nan)
    wrist_ok = frame["frame_valid"] & valid[:, layout.BODY_WRIST[side]]
    wrist_body = np.where(wrist_ok[:, None], wrist_body, np.nan)
    return {
        "valid": ok,
        "elbow_bend": bend,
        "forearm_dir": forearm,
        "wrist_body": wrist_body,
        "wrist_valid": wrist_ok,
    }


def hand_evidence(points, valid, frame, side, cfg):
    start = layout.HAND_START[side]
    pts = points[:, start:start + layout.HAND_POINTS]
    val = valid[:, start:start + layout.HAND_POINTS]
    num_frames = pts.shape[0]

    mcp = pts[:, list(layout.MCP_OFFSETS)]
    mcp_ok = val[:, list(layout.MCP_OFFSETS)].all(-1)
    hand_ok = (
        frame["frame_valid"]
        & val[:, 0]
        & mcp_ok
        & (val.sum(-1) >= int(cfg["hands"]["min_valid_points"]))
    )

    palm_scale = np.median(
        np.linalg.norm(mcp - pts[:, :1], axis=-1), axis=-1
    )
    palm_scale = np.maximum(palm_scale, 1e-6)

    curls = np.full((num_frames, 5), np.nan)
    finger_dir = np.full((num_frames, 5, 2), np.nan)
    for fi, name in enumerate(layout.FINGER_NAMES):
        offs = layout.FINGER_OFFSETS[name]
        chain = [0, offs[0], offs[1], offs[2], offs[3]]
        chain_ok = val[:, chain].all(-1)
        total = np.zeros(num_frames)
        for j in (1, 2, 3):
            prev_pt = pts[:, chain[j - 1]]
            here = pts[:, chain[j]]
            nxt = pts[:, chain[j + 1]]
            total += np.pi - _angle_between(prev_pt - here, nxt - here)
        curls[:, fi] = np.where(chain_ok, total, np.nan)
        tip_ok = val[:, offs[0]] & val[:, offs[3]]
        d = _rotate_dir(_unit(pts[:, offs[3]] - pts[:, offs[0]]), frame)
        finger_dir[:, fi] = np.where(tip_ok[:, None], d, np.nan)

    # adjacent-finger spread over index/middle/ring/pinky MCP->tip directions
    dirs_img = np.full((num_frames, 4, 2), np.nan)
    for i, name in enumerate(layout.FINGER_NAMES[1:]):
        offs = layout.FINGER_OFFSETS[name]
        ok = val[:, offs[0]] & val[:, offs[3]]
        d = _unit(pts[:, offs[3]] - pts[:, offs[0]])
        dirs_img[:, i] = np.where(ok[:, None], d, np.nan)
    spread = np.nanmean(
        np.stack(
            [_angle_between(dirs_img[:, i], dirs_img[:, i + 1])
             for i in range(3)],
            axis=-1,
        ),
        axis=-1,
    )

    thumb_tip = pts[:, layout.FINGER_OFFSETS["thumb"][3]]
    tips = np.stack(
        [pts[:, layout.FINGER_OFFSETS[n][3]]
         for n in layout.FINGER_NAMES[1:]],
        axis=1,
    )
    opposition = (
        np.linalg.norm(tips - thumb_tip[:, None], axis=-1).min(-1)
        / palm_scale
    )
    aperture = (
        np.linalg.norm(
            pts[:, layout.FINGER_OFFSETS["index"][3]]
            - pts[:, layout.FINGER_OFFSETS["pinky"][3]],
            axis=-1,
        )
        / palm_scale
    )

    # palm-orientation proxy: signed cross of hand axis x knuckle line.
    hand_axis = pts[:, layout.FINGER_OFFSETS["middle"][0]] - pts[:, 0]
    knuckle = (
        pts[:, layout.FINGER_OFFSETS["pinky"][0]]
        - pts[:, layout.FINGER_OFFSETS["index"][0]]
    )
    cross = (
        hand_axis[:, 0] * knuckle[:, 1] - hand_axis[:, 1] * knuckle[:, 0]
    )
    cross = cross / np.maximum(
        np.linalg.norm(hand_axis, axis=-1)
        * np.linalg.norm(knuckle, axis=-1),
        1e-8,
    )
    # mirror so that for BOTH hands oriented < 0 <=> palm appears to face
    # the camera (heuristic mapping, see design doc 4.1)
    oriented = cross * (1.0 if side == "right" else -1.0)
    span_ratio = np.linalg.norm(knuckle, axis=-1) / palm_scale

    center_body = to_body(mcp.mean(1), frame)

    def mask1(a):
        return np.where(hand_ok, a, np.nan)

    return {
        "valid": hand_ok,
        "curls": np.where(hand_ok[:, None], curls, np.nan),
        "spread": mask1(spread),
        "opposition": mask1(opposition),
        "aperture": mask1(aperture),
        "palm_cross": mask1(oriented),
        "span_ratio": mask1(span_ratio),
        "center_body": np.where(hand_ok[:, None], center_body, np.nan),
        "finger_dir": np.where(
            hand_ok[:, None, None], finger_dir, np.nan
        ),
    }


def face_evidence(points, valid, frame):
    """Neutral-relative non-manual evidence from the 68 face landmarks."""
    re_pts = points[:, layout.FACE_RIGHT_EYE]
    le_pts = points[:, layout.FACE_LEFT_EYE]
    re_ok = valid[:, layout.FACE_RIGHT_EYE].sum(-1) >= 4
    le_ok = valid[:, layout.FACE_LEFT_EYE].sum(-1) >= 4
    eyes_ok = re_ok & le_ok & frame["frame_valid"]

    re_c = re_pts.mean(1)
    le_c = le_pts.mean(1)
    interocular = np.linalg.norm(le_c - re_c, axis=-1)
    eyes_ok &= interocular > 1e-6
    interocular = np.maximum(interocular, 1e-6)
    eye_y = 0.5 * (re_c[:, 1] + le_c[:, 1])

    brow_ok = eyes_ok & (valid[:, layout.FACE_BROWS].sum(-1) >= 6)
    brow_y = np.nanmean(
        np.where(
            valid[:, layout.FACE_BROWS],
            points[:, layout.FACE_BROWS, 1],
            np.nan,
        ),
        axis=-1,
    )
    # image y grows downward: brows above eyes -> positive height
    brow = np.where(brow_ok, (eye_y - brow_y) / interocular, np.nan)

    inner_ok = eyes_ok & (valid[:, layout.FACE_MOUTH_INNER].sum(-1) >= 6)
    inner_y = np.where(
        valid[:, layout.FACE_MOUTH_INNER],
        points[:, layout.FACE_MOUTH_INNER, 1],
        np.nan,
    )
    mouth_open = np.where(
        inner_ok,
        (np.nanmax(inner_y, -1) - np.nanmin(inner_y, -1)) / interocular,
        np.nan,
    )

    outer_ok = eyes_ok & (valid[:, layout.FACE_MOUTH_OUTER].sum(-1) >= 8)
    outer_x = np.where(
        valid[:, layout.FACE_MOUTH_OUTER],
        points[:, layout.FACE_MOUTH_OUTER, 0],
        np.nan,
    )
    mouth_width = np.where(
        outer_ok,
        (np.nanmax(outer_x, -1) - np.nanmin(outer_x, -1)) / interocular,
        np.nan,
    )

    # head roll proxy: eye line vs. the body lateral axis
    eye_line = _unit(le_c - re_c)
    lat = frame["lateral"]
    cross = eye_line[:, 0] * lat[:, 1] - eye_line[:, 1] * lat[:, 0]
    dot = (eye_line * lat).sum(-1)
    tilt = np.where(eyes_ok, np.arctan2(np.abs(cross), np.abs(dot)), np.nan)
    tilt_sign = np.where(cross >= 0, 1.0, -1.0)

    anchor_ok = eyes_ok & valid[:, layout.FACE_ANCHOR]
    nose_body = to_body(points[:, layout.FACE_ANCHOR], frame)
    nose_body = np.where(anchor_ok[:, None], nose_body, np.nan)

    def neutral(a):
        return np.nanmedian(a) if np.isfinite(a).any() else np.nan

    return {
        "valid": eyes_ok,
        "brow": brow,
        "brow_neutral": neutral(brow),
        "mouth_open": mouth_open,
        "mouth_open_neutral": neutral(mouth_open),
        "mouth_width": mouth_width,
        "tilt": tilt,
        "tilt_sign": tilt_sign,
        "nose_body": nose_body,
    }


def compute_evidence(keypoints, scores, cfg):
    """Full evidence dict for one clip.

    keypoints: [T,133,2] float, scores: [T,133] float.
    """
    conf_min = float(cfg["confidence_min"])
    valid = scores > conf_min
    points, valid = interpolate_gaps(
        keypoints.astype(np.float64), valid, int(cfg["max_interp_frames"])
    )

    frame = body_frame(points, valid)
    evidence = {
        "num_frames": keypoints.shape[0],
        "frame": frame,
        "face": face_evidence(points, valid, frame),
        "arms": {},
        "hands": {},
    }
    for side in layout.SIDES:
        evidence["arms"][side] = arm_evidence(points, valid, frame, side)
        evidence["hands"][side] = hand_evidence(
            points, valid, frame, side, cfg
        )
    return evidence
