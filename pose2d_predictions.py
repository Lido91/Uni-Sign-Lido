"""Save generated poses in the format read by back_translation_eval --mode pred_dir."""

import os
import pickle
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def save_pose_prediction_batch(
    output_dir, pred_parts, ref_parts, pred_lengths, ref_lengths,
    names, texts, mean, std, part_order, part_joints,
    back_translations=None, metadata=None,
):
    """Save true-length, unclipped xy and GT confidence; retain exact sample IDs.

    feats_rst/feats_ref are z-scored with the *motion* VAE statistics, not
    the 512D latent statistics. The embedded statistics make the pose
    representation explicit. Each pickle can be consumed by pred_dir.
    """
    batch = len(names)
    if not all(len(values) == batch for values in
               (texts, pred_lengths, ref_lengths)):
        raise ValueError("prediction batch metadata lengths do not match")
    if back_translations is not None and len(back_translations) != batch:
        raise ValueError("back-translation count does not match the pose batch")
    names = [str(name) for name in names]
    if any(not name for name in names) or len(set(names)) != batch:
        raise ValueError("sample names must be nonempty and unique")

    mean, std = _numpy(mean).reshape(-1), _numpy(std).reshape(-1)
    width = sum(part_joints[part] for part in part_order) * 2
    if mean.shape != (width,) or std.shape != (width,):
        raise ValueError("motion mean/std do not match the pose feature width")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()
            and (std > 0).all()):
        raise ValueError("motion normalization statistics must be finite; std > 0")

    pred_xy, ref_xy, ref_conf = [], [], []
    for part in part_order:
        pred, ref = _numpy(pred_parts[part]), _numpy(ref_parts[part])
        for value in (pred, ref):
            if (value.ndim != 4 or value.shape[0] != batch
                    or value.shape[2] != part_joints[part]
                    or value.shape[3] < 3):
                raise ValueError(f"invalid [B,T,J,3] pose shape for {part}")
        pred_xy.append(pred[..., :2].reshape(batch, pred.shape[1], -1))
        ref_xy.append(ref[..., :2].reshape(batch, ref.shape[1], -1))
        ref_conf.append(ref[..., 2])
    pred_xy = np.concatenate(pred_xy, axis=-1)
    ref_xy = np.concatenate(ref_xy, axis=-1)
    ref_conf = np.concatenate(ref_conf, axis=-1)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, name in enumerate(names):
        pred_len, ref_len = int(pred_lengths[index]), int(ref_lengths[index])
        if not (0 < pred_len <= pred_xy.shape[1]
                and 0 < ref_len <= ref_xy.shape[1]):
            raise ValueError(f"invalid true lengths for sample {name!r}")
        pred = pred_xy[index, :pred_len]
        if not np.isfinite(pred).all():
            raise ValueError(f"nonfinite generated coordinates for {name!r}")
        # Encode path separators without discarding the original ID or
        # colliding with another basename; name is also retained in the payload.
        path = directory / (quote(name, safe="-_.:") + ".pkl")
        payload = {
            "name": name,
            "text": str(texts[index]),
            "feats_rst": ((pred - mean) / std).astype(np.float32),
            "feats_ref": ((ref_xy[index, :ref_len] - mean) / std).astype(np.float32),
            "conf_ref": ref_conf[index, :ref_len].copy(),
            "num_frames_rst": pred_len,
            "num_frames_ref": ref_len,
            "motion_mean": mean.copy(),
            "motion_std": std.copy(),
            "pose_part_order": list(part_order),
            "pose_part_joints": {part: int(part_joints[part]) for part in part_order},
            "representation": "VAE-motion-zscored xy; body, lhand, rhand, face",
            "metadata": dict(metadata or {}),
        }
        if back_translations is not None:
            payload["back_translation"] = str(back_translations[index])
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, prefix=".pose-", suffix=".tmp", delete=False
            ) as handle:
                temp_path = handle.name
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)
        saved_paths.append(str(path))
    return saved_paths
