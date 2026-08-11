"""Dataset for the pose2d KL-VAE path: temporal sampling matched to the VAE.

The original ``datasets.S2T_Dataset`` randomly subsamples long clips to
``--max_length`` frames (non-uniform, changes local speed and temporal
continuity every epoch).  The KL-VAE corpus was instead built by
deterministic linear-index resampling of whole clips into [40, 400] frames
(MoMADiff ``prepare/prepare_how2sign_pose2d.py``), and the VAE trained on
contiguous 40-frame windows of that — exactly the temporal statistics a
temporal-conv encoder is sensitive to.

``S2T_Dataset_Pose2D_KLVAE`` therefore defaults to the VAE rule
(``--frame_sampling vae_linear``): linear-index resample every clip into
``[VAE_MIN_LEN, --max_length]``; clips already inside the range are kept
whole.  This is also deterministic, so evaluation no longer varies across
runs for long clips.  ``--frame_sampling unisign_random`` restores the
original behavior for ablations.

Everything else (part split, normalization, collate) is inherited unchanged.
"""

import os
import pickle

import numpy as np

from datasets import S2T_Dataset, load_part_kp

# Matches min_len of the KL-VAE corpus preparation (clips shorter than this
# were linearly upsampled before VAE training).
VAE_MIN_LEN = 40

FRAME_SAMPLING_MODES = ("vae_linear", "unisign_random")


class S2T_Dataset_Pose2D_KLVAE(S2T_Dataset):
    def __init__(self, path, args, phase):
        super().__init__(path, args, phase)
        if self.rgb_support:
            raise ValueError(
                "--rgb_support belongs to the original ST-GCN path"
            )
        self.frame_sampling = getattr(args, "frame_sampling", "vae_linear")
        if self.frame_sampling not in FRAME_SAMPLING_MODES:
            raise ValueError(
                f"frame_sampling must be one of {FRAME_SAMPLING_MODES}, "
                f"got {self.frame_sampling!r}"
            )

    def load_pose(self, path):
        if self.frame_sampling == "unisign_random":
            return super().load_pose(path)

        pose = pickle.load(
            open(os.path.join(self.pose_dir, path.replace(".mp4", ".pkl")), "rb")
        )

        if "start" in pose.keys():
            assert pose["start"] < pose["end"]
            duration = pose["end"] - pose["start"]
            start = pose["start"]
        else:
            duration = len(pose["scores"])
            start = 0

        # VAE-corpus rule: deterministic linear-index resampling into
        # [VAE_MIN_LEN, max_length]; in-range clips are kept whole.
        target = int(np.clip(duration, VAE_MIN_LEN, self.max_length))
        if target == duration:
            tmp = np.arange(duration)
        else:
            tmp = np.round(
                np.linspace(0, duration - 1, target)
            ).astype(np.int64)
        tmp = tmp + start

        skeletons = pose["keypoints"]
        confs = pose["scores"]
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        kps_with_scores = load_part_kp(skeletons_tmp, confs_tmp, force_ok=True)
        return kps_with_scores, {}
