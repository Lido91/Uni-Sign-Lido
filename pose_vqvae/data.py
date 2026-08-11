"""Deterministic-eval variant of Uni-Sign's S2T_Dataset.

The base ``S2T_Dataset.load_pose`` uses ``random.sample`` whenever a clip
exceeds ``max_length`` frames, regardless of phase — so successive dev/test
evaluations (and token extraction) would see different frames each time,
making metrics and checkpoint ranking stochastic. This subclass keeps the
exact same part decomposition and normalization but subsamples long clips
with evenly spaced (deterministic) indices instead.

Use ``S2T_Dataset`` for training (random subsampling doubles as temporal
augmentation) and this class for validation, testing, and token extraction.
"""

import os
import pickle

import numpy as np

from datasets import S2T_Dataset, load_part_kp


class DeterministicS2T_Dataset(S2T_Dataset):

    def load_pose(self, path):
        pose = pickle.load(
            open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))

        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0

        if duration > self.max_length:
            # evenly spaced instead of random.sample: unique because the
            # spacing (duration-1)/(max_length-1) is > 1 here
            tmp = np.linspace(0, duration - 1, num=self.max_length).astype(np.int64)
        else:
            tmp = np.arange(duration, dtype=np.int64)

        tmp = tmp + start

        skeletons = [pose['keypoints'][i] for i in tmp]
        confs = [pose['scores'][i] for i in tmp]

        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        if self.rgb_support:
            raise NotImplementedError(
                "DeterministicS2T_Dataset is pose-only; rgb_support unsupported")

        return kps_with_scores, {}
