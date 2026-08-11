"""Dataset for the baseline + SignSMD text fusion path.

Additive subclass of the ORIGINAL ``datasets.S2T_Dataset`` (the ST-GCN
baseline loader): part split, normalization, padding, and masks are all
inherited unchanged.  Two additions:

1. a per-sample structured sign description (SignSMD-2D text, compiled
   offline from the same RTMPose keypoints) loaded from
   ``--signsmd_text_root`` and carried through collate as
   ``src_input["signsmd_text"]`` (a list[str] —
   ``fine_tuning.move_src_input_to_device`` ignores non-tensor entries);
2. deterministic frame sampling (``--frame_sampling linear``, the default)
   so the pose stream and the description cover exactly the same time.

Why 2: SignSMD texts are compiled over ALL frames of a clip and express
interval spans as percentages of it.  The baseline loader randomly
subsamples clips longer than ``--max_length`` (a different subset every
epoch), so a percentage span only approximately matches the model's pose
timeline.  Linear-index resampling is an affine map of the clip onto the
model timeline: fraction p of the text corresponds exactly (up to frame
rounding) to fraction p of the pose sequence, every epoch, and evaluation
of long clips becomes deterministic.  Clips within ``--max_length`` are
kept whole, identical to the baseline.  ``--frame_sampling unisign_random``
restores the baseline behavior for ablations.

Text file contract (see docs/POSE_SIGNSMD_FUSION.md):

    <signsmd_text_root>/<clip-name>.txt      # clip-name = video_path minus .mp4

Missing-file policy is explicit: ``--signsmd_on_missing error`` (default)
fails fast so a partially generated text corpus cannot silently train as a
mixed with/without-text run; ``empty`` substitutes "" (useful for smoke
tests before the compiler has covered every split).
"""

import os
import pickle

import numpy as np

from datasets import S2T_Dataset, load_part_kp

SIGNSMD_MISSING_POLICIES = ("error", "empty")

# Same rule family as datasets_pose2d_klvae (minus its VAE minimum-length
# upsampling): "linear" = deterministic linear-index subsampling of clips
# longer than --max_length, shorter clips kept whole.
FRAME_SAMPLING_MODES = ("linear", "unisign_random")


class S2T_Dataset_SignSMD(S2T_Dataset):
    def __init__(self, path, args, phase):
        super().__init__(path, args, phase)
        if self.rgb_support:
            raise ValueError(
                "--rgb_support is not supported on the SignSMD fusion path yet"
            )

        self.signsmd_text_root = args.signsmd_text_root
        if not os.path.isdir(self.signsmd_text_root):
            raise FileNotFoundError(
                f"--signsmd_text_root does not exist or is not a directory: "
                f"{self.signsmd_text_root}"
            )

        self.signsmd_on_missing = getattr(args, "signsmd_on_missing", "error")
        if self.signsmd_on_missing not in SIGNSMD_MISSING_POLICIES:
            raise ValueError(
                f"signsmd_on_missing must be one of {SIGNSMD_MISSING_POLICIES}, "
                f"got {self.signsmd_on_missing!r}"
            )

        self.frame_sampling = getattr(args, "frame_sampling", "linear")
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

        # Deterministic linear-index subsampling: an affine map of the clip
        # onto the model timeline, so the compiled text's percentage spans
        # land exactly on the pose positions the model sees.  The SignSMD
        # compiler slices the same [start, end) window (compiler.compile_clip),
        # so text and pose share one time origin.
        if duration > self.max_length:
            tmp = np.round(
                np.linspace(0, duration - 1, self.max_length)
            ).astype(np.int64)
        else:
            tmp = np.arange(duration)
        tmp = tmp + start

        skeletons = pose["keypoints"]
        confs = pose["scores"]
        skeletons_tmp = [skeletons[index] for index in tmp]
        confs_tmp = [confs[index] for index in tmp]

        kps_with_scores = load_part_kp(skeletons_tmp, confs_tmp, force_ok=True)
        return kps_with_scores, {}

    def _load_signsmd_text(self, video_path):
        text_path = os.path.join(
            self.signsmd_text_root, video_path.replace(".mp4", ".txt")
        )
        if not os.path.isfile(text_path):
            if self.signsmd_on_missing == "empty":
                return ""
            raise FileNotFoundError(
                f"SignSMD text missing for sample {video_path!r}: {text_path} "
                f"(generate it first, or pass --signsmd_on_missing empty)"
            )
        with open(text_path, encoding="utf-8") as f:
            return f.read().strip()

    def __getitem__(self, index):
        name_sample, pose_sample, text, gloss, support_rgb_dict = (
            super().__getitem__(index)
        )
        key = self.list[index]
        description = self._load_signsmd_text(self.raw_data[key]["video_path"])
        return name_sample, pose_sample, text, gloss, support_rgb_dict, description

    def collate_fn(self, batch):
        descriptions = [item[-1] for item in batch]
        src_input, tgt_input = super().collate_fn(
            [item[:-1] for item in batch]
        )
        src_input["signsmd_text"] = descriptions
        return src_input, tgt_input
