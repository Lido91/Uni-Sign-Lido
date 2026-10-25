"""Dataset for the baseline + SignSMD text fusion path.

Additive subclass of the ORIGINAL ``datasets.S2T_Dataset`` (the ST-GCN
baseline loader): part split, normalization, padding, and masks are all
inherited unchanged.  Two additions:

1. either one per-sample structured description for early/dual fusion, or
   four standalone part texts plus structured event spans for paired fusion;
   Python text/event objects pass through collate unchanged because
   ``fine_tuning.move_src_input_to_device`` ignores non-tensor entries;
2. deterministic frame sampling (``--frame_sampling linear``, the default)
   so the pose stream and the description cover exactly the same time.

Why 2: current SignSMD texts are compiled on the trainer-aligned pose steps.
The baseline loader randomly subsamples clips longer than ``--max_length``
(a different subset every epoch), so explicit ``[step a-b]`` spans would no
longer index the model input. Linear-index resampling mirrors the compiler
exactly, and evaluation of long clips becomes deterministic. Clips within
``--max_length`` are
kept whole, identical to the baseline.  ``--frame_sampling unisign_random``
restores the baseline behavior for ablations.

Text file contracts (see docs/POSE_SIGNSMD_FUSION.md):

    # early / dual
    <signsmd_text_root>/<clip-name>.txt      # clip-name = video_path minus .mp4

    # paired (contract signsmd2d_parts_v0_2)
    <signsmd_text_root>/<part>/<clip-name>.txt
    <signsmd_event_root>/<clip-name>.json

Missing-file policy is explicit: ``--signsmd_on_missing error`` (default)
fails fast so a partially generated text corpus cannot silently train as a
mixed with/without-text run; ``empty`` substitutes "" (useful for smoke
tests before the compiler has covered every split).
"""

import json
import os
import pickle

import numpy as np

from datasets import S2T_Dataset, load_part_kp
from signsmd2d import PARTS_VERSION
from signsmd2d.realization_parts import PARTS

SIGNSMD_MISSING_POLICIES = ("error", "empty")
PAIRED_EVENT_KINDS = ("hold", "movement", "role", "transition")

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

        self.paired_fusion = getattr(args, "fusion", "early") == "paired"
        self.events_required = bool(getattr(args, "events_required", False))
        self.signsmd_event_root = getattr(args, "signsmd_event_root", "")
        if self.paired_fusion:
            if self.frame_sampling != "linear":
                raise ValueError(
                    "paired SignSMD event spans require --frame_sampling linear"
                )
            if not self.signsmd_event_root:
                text_root = os.path.normpath(self.signsmd_text_root)
                self.signsmd_event_root = os.path.join(
                    os.path.dirname(text_root), "events"
                )
            if self.events_required and not os.path.isdir(
                self.signsmd_event_root
            ):
                raise FileNotFoundError(
                    "--signsmd_event_root does not exist or is not a "
                    f"directory: {self.signsmd_event_root}"
                )
            if self.signsmd_on_missing == "error":
                missing_part_dirs = [
                    part for part in PARTS
                    if not os.path.isdir(
                        os.path.join(self.signsmd_text_root, part)
                    )
                ]
                if missing_part_dirs:
                    raise FileNotFoundError(
                        "paired SignSMD part text directories are missing "
                        f"under {self.signsmd_text_root}: {missing_part_dirs}"
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
        # onto the model timeline, so the compiled text's step spans land
        # exactly on the pose positions the model sees.  The SignSMD
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

    @staticmethod
    def _clip_name(video_path):
        return video_path[:-4] if video_path.endswith(".mp4") else video_path

    def _load_part_texts(self, video_path):
        clip = self._clip_name(video_path)
        texts = {}
        for part in PARTS:
            text_path = os.path.join(
                self.signsmd_text_root, part, clip + ".txt"
            )
            if not os.path.isfile(text_path):
                if self.signsmd_on_missing == "empty":
                    texts[part] = ""
                    continue
                raise FileNotFoundError(
                    f"SignSMD {part} text missing for sample {video_path!r}: "
                    f"{text_path}"
                )
            with open(text_path, encoding="utf-8") as f:
                texts[part] = f.read().strip()
        return texts

    def _load_signsmd_events(self, video_path, num_steps, part_texts):
        clip = self._clip_name(video_path)
        event_path = os.path.join(self.signsmd_event_root, clip + ".json")
        if not os.path.isfile(event_path):
            if self.events_required:
                raise FileNotFoundError(
                    f"SignSMD events missing for sample {video_path!r}: "
                    f"{event_path}"
                )
            return []

        with open(event_path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("contract") != PARTS_VERSION:
            raise ValueError(
                f"SignSMD event contract mismatch for {video_path!r}: "
                f"{payload.get('contract')!r}, expected {PARTS_VERSION!r}"
            )
        if payload.get("clip") != clip:
            raise ValueError(
                f"SignSMD event clip mismatch in {event_path}: "
                f"{payload.get('clip')!r}, expected {clip!r}"
            )
        event_steps = payload.get("num_steps")
        if event_steps != num_steps:
            raise ValueError(
                f"SignSMD/pose step mismatch for {video_path!r}: events have "
                f"num_steps={event_steps}, pose loader produced {num_steps}; "
                "regenerate with the trainer's --max_length"
            )

        events = payload.get("events")
        if not isinstance(events, list):
            raise ValueError(f"events must be a list in {event_path}")
        validated = []
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(
                    f"event {index} must be an object in {event_path}"
                )
            part = event.get("part")
            kind = event.get("kind")
            phrase = event.get("phrase")
            start = event.get("start")
            end = event.get("end")
            if part not in PARTS:
                raise ValueError(
                    f"event {index} has invalid part {part!r} in {event_path}"
                )
            if kind not in PAIRED_EVENT_KINDS:
                raise ValueError(
                    f"event {index} has invalid kind {kind!r} in {event_path}"
                )
            if not isinstance(phrase, str) or not phrase.strip():
                raise ValueError(
                    f"event {index} has an empty phrase in {event_path}"
                )
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start <= end < num_steps
            ):
                raise ValueError(
                    f"event {index} span [{start}, {end}] is outside "
                    f"[0, {num_steps}) in {event_path}"
                )
            # The per-part text is an auditable rendering of the same event
            # records. Check both phrase and span so stale TXT/JSON pairs fail
            # before training instead of silently diverging.
            part_text = part_texts.get(part, "")
            rendered = f"{phrase} [step {start}-{end}]"
            if part_text and rendered not in part_text:
                raise ValueError(
                    f"event {index} is not rendered in the {part} text for "
                    f"{video_path!r}; regenerate the complete parts contract"
                )
            validated.append({
                "part": part,
                "kind": kind,
                "phrase": phrase.strip(),
                "start": start,
                "end": end,
            })
        return validated

    def __getitem__(self, index):
        name_sample, pose_sample, text, gloss, support_rgb_dict = (
            super().__getitem__(index)
        )
        key = self.list[index]
        video_path = self.raw_data[key]["video_path"]
        if self.paired_fusion:
            part_texts = self._load_part_texts(video_path)
            num_steps = len(pose_sample[PARTS[0]])
            if any(len(pose_sample[part]) != num_steps for part in PARTS):
                raise ValueError(
                    f"pose parts have inconsistent lengths for {video_path!r}"
                )
            events = self._load_signsmd_events(
                video_path, num_steps, part_texts
            )
            return (
                name_sample, pose_sample, text, gloss, support_rgb_dict,
                part_texts, events,
            )

        description = self._load_signsmd_text(video_path)
        return name_sample, pose_sample, text, gloss, support_rgb_dict, description

    def collate_fn(self, batch):
        if self.paired_fusion:
            part_texts = {
                part: [item[-2][part] for item in batch] for part in PARTS
            }
            events = [item[-1] for item in batch]
            src_input, tgt_input = super().collate_fn(
                [item[:-2] for item in batch]
            )
            src_input["signsmd_part_texts"] = part_texts
            src_input["signsmd_events"] = events
            return src_input, tgt_input

        descriptions = [item[-1] for item in batch]
        src_input, tgt_input = super().collate_fn(
            [item[:-1] for item in batch]
        )
        src_input["signsmd_text"] = descriptions
        return src_input, tgt_input
