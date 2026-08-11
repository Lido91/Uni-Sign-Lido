"""Additive 133D How2Sign dataset path for the trainable VAE encoder.

The original ``datasets.py`` remains untouched.  This loader consumes the
canonical SOKE/MotionGPT3 processed layout:

    root/
      new_joint_vecs/{id}.npy  # [T,133], unnormalized
      texts/{id}.txt
      train.txt
      val.txt
      test.txt

Normalization uses the mean/std persisted by the trained VAE, not an arbitrary
downstream copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from vae_encoder import DOWNSAMPLE_FACTOR, FEATURE_DIM


def load_reference(text_path: Path) -> str:
    if not text_path.is_file():
        raise FileNotFoundError(f"How2Sign reference not found: {text_path}")
    for line in text_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            caption = line.split("#", 1)[0].strip()
            if caption:
                return caption
    raise ValueError(f"How2Sign reference is empty: {text_path}")


class How2SignVaeDataset(Dataset):
    """Pair normalized online 133D motion with an English reference."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        mean_path: Union[str, Path],
        std_path: Union[str, Path],
        max_motion_length: int = 400,
        unit_length: int = DOWNSAMPLE_FACTOR,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if unit_length != DOWNSAMPLE_FACTOR:
            raise ValueError(
                f"this VAE downsamples by {DOWNSAMPLE_FACTOR}; "
                f"unit_length={unit_length} is incompatible"
            )
        if max_motion_length < unit_length:
            raise ValueError("max_motion_length must contain at least one VAE unit")

        self.root = Path(root).expanduser().resolve()
        self.motion_dir = self.root / "new_joint_vecs"
        self.text_dir = self.root / "texts"
        self.split = split
        self.unit_length = unit_length
        self.max_motion_length = (
            int(max_motion_length) // unit_length
        ) * unit_length

        for directory in (self.motion_dir, self.text_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"How2Sign directory not found: {directory}")
        split_path = self.root / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"How2Sign split not found: {split_path}")
        self.sample_ids = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.sample_ids:
            raise ValueError(f"How2Sign {split} split is empty: {split_path}")

        self.mean = self._load_stat(mean_path, "mean")
        self.std = self._load_stat(std_path, "std")
        if np.any(self.std == 0):
            indices = np.flatnonzero(self.std == 0).tolist()
            raise ValueError(f"VAE std contains zero entries at indices {indices[:10]}")

    @staticmethod
    def _load_stat(path: Union[str, Path], name: str) -> np.ndarray:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"VAE {name} file not found: {resolved}")
        value = np.asarray(np.load(resolved), dtype=np.float32).reshape(-1)
        if value.shape != (FEATURE_DIM,):
            raise ValueError(
                f"VAE {name} must have shape ({FEATURE_DIM},), got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"VAE {name} contains non-finite values: {resolved}")
        return value

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, str]:
        sample_id = self.sample_ids[index]
        motion_path = self.motion_dir / f"{sample_id}.npy"
        if not motion_path.is_file():
            raise FileNotFoundError(f"How2Sign motion not found: {motion_path}")
        motion = np.asarray(np.load(motion_path), dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"How2Sign motion must have shape [T,{FEATURE_DIM}], "
                f"got {motion.shape}: {sample_id}"
            )
        if not np.isfinite(motion).all():
            raise ValueError(f"How2Sign motion contains non-finite values: {sample_id}")

        usable_length = min(len(motion), self.max_motion_length)
        usable_length = (usable_length // self.unit_length) * self.unit_length
        if usable_length < self.unit_length:
            raise ValueError(
                f"How2Sign motion is shorter than one VAE unit: "
                f"{sample_id} has {len(motion)} frames"
            )
        motion = motion[:usable_length]
        motion = (motion - self.mean) / self.std
        caption = load_reference(self.text_dir / f"{sample_id}.txt")
        return sample_id, torch.from_numpy(np.ascontiguousarray(motion)), caption

    @staticmethod
    def collate_fn(
        batch: list[tuple[str, torch.Tensor, str]],
    ) -> tuple[dict[str, object], dict[str, list[str]]]:
        names, motions, captions = zip(*batch)
        lengths = torch.tensor([motion.shape[0] for motion in motions], dtype=torch.long)
        padded_motion = pad_sequence(motions, batch_first=True, padding_value=0.0)
        src_input: dict[str, object] = {
            "motion": padded_motion,
            "src_length_batch": lengths,
            "name_batch": list(names),
        }
        tgt_input = {"gt_sentence": list(captions)}
        return src_input, tgt_input

    def __str__(self) -> str:
        return f"#total {len(self)} ({self.split}, 133D motion -> trainable KL-VAE)"
