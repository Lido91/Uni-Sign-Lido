from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets_vae import How2SignVaeDataset
from models_vae import build_latent_attention_mask
from vae_encoder import (
    DOWNSAMPLE_FACTOR,
    FEATURE_DIM,
    LATENT_DIM,
    MOMENT_DIM,
    TrainableKlVaeEncoder,
    posterior_mean,
)


class VaeEncoderPathTest(unittest.TestCase):
    def test_encoder_shape_and_translation_gradient(self) -> None:
        encoder = TrainableKlVaeEncoder()
        motion = torch.randn(1, 16, FEATURE_DIM)
        moments = encoder(motion)
        mean = posterior_mean(moments)

        self.assertEqual(tuple(moments.shape), (1, 4, MOMENT_DIM))
        self.assertEqual(tuple(mean.shape), (1, 4, LATENT_DIM))

        mean.square().mean().backward()
        first_conv = encoder.encoder.model[0]
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertTrue(torch.isfinite(first_conv.weight.grad).all())
        self.assertIsNotNone(encoder.quant_conv.weight.grad)
        self.assertTrue(torch.isfinite(encoder.quant_conv.weight.grad).all())

    def test_dataset_contract_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "new_joint_vecs").mkdir()
            (root / "texts").mkdir()
            np.save(root / "mean.npy", np.zeros(FEATURE_DIM, dtype=np.float32))
            np.save(root / "std.npy", np.ones(FEATURE_DIM, dtype=np.float32))

            for split, length in (("train", 10), ("val", 12), ("test", 14)):
                sample_id = f"{split}_sample"
                (root / f"{split}.txt").write_text(
                    f"{sample_id}\n", encoding="utf-8"
                )
                np.save(
                    root / "new_joint_vecs" / f"{sample_id}.npy",
                    np.ones((length, FEATURE_DIM), dtype=np.float32),
                )
                (root / "texts" / f"{sample_id}.txt").write_text(
                    "reference sentence#reference/OTHER sentence/OTHER#0.0#0.0\n",
                    encoding="utf-8",
                )

            datasets = [
                How2SignVaeDataset(
                    root,
                    split,
                    root / "mean.npy",
                    root / "std.npy",
                    max_motion_length=400,
                )
                for split in ("train", "val", "test")
            ]
            samples = [dataset[0] for dataset in datasets]
            self.assertEqual(tuple(samples[0][1].shape), (8, FEATURE_DIM))
            self.assertEqual(tuple(samples[1][1].shape), (12, FEATURE_DIM))
            self.assertEqual(tuple(samples[2][1].shape), (12, FEATURE_DIM))
            self.assertEqual(samples[0][2], "reference sentence")

            src_input, tgt_input = datasets[0].collate_fn(samples[:2])
            self.assertEqual(tuple(src_input["motion"].shape), (2, 12, FEATURE_DIM))
            self.assertEqual(src_input["src_length_batch"].tolist(), [8, 12])
            self.assertEqual(len(tgt_input["gt_sentence"]), 2)

    def test_latent_mask_uses_valid_lengths(self) -> None:
        frame_lengths = torch.tensor([8, 12], dtype=torch.long)
        mask = build_latent_attention_mask(
            frame_lengths,
            latent_steps=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(mask.shape), (2, 3))
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 3])
        self.assertEqual(DOWNSAMPLE_FACTOR, 4)


if __name__ == "__main__":
    unittest.main()
