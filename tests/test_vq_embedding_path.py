from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets_vq_embeddings import (
    ThreePartVQEmbeddingDataset,
    VQ_CODEBOOK_SIZES,
    VQ_EMBEDDING_DIM,
    VQ_NUM_PARTS,
)
from models_vq_embeddings import ThreePartVQAdapter


class VQEmbeddingPathTest(unittest.TestCase):
    @staticmethod
    def _write_annotation(
        root: Path,
        source: str,
        split: str,
        rows: list[tuple[str, str]],
    ) -> None:
        directory = root / split / "re_aligned"
        directory.mkdir(parents=True, exist_ok=True)
        if source == "how2sign":
            filename = f"how2sign_realigned_{split}_preprocessed_fps.csv"
        else:
            filename = f"youtube_asl_{split}.csv"
        with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["SENTENCE_NAME", "SENTENCE"],
            )
            writer.writeheader()
            for name, caption in rows:
                writer.writerow({"SENTENCE_NAME": name, "SENTENCE": caption})

    @staticmethod
    def _write_manifest(path: Path, checkpoint: Path) -> None:
        stat = checkpoint.stat()
        payload = {
            "version": 1,
            "artifact": "discrete_motion_codes",
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
            "splits": ["train", "val", "test"],
            "metadata": {
                "motion_codebook_size": VQ_CODEBOOK_SIZES[0],
                "lhand_codebook_size": VQ_CODEBOOK_SIZES[1],
                "rhand_codebook_size": VQ_CODEBOOK_SIZES[2],
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_dataset_contract_provenance_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding_root = root / "embeddings"
            annotation_roots = {
                "how2sign": root / "how2sign",
                "youtube3d": root / "youtube3d",
            }
            samples = {
                "how2sign": ("h2s_sample", 4, "How2Sign caption"),
                "youtube3d": ("yt_sample", 6, "YouTube3D caption"),
            }
            for source, (name, length, caption) in samples.items():
                (embedding_root / source).mkdir(parents=True)
                np.save(
                    embedding_root / source / f"{name}.npy",
                    np.ones(
                        (1, length, VQ_NUM_PARTS, VQ_EMBEDDING_DIM),
                        dtype=np.float32,
                    ),
                )
                self._write_annotation(
                    annotation_roots[source], source, "train", [(name, caption)]
                )

            val_name = "yt_val_sample"
            np.save(
                embedding_root / "youtube3d" / f"{val_name}.npy",
                np.ones((1, 5, 3, 512), dtype=np.float32),
            )
            self._write_annotation(
                annotation_roots["youtube3d"],
                "youtube3d",
                "val",
                [(val_name, "YouTube3D validation caption")],
            )

            split_index = embedding_root / "split_index.json"
            split_index.write_text(
                json.dumps(
                    {
                        "train": ["h2s_sample", "yt_sample"],
                        "val": [val_name],
                        "test": ["h2s_test"],
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "vq.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = root / "_manifest.json"
            self._write_manifest(manifest, checkpoint)

            dataset = ThreePartVQEmbeddingDataset(
                embedding_root=embedding_root,
                annotation_roots=annotation_roots,
                sources=("how2sign", "youtube3d"),
                split="train",
                split_index_path=split_index,
                token_manifest_path=manifest,
                vq_checkpoint=checkpoint,
            )
            self.assertEqual(len(dataset), 2)
            first, second = dataset[0], dataset[1]
            self.assertEqual(tuple(first[1].shape), (4, 3, 512))
            self.assertEqual(first[1].dtype, torch.float32)

            src_input, tgt_input = dataset.collate_fn([first, second])
            self.assertEqual(tuple(src_input["vq_embeddings"].shape), (2, 6, 3, 512))
            self.assertEqual(tuple(src_input["attention_mask"].shape), (2, 6))
            self.assertEqual(src_input["attention_mask"].sum(dim=1).tolist(), [4, 6])
            self.assertEqual(src_input["src_length_batch"].tolist(), [4, 6])
            self.assertEqual(len(tgt_input["gt_sentence"]), 2)

            combined = ThreePartVQEmbeddingDataset(
                embedding_root=embedding_root,
                annotation_roots=annotation_roots,
                sources=("youtube3d",),
                split=("train", "val"),
                split_index_path=split_index,
                token_manifest_path=manifest,
                vq_checkpoint=checkpoint,
            )
            self.assertEqual(len(combined), 2)
            self.assertIn("train+val", str(combined))

    def test_three_part_adapter_shape_and_gradients(self) -> None:
        adapter = ThreePartVQAdapter(
            output_dim=768,
            part_hidden_dim=32,
            dropout=0.0,
        )
        embeddings = torch.randn(2, 5, 3, 512)
        output = adapter(embeddings)
        self.assertEqual(tuple(output.shape), (2, 5, 768))
        output.square().mean().backward()

        for name in ("body", "left_hand", "right_hand"):
            gradient = adapter.part_projections[name].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
        self.assertIsNotNone(adapter.fusion_projection.weight.grad)

    def test_rejects_wrong_embedding_geometry(self) -> None:
        adapter = ThreePartVQAdapter(output_dim=768)
        with self.assertRaisesRegex(ValueError, "3,512"):
            adapter(torch.randn(1, 4, 2, 512))


if __name__ == "__main__":
    unittest.main()
