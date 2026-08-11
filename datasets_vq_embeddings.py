"""Dataset for cached SOKE three-part VQ embeddings.

This is an additive Uni-Sign input path.  Each sample is a cached, quantized
codebook lookup with shape ``[1, T', 3, 512]`` in the fixed part order
``body/rest, left hand, right hand``.  The original pose and online-VAE
datasets remain unchanged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


VQ_EMBEDDING_DIM = 512
VQ_NUM_PARTS = 3
VQ_PART_NAMES = ("body", "left_hand", "right_hand")
VQ_CODEBOOK_SIZES = (96, 192, 192)
SUPPORTED_SOURCES = ("how2sign", "youtube3d")


def _checkpoint_fingerprint(path: Union[str, Path]) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def validate_vq_manifest(
    manifest_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    required_splits: Sequence[str],
) -> dict[str, object]:
    """Validate that cached embeddings came from the configured VQ-VAE."""
    resolved = Path(manifest_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SOKE VQ token manifest not found: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("artifact") != "discrete_motion_codes":
        raise ValueError(
            f"unexpected VQ manifest artifact in {resolved}: "
            f"{manifest.get('artifact')!r}"
        )
    expected_checkpoint = _checkpoint_fingerprint(checkpoint_path)
    if manifest.get("checkpoint") != expected_checkpoint:
        raise ValueError(
            "cached VQ artifacts do not match the configured VQ-VAE checkpoint: "
            f"manifest={manifest.get('checkpoint')}, expected={expected_checkpoint}"
        )

    manifest_splits = set(manifest.get("splits", []))
    missing_splits = set(required_splits) - manifest_splits
    if missing_splits:
        raise ValueError(
            f"VQ manifest does not cover splits {sorted(missing_splits)}: {resolved}"
        )

    metadata = manifest.get("metadata", {})
    recorded_sizes = (
        metadata.get("motion_codebook_size"),
        metadata.get("lhand_codebook_size"),
        metadata.get("rhand_codebook_size"),
    )
    if recorded_sizes != VQ_CODEBOOK_SIZES:
        raise ValueError(
            f"expected body/left/right codebooks {VQ_CODEBOOK_SIZES}, "
            f"got {recorded_sizes} in {resolved}"
        )
    return manifest


def _annotation_path(root: Path, source: str, split: str) -> Path:
    if source == "how2sign":
        filename = f"how2sign_realigned_{split}_preprocessed_fps.csv"
    elif source == "youtube3d":
        filename = f"youtube_asl_{split}.csv"
    else:
        raise ValueError(f"unsupported VQ source: {source}")
    return root / split / "re_aligned" / filename


class ThreePartVQEmbeddingDataset(Dataset):
    """Pair cached three-part VQ vectors with English references."""

    def __init__(
        self,
        embedding_root: Union[str, Path],
        annotation_roots: Mapping[str, Union[str, Path]],
        sources: Sequence[str],
        split: Union[str, Sequence[str]],
        split_index_path: Union[str, Path],
        token_manifest_path: Union[str, Path],
        vq_checkpoint: Union[str, Path],
        max_embedding_steps: int = 0,
    ) -> None:
        splits = (split,) if isinstance(split, str) else tuple(split)
        if not splits:
            raise ValueError("at least one cached VQ split is required")
        invalid_splits = set(splits) - {"train", "val", "test"}
        if invalid_splits:
            raise ValueError(f"unsupported cached VQ splits: {sorted(invalid_splits)}")
        if len(set(splits)) != len(splits):
            raise ValueError(f"duplicate cached VQ splits are not allowed: {splits}")
        if not sources:
            raise ValueError("at least one VQ embedding source is required")
        invalid_sources = set(sources) - set(SUPPORTED_SOURCES)
        if invalid_sources:
            raise ValueError(f"unsupported VQ sources: {sorted(invalid_sources)}")
        if len(set(sources)) != len(sources):
            raise ValueError(f"duplicate VQ sources are not allowed: {list(sources)}")
        if max_embedding_steps < 0:
            raise ValueError("max_embedding_steps must be zero (unlimited) or positive")

        self.embedding_root = Path(embedding_root).expanduser().resolve()
        if not self.embedding_root.is_dir():
            raise FileNotFoundError(
                f"VQ embedding root not found: {self.embedding_root}"
            )
        self.sources = tuple(sources)
        self.splits = splits
        self.max_embedding_steps = int(max_embedding_steps)

        split_index = Path(split_index_path).expanduser().resolve()
        if not split_index.is_file():
            raise FileNotFoundError(f"VQ split index not found: {split_index}")
        with split_index.open(encoding="utf-8") as handle:
            split_payload = json.load(handle)
        allowed_names_by_split: dict[str, set[str]] = {}
        for split_name in self.splits:
            if split_name not in split_payload or not isinstance(
                split_payload[split_name], list
            ):
                raise ValueError(
                    f"VQ split index has no list for {split_name!r}: {split_index}. "
                    "Generate that split with the configured SOKE VQ-VAE first."
                )
            allowed_names = set(split_payload[split_name])
            if not allowed_names:
                raise ValueError(
                    f"VQ split index is empty for {split_name!r}: {split_index}"
                )
            allowed_names_by_split[split_name] = allowed_names

        self.manifest = validate_vq_manifest(
            token_manifest_path,
            vq_checkpoint,
            required_splits=self.splits,
        )
        self.records: list[dict[str, str]] = []
        seen_records: set[tuple[str, str]] = set()

        for source in self.sources:
            if source not in annotation_roots:
                raise ValueError(f"annotation root missing for VQ source {source!r}")
            annotation_root = Path(annotation_roots[source]).expanduser().resolve()
            source_embedding_dir = self.embedding_root / source
            if not source_embedding_dir.is_dir():
                raise FileNotFoundError(
                    f"VQ embedding source directory not found: {source_embedding_dir}"
                )
            available_names = {path.stem for path in source_embedding_dir.glob("*.npy")}

            for split_name in self.splits:
                annotation_path = _annotation_path(
                    annotation_root, source, split_name
                )
                if not annotation_path.is_file():
                    raise FileNotFoundError(
                        f"{source} {split_name} annotations not found: {annotation_path}"
                    )
                allowed_names = allowed_names_by_split[split_name]
                selected_names: set[str] = set()
                with annotation_path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    required_columns = {"SENTENCE_NAME", "SENTENCE"}
                    if not required_columns.issubset(reader.fieldnames or ()):
                        raise ValueError(
                            f"annotations require {sorted(required_columns)}: "
                            f"{annotation_path}"
                        )
                    for row in reader:
                        sample_id = row["SENTENCE_NAME"].strip()
                        if sample_id not in allowed_names:
                            continue
                        selected_names.add(sample_id)
                        if sample_id not in available_names:
                            continue
                        record_key = (source, sample_id)
                        if record_key in seen_records:
                            continue
                        seen_records.add(record_key)
                        caption = row["SENTENCE"].strip() or "unknown"
                        self.records.append(
                            {
                                "source": source,
                                "split": split_name,
                                "name": sample_id,
                                "text": caption,
                            }
                        )

                missing_embeddings = selected_names - available_names
                if missing_embeddings:
                    examples = sorted(missing_embeddings)[:5]
                    raise FileNotFoundError(
                        f"{len(missing_embeddings)} indexed {source}/{split_name} "
                        f"embeddings are missing under {source_embedding_dir}; "
                        f"examples={examples}"
                    )

        if not self.records:
            raise ValueError(
                f"no cached embeddings matched splits={self.splits}, "
                f"sources={self.sources}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, str, str]:
        record = self.records[index]
        source = record["source"]
        sample_id = record["name"]
        path = self.embedding_root / source / f"{sample_id}.npy"
        value = np.load(path)
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError(
                f"VQ embedding must have shape [1,T,3,512], got {value.shape}: {path}"
            )
        value = value[0]
        if value.shape[0] < 1 or value.shape[1:] != (
            VQ_NUM_PARTS,
            VQ_EMBEDDING_DIM,
        ):
            raise ValueError(
                f"VQ embedding must have shape [T,{VQ_NUM_PARTS},"
                f"{VQ_EMBEDDING_DIM}], got {value.shape}: {path}"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"VQ embedding must be floating point, got {value.dtype}: {path}")
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"VQ embedding contains non-finite values: {path}")
        if self.max_embedding_steps:
            value = value[: self.max_embedding_steps]
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        name = f"{source}/{record['split']}/{sample_id}"
        return name, tensor, record["text"], source

    @staticmethod
    def collate_fn(
        batch: list[tuple[str, torch.Tensor, str, str]],
    ) -> tuple[dict[str, object], dict[str, list[str]]]:
        names, embeddings, captions, sources = zip(*batch)
        lengths = torch.tensor(
            [embedding.shape[0] for embedding in embeddings], dtype=torch.long
        )
        padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        positions = torch.arange(padded.shape[1]).unsqueeze(0)
        attention_mask = (positions < lengths.unsqueeze(1)).long()
        src_input: dict[str, object] = {
            "vq_embeddings": padded,
            "attention_mask": attention_mask,
            "src_length_batch": lengths,
            "name_batch": list(names),
            "source_batch": list(sources),
        }
        tgt_input = {"gt_sentence": list(captions)}
        return src_input, tgt_input

    def __str__(self) -> str:
        source_text = "+".join(self.sources)
        split_text = "+".join(self.splits)
        return (
            f"#total {len(self)} ({split_text}, {source_text}, "
            "cached VQ [T,3,512])"
        )
