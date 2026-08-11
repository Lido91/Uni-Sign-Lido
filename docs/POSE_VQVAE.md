# Multi-Part 2D Pose VQ-VAE (adapted from SOKE DETO)

Ports SOKE's decoupled tokenizer (T2M-GPT style 1D-conv VQ-VAE per body part)
from 133-D SMPL-X motion features to Uni-Sign's RTMPose 2D pose input.

## Mapping from SOKE to this repo

| | SOKE (DETO) | Here |
|---|---|---|
| Input | flat `[T, 133]` SMPL-X features | four parts `[T, V, 3]` (x, y, conf) |
| Parts | 3: body(43) / lhand(45) / rhand(45), sliced from flat vector | 4: body(9) / left(21) / right(21) / face_all(18), from `datasets.load_part_kp` |
| Codebooks | body 96, hands 192 (`configs/vq/*.yaml`) | same defaults; face 96 (all overridable) |
| Encoder/decoder | Conv1d + Resnet1D, down_t=2 → 1 token / 4 frames | identical stack, identical rate |
| Quantizer | EMA-reset (T2M-GPT) | ported unchanged |
| Recon target | full feature vector, smooth-L1 | xy only; confidence channel is encoder input + loss mask |
| Loss masking | none needed | joints with conf ≤ 0.3 (zeroed by loader) and padded frames excluded |

## Files

- `pose_vqvae/vqvae_2d.py` — single-part `PoseVQVae2D` (`forward` / `encode` / `decode`)
- `pose_vqvae/multipart.py` — `MultiPartPoseVQVAE` (4 parts, dict in/out) and
  `MultiPartReconLoss` (confidence + padding masked)
- `pose_vqvae/resnet.py` — verbatim port from SOKE
- `pose_vqvae/quantize.py` — port from SOKE with two fixes: EMA/init state is
  stored in buffers (survives checkpoint save/load), and forward() accepts a
  latent-step mask excluding padded steps from EMA updates, commitment loss,
  and perplexity
- `pose_vqvae/data.py` — `DeterministicS2T_Dataset`: evenly spaced frame
  subsampling for eval/extraction (base loader random-samples even at eval)
- `train_pose_vqvae.py` — tokenizer training on `S2T_Dataset` batches,
  joint multi-dataset by default (OpenASL + How2Sign)
- `extract_pose_tokens.py` — dump per-clip token pickles (analog of SOKE's
  `scripts/get_motion_code.py`)

## Usage

```bash
# 1) Train the tokenizer — joint OpenASL + How2Sign (default)
python train_pose_vqvae.py --output_dir out/pose_vqvae_asl
#    or a single dataset:
python train_pose_vqvae.py --datasets OpenASL --output_dir out/pose_vqvae_openasl

# 2) Extract tokens for generator training (per dataset/split)
python extract_pose_tokens.py --dataset OpenASL --split train \
    --checkpoint out/pose_vqvae_asl/checkpoint_best.pth \
    --out_dir data/OpenASL/pose-tokens
python extract_pose_tokens.py --dataset How2Sign --split train \
    --checkpoint out/pose_vqvae_asl/checkpoint_best.pth \
    --out_dir data/How2Sign/pose-tokens
```

Token layout per frame group (4 frames → 1 step): one code per part, i.e. a
4-stream sequence `(body, left, right, face_all)` — the 4-head analog of SOKE's
3-head (body, lhand, rhand) multi-head decoding.

## Notes / caveats

- Batches are padded in time by repeating the last frame (Uni-Sign collate);
  `prepare_batch` additionally pads T to a multiple of 4 and extends the
  attention mask with zeros. Padded frames are excluded from the
  reconstruction/velocity loss AND (via a downsampled latent mask) from
  codebook EMA updates, commitment loss, and perplexity.
- Hands/face are in local (wrist/anchor-relative) coordinates divided by the
  body bbox scale — already decoupled, which matches SOKE's motivation for
  separate codebooks per part.
- Frame subsampling of clips over `--max_length` (default 256): random per
  epoch for training (temporal augmentation), evenly spaced and deterministic
  for validation and token extraction (`DeterministicS2T_Dataset`).
- How2Sign has no dev split; `train_pose_vqvae.py` holds out a seeded slice of
  its training labels for model selection (`--holdout_size`, removed from the
  training set). The How2Sign test split is never used during training.
- Resume (`--resume`) is safe: quantizer EMA sums/counts and the init flag are
  registered buffers, so a resumed run continues from the trained codebook
  instead of re-initializing it.
- `MultiPartPoseVQVAE.freeze()` is persistent: a later `.train()` from a parent
  module keeps the VAEs in eval mode, so EMA codebook updates cannot run on a
  frozen tokenizer during downstream LM training.
- The decoder reconstructs only (x, y); confidence is treated as observation
  metadata, not a signal to generate.
- Downstream generator integration (a token reader / 4-head LM head on mT5 or
  mBart) is NOT part of this checkout yet; `extract_pose_tokens.py` output is
  the interface for it.
