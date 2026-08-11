# ST-GCN → pose2d KL-VAE + one head into T5 space

## Scope

Additive sign-to-text path that **fully replaces the ST-GCN pose encoder**
with the MoMADiff KL-VAE jointly trained on How2Sign + OpenASL 138D RTMPose
xy features (`MoMADiff/configs/klvae_h2s_openasl_pose2d.yaml`). The VAE
posterior mean is projected by **one linear head** directly into the mT5
embedding space.

The original implementation stays untouched and default
(`models.py` / `datasets.py` / `fine_tuning.py`); the earlier 133D canonical
VAE path (`models_vae.py` / `fine_tuning_vae.py`) also stays as is.

## Runtime flow

```text
datasets_pose2d_klvae.S2T_Dataset_Pose2D_KLVAE
  (native How2Sign/OpenASL RTMPose pickles; deterministic linear-index
   resampling into [40, max_length], matching the VAE corpus rule)
  -> body [B,T,9,3] | left [B,T,21,3] | right [B,T,21,3] | face_all [B,T,18,3]
  -> drop conf, flatten xy in MoMADiff part order  -> [B,T,138]
  -> z-score with kl_vae_h2s_openasl_pose2d/meta/{mean,std}.npy
  -> PER SAMPLE: truncate to the sample's true length (multiple of 4),
     encode with the FROZEN KL-VAE (grouped by equal length, torch.no_grad),
     posterior mean [T_i/4,512], then pad latents to [B,T_max/4,512]
  -> ONE Linear(512, 768) head (pose_proj)  -> mT5 inputs_embeds
  -> prepend "Translate sign language video to English: " prompt embeddings
  -> token-level cross-entropy / beam generation (unchanged Uni-Sign objective)
```

Per-sample encoding is required for correctness: the VAE is a non-causal
temporal CNN with a large receptive field, so encoding the collate-padded
batch would leak repeat-last padding into valid latents (the attention mask
cannot undo that, and a sample's representation would depend on the other
samples' lengths). The latent `attention_mask` is `true_length // 4` (min 1).

## Feature-contract alignment with the VAE

`datasets.load_part_kp` and MoMADiff `utils/pose2d_repr.rtmpose_to_pose2d`
compute the same representation (body square-bbox norm to [-1,1]; hands minus
per-frame wrist / body scale; face minus anchor joint 53 / body scale;
joints with conf ≤ 0.3 zero-filled; clip to [-1,1]; part order
body|left|right|face). Temporally, the default `--frame_sampling vae_linear`
reuses the VAE corpus rule (deterministic linear-index resampling into
[40, max_length]; in-range clips kept whole), eliminating the speed/
continuity shift that the original random subsampling would introduce for a
temporal-conv encoder — and making evaluation deterministic for long clips.
`--frame_sampling unisign_random` restores the original sampling for
ablations. The remaining benign difference is bbox timing (Uni-Sign computes
it on resampled frames, the corpus on all frames before resampling); linear
resampling covers the clip uniformly so bbox extremes are nearly identical.

## Files

| File | Purpose |
|---|---|
| `vae_encoder.py` | Generalized: `input_emb_width` now configurable (default 133 unchanged; this path passes 138); weight-dtype-following forward |
| `models_pose2d_klvae.py` | `Uni_Sign_Pose2D_KLVAE`: flatten → z-score → per-sample KL-VAE → `Linear(512,768)` → mT5 |
| `datasets_pose2d_klvae.py` | `S2T_Dataset_Pose2D_KLVAE`: VAE-corpus-consistent linear frame resampling |
| `fine_tuning_pose2d_klvae.py` | Entrypoint; reuses `fine_tuning.train_one_epoch/evaluate`, original data pipeline, BLEU-4 selection |
| `script/train_openasl_then_how2sign_klvae.sh` | Automated pipeline: OpenASL pretrain (best selected on How2Sign test via `--eval-on-how2sign-test`) → How2Sign finetune → final eval, with `set -e` |
| `script/train_pose2d_klvae.sh` | Stage-selectable runner: `pretrain` / `finetune` / `eval` / `all` |

New CLI args on top of the standard parser: `--vae_checkpoint` (required),
`--vae_mean_path`, `--vae_std_path` (default to the
`kl_vae_h2s_openasl_pose2d/meta` copies),
`--frame_sampling vae_linear|unisign_random` (default `vae_linear`).

The VAE encoder is **permanently frozen** (kept in eval mode, forward under
`torch.no_grad()`): only the linear head and mT5 train, so the encoder stays
numerically identical to the MoMADiff checkpoint. Saved checkpoints therefore
contain only head + mT5 (+ the mean/std buffers); the VAE always comes from
`--vae_checkpoint`, and `--finetune` loads with `strict=False`.

## Notes

- Restricted to `--dataset How2Sign|OpenASL`, `--task SLT`, `--llm_type mt5`,
  no `--rgb_support`.
- Checkpoints of this path are NOT loadable into `models.Uni_Sign`
  (different parameter families), matching the repo's additive-path
  convention.
- The stage-1 VAE checkpoint is
  `MoMADiff/checkpoints/how2sign/kl_vae_h2s_openasl_pose2d/net_best_hand.pth`
  (the `{'net': ...}` format handled by `vae_encoder.py`'s loader), already
  wired into `script/train_pose2d_klvae.sh`.
- Not smoke-run here (per project rule the user runs Python).
