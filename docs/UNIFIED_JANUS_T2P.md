# Unified sign model: one mT5 for understanding AND generation

JanusFlow/MotionGPT3-style unified architecture on Uni-Sign. The research
plan behind this doc lives at
`~/.claude/plans/t5-janus-deep-seek-sunny-popcorn.md`; this file is the
authoritative in-repo version.

## Motivation

- The ST-GCN → mT5 SLT path works well; the frozen-KL-VAE-as-understanding-
  encoder experiment plateaued at BLEU-4 ≈ 2 — reconstruction latents are
  poor understanding features (Janus's core finding).
- Janus resolution: decouple. The semantic encoder (ST-GCN ≙ SigLIP) serves
  understanding; the reconstruction autoencoder (KL-VAE ≙ SDXL-VAE) serves
  generation; ONE LLM hosts both.
- Generation machinery is MoMADiff's proven recipe (masked-AR transformer +
  per-token MLP diffusion head over continuous KL-VAE latents), re-targeted
  onto the shared mT5 encoder (natively bidirectional → MAR needs no
  attention surgery).

## Architecture

```
SLT:  pose parts → ST-GCN ×4 → Linear(1024,768) ─┐
                                                  ├→ shared mT5 encoder → decoder → text (CE)
T2P:  [T2P prompt+caption embeds ; latent slots] ─┘
      slot = Linear(512→768)(latent)  or  mask_emb   (+ sinusoidal PE)
      encoder hidden @ slots → cond_out_head(768→512) → LDM diffusion head
      → sampled 512-D latents → frozen KL-VAE decoder → [T,138] → de-norm
      → body/left/right/face_all (conf=1)
```

- KL-VAE encoder+decoder (`net_best_hand.pth`) permanently frozen; never in
  unified checkpoints (loaded from `--vae_checkpoint`; `strict=False`).
- Training (MoMADiff recipe): frozen encoder → `posterior.sample()` GT
  latents at per-sample true length (grouped; padding never crosses the
  non-causal convs) → cosine random masking → diffusion MSE on masked
  positions only. Per-sample CFG dropout zeroes the text attention mask
  (JanusFlow's uncond construction).
- Inference: 9-step shrinking-mask MAR loop (`MaskGeneration`), CFG at the
  cond-vector level (`uncond + (cond−uncond)·scale`, guidance 3.0), DDIM-100
  per newly-unmasked token, grouped true-length decode.

## Files

| File | Role |
|---|---|
| `models_unified.py` | `Uni_Sign_Unified(Uni_Sign)`: SLT inherited; `forward_t2p`, `generate_pose`, part↔138D conversion, frozen codec |
| `vae_decoder.py` | Frozen decode-only KL-VAE port (`decoder.*`, `post_quant_conv.*`; strict load) + `posterior_sample` |
| `vae_encoder.py` | (existing) encode-only port, input dim 138 |
| `diffusion_gen/` | guided-diffusion port from MoMADiff: schedules, SpacedDiffusion, DDIM, x0-MSE `training_losses`, uniform sampler, `factory.py` |
| `diffusion_head.py` | `LDM` + `MLP_Denoiser` (DiT AdaLN on cond+timestep) |
| `mask_generation.py` | `MaskGeneration` shrinking-mask schedule + `lengths_to_mask`/`cosine_schedule` |
| `train_unified.py` | 3-stage entrypoint (below) |
| `back_translation_eval.py` | gt / gt_conf1 / vae_recon / gen evaluation ladder |
| `script/train_unified.sh` | `bash script/train_unified.sh 1|2|3|eval` |
| `script/eval_backtrans.sh` | `bash script/eval_backtrans.sh ceilings|gen` |

## Three training stages (JanusFlow recipe)

| Stage | Trainable | Frozen | Data / tasks |
|---|---|---|---|
| 1 Adaptation | latent_in_proj, mask_emb, cond_out_head, diff_head | mT5, ST-GCN, VAE | OpenASL, T2P only |
| 2 Unified | + mT5 | ST-GCN, VAE | OpenASL, SLT+T2P mixed (one batch each per step, summed loss) |
| 3 SFT | + ST-GCN | VAE only | How2Sign, SLT+T2P mixed |

Checkpoints store everything except the VAE codec. `--slt_init` (a trained
ST-GCN SLT checkpoint) must be passed in EVERY stage: it seeds mT5+ST-GCN,
and frozen modules are excluded from `get_requires_grad_dict`-style saving
only in the SLT baseline — here `unified_state_dict` keeps them, so stage
N+1 needs only `--slt_init` + `--finetune stageN/best_checkpoint.pth`.

Selection: stage 1 → min T2P val diffusion loss; stages 2/3 → SLT BLEU-4
(dev where it exists, else test — inherited convention).

### Crash recovery (auto-resume, epoch granularity)

Every stage snapshots the FULL training state each epoch via DeepSpeed
`engine.save_checkpoint` into `<output_dir>/resume_state/`: module weights,
per-rank ZeRO-2 optimizer shards, lr-scheduler state, plus
`{epoch+1, best_metric}` as client state. The new epoch tag is written
before rank 0 prunes the previous one, so an interruption mid-save leaves
`latest` pointing at the older complete snapshot.

On startup (`--resume auto`, the default), a stage that finds
`resume_state/latest` reloads it AFTER `--slt_init`/`--finetune` (resumed
weights win) and continues from the stored epoch; a fully finished stage
fast-forwards through its loop in seconds — rerunning
`bash script/train_unified.sh all` after a crash is therefore idempotent.
The wandb run id is persisted to `<output_dir>/wandb_run_id.txt` and
re-attached with `wandb.init(id=…, resume="allow")`, keeping one curve per
stage across restarts.

Constraints: resume requires the SAME world size (ZeRO shards are
per-rank); RNG streams and the partial epoch in flight are not restored
(at most one epoch of work is lost). To restart a stage from scratch,
delete its `resume_state/` and `wandb_run_id.txt` (or pass `--resume off`).

## Evaluation ladder (run `ceilings` BEFORE any training)

| Mode | Measures | Needs trained gen model? |
|---|---|---|
| `gt` | judge BLEU ceiling on real poses | no |
| `gt_conf1` | cost of conf=1 (generated poses carry no confidence) | no |
| `vae_recon` | generation quality ceiling: encode→decode then back-translate; + MPJPE2D | no |
| `gen` | text→pose back-translation BLEU + MPJPE2D (GT-length oracle) | yes |

Judge = frozen trained ST-GCN SLT checkpoint; sequences longer than 256
frames are linearly resampled before judging (the judge never saw more).
Interpret `gen` BLEU relative to `vae_recon` and `gt`, not in absolute terms.

## Commands

```bash
# sanity + ceilings (no training needed)
python -c "import vae_decoder, diffusion_head, mask_generation, models_unified, train_unified, back_translation_eval"
bash script/eval_backtrans.sh ceilings

bash script/train_unified.sh 1
bash script/train_unified.sh 2
bash script/train_unified.sh 3
bash script/train_unified.sh eval      # SLT metrics of the unified model
bash script/eval_backtrans.sh gen      # T2P quality
```

## Deliberate deviations from upstream MoMADiff

- `DiTAdaLayerNorm` uses the correct `(x − mean)/sqrt(var + eps)`
  normalization; upstream's `x − mean/(var+eps)` mis-parenthesization breaks
  shift invariance. The head trains from scratch here, so nothing is lost.
- Inference uses TRUE DDIM (`ddim_sample_loop`, eta=0) — upstream called
  `p_sample_loop` on the respaced process, which is ancestral sampling.
- Train-time masking ranks random scores among valid positions only, so
  every sample keeps ≥1 supervised token; the loss normalizes by masked-token
  count, not padded batch length.
- Rank-0 validation runs on the raw module behind dist barriers (the
  wrapped model's collectives would deadlock against other ranks).
- Checkpoint contract is enforced on load: unified checkpoints may miss only
  VAE-owned keys (`vae_encoder.*`, `vae_decoder.*`, `motion_mean/std` — the
  stats always follow `--vae_mean_path/--vae_std_path`); an SLT checkpoint
  passed as `--finetune` fails loudly instead of silently keeping a random
  generation head.

## Notes / open items

- SLT init + back-translation judge:
  `/data/hwu/Uni_sign_repo/out/how2sign_pose_only/how2sign_pose_only_slt.pth`
  (How2Sign pose-only ST-GCN SLT checkpoint), wired into both scripts.
- Generation length uses the GT-length oracle (MoMADiff protocol); a
  text→length predictor is future work.
- `vae_recon` MPJPE2D should land in the same range as the MoMADiff stage-1
  `test_scores.json` — a cheap correctness check of the decoder port.
- Stage-1 T2P training with frozen mT5 relies on the encoder being a good
  frozen text/motion mixer; if stage-1 loss plateaus high, move faster to
  stage 2 (mT5 unfrozen) rather than extending stage 1.
