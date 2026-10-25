# Unified sign model: one mT5 for understanding AND generation

Janus-inspired unified architecture on Uni-Sign, with MoMADiff-style
masked latent generation. This is a sign-specific adaptation, not a
reproduction of Janus, JanusFlow, or MotionGPT3. The research
plan behind this doc lives at
`~/.claude/plans/t5-janus-deep-seek-sunny-popcorn.md`; this file is the
authoritative in-repo version.

## Motivation

- The ST-GCN → mT5 SLT path works well; the frozen-KL-VAE-as-understanding-
  encoder experiment plateaued at BLEU-4 ≈ 2 — reconstruction latents are
  poor understanding features in that local experiment. Janus motivates
  separating semantic understanding features from reconstruction features;
  it does not establish this sign-specific BLEU result.
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
  positions only. Per-sample conditioning dropout zeroes the whole text
  attention mask. JanusFlow's released sampler also masks caption attention,
  but retains an initial token and mixes predicted velocities instead.
- Inference: 9 mask states / 8 transitions in the shrinking-mask MAR loop
  (`MaskGeneration`, empty transitions are skipped), guidance at the
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

## Three training stages (Janus-inspired adaptation)

| Stage | Trainable | Frozen | Data / tasks |
|---|---|---|---|
| 1 Adaptation | latent_in_proj, mask_emb, cond_out_head, diff_head | mT5, ST-GCN, VAE | OpenASL, T2P only |
| 2 Unified | + mT5 | ST-GCN, VAE | OpenASL, SLT+T2P mixed (one batch each per step, summed loss) |
| 3 SFT | + ST-GCN | VAE only | How2Sign, SLT+T2P mixed |

Checkpoints store everything except VAE-owned weights and statistics.
`--slt_init` seeds mT5+ST-GCN for the initial adaptation stage. Later stages
can load the complete preceding unified checkpoint with `--finetune` alone:
`unified_state_dict` also stores the frozen SLT weights. The checked runner
still supplies `--slt_init` before `--finetune`; the latter overrides it.
The checked runner currently initializes from a How2Sign-trained SLT model,
so Stages 1/2 are not an OpenASL-only training history.

Selection: stage 1 → min T2P validation diffusion loss; stages 2/3 → max
SLT BLEU-4. As requested on 2026-10-13, new runs default to `--val_split test`
for both directions on both datasets. `VAL_SPLIT=test` is also the default
in `script/run_bidirectional_checked.sh`; `VAL_SPLIT=dev` explicitly restores
dev selection for OpenASL. How2Sign has no configured dev labels and an
explicit dev request fails rather than silently changing splits.

Each epoch runs SLT loss plus beam-search BLEU-1/2/3/4 and ROUGE-L, then
T2P diffusion loss and full text-to-pose generation on the same selected split.
The generated poses are scored with the confidence-masked DTW-MPJPE2D port
from SOKE/MoMADiff in `pose2d_metrics.py`. `--skip_slg_generation_val` disables
only this generation pass for debugging; by default every sample is scored
every epoch, using the configured MAR/DDIM/CFG settings and a fixed sampling
seed. Sampling preserves the training RNG state. Rank 0 runs the raw model
behind the existing distributed barriers. The extra generation pass adds
validation time. Back-translation remains a separate `gen` evaluation;
BLEURT runs only in explicit SLT evaluation mode, unless skipped.

DTW protocol (identical cost/aggregation to the supplied SOKE/MoMADiff files):
native, de-z-normalized xy coordinates; GT confidence > 0; drop GT frames
without detected joints for the selected part; independent alignment for
body/lhand/rhand/face/all. A sequence score is accumulated DTW cost WITHOUT
dividing by path length. Dataset scores average these costs over evaluable
sequences; hand = (mean lhand + mean rhand)/2. Hands and face use the existing
wrist/anchor-local representation. DTW uses separate true generated/GT
lengths, before any resampling for the SLT judge. Generated geometry is not
additionally clipped for DTW; the SLT judge keeps its prior [-1,1] clipping.

Each stage also saves `best_slg_checkpoint.pth` at the minimum
`{dataset}_DTW_MPJPE2D_hand`. `best_checkpoint.pth` retains the original SLT
BLEU-4 criterion in joint stages (diffusion loss in Stage 1). W&B includes
keys such as `test/openasl_DTW_MPJPE2D_hand` and coverage counts. Resume
preserves the best SLG score only when the saved split/sampling/codec protocol
matches. The checked runner's `gen` mode defaults to the Stage-3 SLG-best
checkpoint; `GEN_CHECKPOINT` can select another stage explicitly.

Logs record `validation_split`, and split-qualified W&B keys such as
`test/slt_bleu4` and `test/t2p_loss` distinguish them from older dev curves.
Best-checkpoint metadata and full-state resume metadata also record the
selection split. Switching dev/test on resume keeps weights and optimizer
state but resets the best selection metric, so the two splits do not compete.
If a run has already completed all epochs, a resumed invocation cannot
reselect it on test without further evaluation/training; its existing best
checkpoint still follows the original protocol. Existing running processes
keep their loaded validation code until restarted.

These are test-selected checkpoints: repeated selection on test uses that
split for model selection. Report this setting when comparing published
results; the scores are not independently held-out final-test estimates.

## Fidelity to Janus and JanusFlow

The common design is separate understanding/generation encoders with a
shared language backbone and progressive unfreezing. Important differences:

- Janus uses autoregressive discrete image tokens and cross-entropy for
  generation. This implementation uses continuous KL-VAE latents and x0
  diffusion MSE with masked latent slots.
- JanusFlow predicts rectified-flow velocities, feeding the changing noisy
  latent and timestep through its shared LLM at every Euler step. Here mT5
  produces conditioning once per MAR transition, and the diffusion MLP
  performs the subsequent denoising steps.
- JanusFlow uses representation alignment (REPA). No REPA loss is implemented
  here. Stage 1 here is T2P-only rather than both understanding/generation;
  Stages 2/3 have no text-only replay or JanusFlow data-ratio schedule.
- Guidance here extrapolates conditioning vectors before a nonlinear head.
  It is not mathematically equivalent to JanusFlow's velocity-output CFG.
- MotionGPT3 uses two modality streams sharing attention; this model uses
  the same mT5 encoder parameters for both tasks.

Primary references: [Janus](https://arxiv.org/abs/2410.13848),
[JanusFlow](https://arxiv.org/html/2411.07975v1),
[MotionGPT3](https://arxiv.org/html/2506.24086v2).

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
pass `--resume off`, or `RESUME=off` to the checked runner. This skips old
weights/optimizer recovery, starts a new W&B run, and writes new recovery
snapshots into the same output directory. Same-named model checkpoints are
overwritten as training progresses; existing epoch logs remain appended.

## Evaluation ladder (run `ceilings` BEFORE any training)

| Mode | Measures | Needs trained gen model? |
|---|---|---|
| `gt` | judge BLEU ceiling on real poses | no |
| `gt_conf1` | cost of conf=1 (generated poses carry no confidence) | no |
| `vae_recon` | encode→decode then back-translate; MPJPE2D + masked DTW-MPJPE2D | no |
| `gen` | text→pose back-translation BLEU + MPJPE2D + masked DTW-MPJPE2D (GT-length oracle) | yes |

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
- The 2026-10-12 audit passed real CPU forward/backward contracts on one
  OpenASL dev clip from `unified_stage2_openasl/checkpoint_1.pth`; see
  `validation/audit_unified_pipeline.py` and `out/janus_contract_audit_20261012.json`.
  This is a short execution check, not a full quality or distributed audit.
- The active Stage 2 log contains extremely large finite SLT loss aggregate
  values (about 1e26). The cause is unresolved; normal-looking dev results
  do not prove those training spikes are harmless. Diagnose the offending
  batch/rank and gradient behavior before using this run as an RL baseline.
