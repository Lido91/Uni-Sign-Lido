# Implementation Plan 1: SignSMD Pretext Pretraining

Standalone plan for **Idea 1** of
`signsmd_pretext_and_dual_stream_fusion.md` (§1: describe/match pretext
tasks with compiler-generated counterfactual negatives). This plan does
**not** depend on the dual-stream work (`signsmd_dual_stream_fusion_plan.md`);
its deliverable — a multimodally-pretrained checkpoint — is evaluated on its
own via E2 (pretrain → early fusion) and later feeds E4 in the dual-stream
plan.

## Scope

- In: counterfactual negative generation, pretext dataset, multi-task model,
  Stage-1 pretraining entrypoint, E2 experiment.
- Out: any fusion-architecture change (early fusion `Uni_Sign_SignSMD` is
  reused as-is for E2).

## Design recap (details in the design doc §1)

- Three tasks share one T5 encoder/decoder/LM head, routed by **plain-word
  prefixes** (no new special tokens → the zero-new-params / `strict=True`
  checkpoint invariant is preserved in both directions):
  - translate: `"Translate sign language video to English: "` + SMD + pose → caption (unchanged parent behavior);
  - describe: `"Describe the signing motion: "` + pose → SMD text;
  - match: `"Does the following description match the signing? "` + SMD + pose → `yes`/`no`.
- Batch-level task sampling `--task_ratios 0.7,0.2,0.1`; SLT stays dominant
  so dev-BLEU-4 checkpoint selection remains meaningful.
- **Token-imbalance control**: describe targets capped
  (`--describe_max_target_tokens 128`, prefer the compact `coarse` variant),
  per-task loss scales available, per-task loss meters always logged.
- **Matching negatives**: 50/50 pos/neg; negatives predominantly
  compiler-counterfactual (`--match_neg_mix hard=0.8,random=0.2`), random
  negatives length-matched.
- Pretraining corpus: **OpenASL**; E2 finetunes on How2Sign.

## Phase P1 — Counterfactual negatives (compiler side)

**P1.1 `signsmd2d/counterfactuals.py`** (new)
- Perturbation operators applied to the evidence dict *before*
  `segment`/`realize`:

  | Operator | Evidence perturbation | Text effect |
  |---|---|---|
  | `swap_hands` | exchange left/right evidence subtrees | hand identity flips |
  | `flip_vertical` | negate y of wrist trajectories | upward ↔ downward |
  | `mirror_horizontal` | mirror x about body midline | side of space flips |
  | `shift_location` | offset wrist positions one zone | chest ↔ face, etc. |
  | `alter_handshape` | permute finger-curl channels | fist ↔ open flat, etc. |

- `realize_counterfactual(pose, cfg, op, ...)`: perturb → re-segment →
  re-realize (coarse) → `compiler.assert_clean` → assert text differs from
  the unperturbed realization (mark clip/op pairs where it doesn't, e.g.
  `swap_hands` on symmetric motion).
- Deterministic given (clip, op); no RNG inside the compiler.

**P1.2 Offline generation** — extend the existing compile entrypoint with
`--negatives swap_hands,flip_vertical,mirror_horizontal,shift_location,alter_handshape`:
- writes `data/<ds>/signsmd2d_v0_1/negatives/<clip>.<op>.txt`;
- writes `negatives/manifest.jsonl` rows `{clip, op, ok}` (ok=false when the
  op produced identical text) — used for per-attribute diagnostics.

## Phase P2 — Pretext dataset

**`datasets_signsmd_pretext.py`** (new):
`S2T_Dataset_SignSMD_Pretext(S2T_Dataset_SignSMD)`
- New args: `--signsmd_neg_root`, `--task_ratios`, `--match_neg_mix`,
  `--describe_max_target_tokens`.
- `__getitem__` additionally loads the clip's available negative texts.
- **`collate_fn` samples ONE task per batch** (train phase only) and attaches:
  - `src_input["pretext_task"] ∈ {"translate","describe","match"}`;
  - match: per-sample candidate text (50% pos / 50% neg) and
    `tgt_input["gt_sentence"] = ["yes"/"no", ...]`;
  - describe: `tgt_input["gt_sentence"] = SMD text` (model applies the cap).
- Dev/test loaders stay plain `S2T_Dataset_SignSMD` (pure SLT) — BLEU-4
  selection untouched.

## Phase P3 — Multi-task model

**`models_signsmd_pretext.py`** (new):
`Uni_Sign_SignSMD_MultiTask(Uni_Sign_SignSMD)`
- `forward` branches on `src_input["pretext_task"]` (absent → `"translate"`
  → exact parent behavior, so eval is unchanged);
- describe labels tokenized with
  `max_length=describe_max_target_tokens`; match labels are `yes`/`no`;
- per-task loss scaling via `--task_loss_scales` (default 1,1,1); returns
  `loss` plus a `task` key for logging;
- zero new parameters ⇒ `strict=True` round-trip with baseline checkpoints.

## Phase P4 — Pretraining entrypoint

**`pretraining_signsmd.py`** (new), mirroring `fine_tuning_signsmd.py`:
- reuses `fine_tuning.train_one_epoch` / `fine_tuning.evaluate` unchanged
  (branching lives in dataset + model);
- per-epoch **matching-accuracy probe** over a fixed dev subset, reported
  overall and per perturbation op (P1.2 manifest);
- per-task train loss meters (`loss_translate`, `loss_describe`,
  `loss_match`);
- checkpoint selection: dev BLEU-4 (baseline convention).

## Phase P5 — Experiments

```bash
# P1.2: compile negatives (OpenASL first; How2Sign only needed for analysis)
python <compile_entrypoint> ... \
    --negatives swap_hands,flip_vertical,mirror_horizontal,shift_location,alter_handshape

# Stage 1: multimodal pretraining on OpenASL, warm-started from baseline
deepspeed pretraining_signsmd.py --dataset OpenASL --task SLT --dtype fp32 \
    --finetune <baseline_openasl_ckpt.pth> \
    --signsmd_text_root data/OpenASL/signsmd2d_v0_1/texts \
    --signsmd_neg_root  data/OpenASL/signsmd2d_v0_1/negatives \
    --task_ratios 0.7,0.2,0.1 --output_dir out/pretext_openasl

# E2: pretrained checkpoint → EARLY fusion finetune on How2Sign (existing path)
deepspeed fine_tuning_signsmd.py --dataset How2Sign --task SLT --dtype fp32 \
    --finetune out/pretext_openasl/best_checkpoint.pth \
    --signsmd_text_root data/How2Sign/signsmd2d_v0_1/texts \
    --output_dir out/e2_early_pretrained
```

Success criteria:
- **E2 vs E1** (early fusion without pretraining) BLEU-4/ROUGE on How2Sign —
  the direct test of the "missing alignment training" hypothesis;
- matching probe accuracy well above chance, and per-attribute breakdown
  showing which attributes the encoder actually reads;
- `out/pretext_openasl/best_checkpoint.pth` loads into baseline
  `models.Uni_Sign` with `strict=True` (invariant check), and is handed to
  the dual-stream plan for E4.

## Smoke tests (run before any GPU time; user runs all Python)

1. Compile ~5 clips with all ops; check each negative differs in exactly the
   intended attribute and passes `assert_clean`.
2. Confirm `yes`/`no` and both task prefixes tokenize to short mT5 pieces
   (no `<unk>`).
3. One forward+backward per task branch on a 2-sample batch
   (`--signsmd_on_missing empty` where texts are incomplete).
4. Pretrained checkpoint ↔ baseline `strict=True` round-trip.

## Deliverables

| File | Status |
|---|---|
| `signsmd2d/counterfactuals.py` | new |
| compile entrypoint | modified (`--negatives`, manifest) |
| `datasets_signsmd_pretext.py` | new |
| `models_signsmd_pretext.py` | new |
| `pretraining_signsmd.py` | new |
| `out/pretext_openasl/best_checkpoint.pth` | trained artifact (feeds E4) |
