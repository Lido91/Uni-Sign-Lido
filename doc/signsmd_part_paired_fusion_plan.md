# Experiment: SignSMD Part-Paired Text–Pose Fusion (time-aligned injection)

Self-contained experiment on top of the **baseline** `models.Uni_Sign` — it
does NOT depend on the dual-stream implementation
(`signsmd_dual_stream_fusion_plan.md`); the two lines share only the
compiled-text corpora. The generic entrypoint can warm-start an ordinary
baseline checkpoint; the staged launcher trains paired OpenASL first and then
warm-starts paired How2Sign from that checkpoint.

## 1. Motivation and Hypothesis

Three facts line up:

1. **Both sides are part-factored.** The pose encoder runs one stream per
   part (`body / left / right / face_all`); the SignSMD stats channels
   partition into the same four articulators.
2. **The two modalities are step-synchronized by construction.** The SMD
   compiler resamples with the trainer's rule, so *text step i = pose token
   i* (rules doc §3); every event carries an inclusive `[step a-b]` span.
3. **The codebase already validated this fusion pattern.** PGF
   (`gather_feat_pose_rgb`) injects a second modality (RGB hand crops) into
   the hand GCN streams *at synchronized frame indices*, behind a
   zero-initialized gate — and improves translation.

**Hypothesis (H-inject):** injecting each part's description events into
that part's pose stream at exactly the steps the event describes (the text
analogue of PGF) supplies the decoder with compiled articulation semantics
without disturbing the pose representation — avoiding the E1 early-fusion
failure, because the mT5 input sequence is unchanged and the injection
starts at the identity.

Sequence-level concatenation of part texts (a part-interleaved encoder
input) is explicitly rejected as the primary design: self-attention cannot
see that `[step a-b]` refers to pose positions a..b, so it wastes the
synchronization, multiplies encoder length by ~4, and destroys baseline
warm-start. It remains a fallback (§7).

## 2. Compiled corpus: `parts` variant with structured events

### 2.1 Compiler — coarse-style descriptions, segmented per part

New variant `--variant parts` (contract `signsmd2d_parts_v0_2`), built on
the **coarse** rules (evidence §7, categories §9.2–9.5 of the rules doc),
NOT on the stats numeric timelines. Coarse phrases are categorical
("fist with the index finger extended", "at head height, close to the
face", "moves fast along a straight path") — a small phrase vocabulary,
which suits the embed-and-cache injection design (§3.1) far better than
numeric spans.

Two changes relative to the existing coarse variant:

1. **Per-part activity segmentation.** The existing coarse variant
   segments once, globally, on the max normalized rate across ALL
   articulators (§9.1) — one hand moving marks the face "active" too.
   Here each part is segmented on its own activity channels, using the
   existing per-articulator thresholds unchanged:

   | Part | Activity channels (existing thresholds) |
   |---|---|
   | `left` / `right` | own handshape rate + palm-orientation rate (config only — no wrist/arm channels) |
   | `body` | inter-hand separation rate; both sides' wrist speed + arm rate |
   | `face_all` | brow rate, mouth-opening rate, head-tilt rate |

   Same smoothing (5-step), same activity rule (>1.0), same minimum
   interval length (8 steps, short intervals merged).

   **Known risk — interval collapse.** Inspection of the existing
   How2Sign `signsmd2d_v0_2` corpus shows whole clips (incl. 13s ones)
   collapsing to a single global `movement` interval (the long-interval
   issue already noted in `docs/SIGNSMD_TWO_VARIANT_PLAN.md`). Spans with
   no temporal resolution make time-aligned injection degenerate to
   whole-clip broadcast. Per-part segmentation must therefore be
   validated for granularity, not just correctness: the compiler smoke
   run (§6.1) reports mean intervals-per-part, and if hand parts still
   average ≈1 interval on normal-length clips, the per-part activity
   threshold switches from the fixed 1.0 to a per-clip quantile of that
   part's smoothed activity (flagged in the manifest).

2. **Per-interval, per-part realization.** Each (part, interval) becomes
   one event with the interval's `[step a-b]` span:

   - `left`/`right` — **finger-level articulation ONLY** (no wrist
     location, no arm bend, no wrist motion — those are body events,
     matching the pose encoder where hand streams carry articulation and
     location arrives via the body-anchor addition): handshape category
     (§9.2, omitted if unsupported) + per-finger
     extended/partially-curled/closed clause when the category leaves the
     configuration underdetermined (absent / curved / pinching) + up to
     `finger_events_max` (default 2) strongest per-finger net curl
     motions per interval ("the index finger curls", threshold
     `finger_curl_change_min` ≈ 40°, fingers-variant style; they replace
     the aggregate "fingers curling overall" wording when present) +
     palm-orientation appearance + finger direction. Kinds: hold /
     transition.
   - `body` — the arms and bimanual structure, per interval: one event
     per tracked arm (`left arm: wrist at chest height near the body
     midline; arm partially bent; moves upward along a curved path at
     slow speed`) with wrist zone (§9.3), elbow bend, and wrist motion
     category (§9.4); one bimanual-coordination event (hands
     apart/together/overlap, articulates-while-base, mirror symmetry);
     plus ONE clip-level event spanning all steps for the bilateral role
     (§9.5).
   - `face_all`: brow raised/lowered vs clip-median neutral, mouth
     opening/closing, head tilt — per interval.

   Hold intervals emit a phrase too ("right hand holds at chest height,
   fist") — a hold is information. A part whose detection is unreliable
   (§5) emits no events.

The forbidden-vocabulary assertion (§12) applies to every phrase.

### 2.2 Output: one text per part plus the event contract

Per clip, the compiler writes:

```text
<root>/texts/body/<clip>.txt      # body events only
<root>/texts/left/<clip>.txt      # left-hand events only
<root>/texts/right/<clip>.txt     # right-hand events only
<root>/texts/face_all/<clip>.txt  # face events only
<root>/events/<clip>.json         # structured event/span contract
```

No text file mixes parts. At training time each event phrase from one part is
tokenized independently with `--signsmd_part_max_tokens`, so a long body
description cannot truncate hand or face text. The JSON keeps all events
together as structured records because consumers select them by the explicit
`part` field without tokenizing the JSON as one sequence.

`events/<clip>.json`:

```json
{"contract": "signsmd2d_parts_v0_2", "num_steps": 214,
 "events": [
   {"part": "right", "kind": "transition",
    "phrase": "right hand: fist with the index finger extended; the index finger curls; palm appears to face the camera",
    "start": 138, "end": 146},
   {"part": "left", "kind": "hold",
    "phrase": "left hand: fist; palm edge-on to the camera",
    "start": 130, "end": 170},
   {"part": "body", "kind": "movement",
    "phrase": "right arm: wrist at head height on the signer-right side, close to the face; arm strongly bent; moves upward along a straight path at fast speed",
    "start": 120, "end": 155},
   {"part": "face_all", "kind": "movement",
    "phrase": "eyebrows raised; mouth open", "start": 0, "end": 41},
   {"part": "body", "kind": "role",
    "phrase": "the right hand acts as the dominant articulator; the left hand serves as the supporting or base hand",
    "start": 0, "end": 213}
 ]}
```

- `start`/`end` inclusive, in trainer-aligned steps (0-based).
- Static/hold events are included (a hold is information too); channels
  with `not detected throughout` produce no events.
- The dataloader uses the JSON event records for model input and verifies that
  every event phrase/span is present in its corresponding standalone part TXT;
  stale or mixed TXT/JSON outputs fail before training.

### 2.3 Generation (MUST be step-synchronized)

The existing corpora were generated with `max_length 512` while training
uses 256 (rules doc §14) — for span-level injection this is fatal, not
cosmetic. Regenerate:

```bash
python generate_signsmd2d.py --dataset How2Sign --splits train test \
  --variant parts --max_length 256
python generate_signsmd2d.py --dataset OpenASL --splits train dev test \
  --variant parts --max_length 256
```

Runtime guard: the dataloader asserts `num_steps == len(pose sequence)`
per clip and hard-fails on mismatch.

## 3. Model: time-aligned part-paired injection (F1)

Implemented in `models_signsmd_paired.py` as
`Uni_Sign_SignSMD_Paired(Uni_Sign)`; the baseline `encode_pose` has one
optional injection hook whose default `None` path is unchanged.

### 3.1 Event phrase embedding

Each event phrase → mT5 tokenizer (its own independent token cap) →
`mt5_model.encoder.embed_tokens` (shared with the LLM and following the LLM's
train/freeze state) → masked mean-pool → 768-d vector. Phrases are short and
categorical; a full encoder pass is deliberately avoided.

### 3.2 Scatter to steps, per part

```text
v_e = W_txt · embed(phrase_e)              W_txt: 768 → 256, shared across parts
text_feat[p][t] = mean of v_e over events e of part p active at step t
                  (zero vector where no event is active)
→ text_feat[p] ∈ (B, T, 256)
```

`text_feat` is built at the *trainer-aligned* step resolution, i.e. it is
index-compatible with the pose token axis by construction (§2.3).

### 3.3 Injection point and gating — mirror PGF

Inside `encode_pose`, per part, **after** the spatial GCN + body-anchor
addition and **before** the temporal GCN (exactly where PGF injects RGB),
broadcast over the joint axis V:

```text
g[p]        = fusion_text_gate(concat([gcn_feat[p], text_feat[p]]))   # per (t)
gcn_feat[p] = gcn_feat[p] + g[p] · text_feat[p]                        # broadcast over V
```

- `signsmd_text_gate`: Conv1d→GELU→Conv1d→Tanh. Only the final Conv1d is
  **zero-initialized** → at init `g ≡ 0` and the model computes exactly the
  baseline pose path. Keeping the first Conv1d live means the zero gate still
  receives gradients; zeroing every layer behind ReLU would deadlock learning.
- Injecting before the temporal GCN lets the temporal convolutions
  propagate event information along time, and keeps the fusion inside the
  part whose text it is — the part pairing is structural, no alignment
  loss needed to enforce it.
- The mT5 encoder input is unchanged: `[translate prompt][T pose tokens]`.
  No sequence-length cost.

### 3.4 Training-time controls

- `--text_event_dropout p` (default 0.3): per sample, drop ALL events with
  probability p (zero `text_feat`) — blocks shortcut reliance, and gives
  the free diagnostic `--eval_mask_text` (evaluate with events zeroed;
  should stay ≥ pose-only baseline).
- Warm start: `strict=False`; `W_txt` + gate convs are new. The final gate
  projection alone is zero-init, which is sufficient for functional identity
  with the baseline at step 0.
- `--signsmd_event_root` selects the events directory; a missing events
  file = no events (clip trains as pure baseline) unless
  `--events_required` is set.

## 4. Experiment Design

Recipe identical to E0/E1 (How2Sign finetune from
`openasl_pose_only_slt.pth`, batch 8 / grad-accum 8 / 20 epochs / AdamW
1e-4).

| # | Config | Question answered |
|---|---|---|
| E0 | pose-only baseline | (done) |
| E1 | early fusion, SMD as flat text | (done, negative — context) |
| F1 | time-aligned part injection (this plan) | does synchronized part-paired fusion help where E1 didn't? |
| F1m | F1 evaluated with `--eval_mask_text` | pose-stream integrity: must be ≈ E0 |

Readout:

- **F1 vs E0** is the primary comparison (F1 does not need the dual-stream
  line to exist).
- **F1 vs E1** attributes any difference to *how* the same compiled
  content enters the model (time-aligned feature injection vs flat token
  concat).
- F1m « E0 ⇒ the gate leaked and degraded the pose path — raise
  `--text_event_dropout`, or lower gate lr.
- Optional add-on after a positive F1: the part-level InfoNCE alignment
  loss (pooled `pool_feat[p]` ↔ pooled active-event vectors of part p) as
  a regularizer — kept OUT of the first run so F1 varies exactly one
  factor vs E0.

Metrics unchanged: BLEU-4 / ROUGE, dev-BLEU-4 selection, BLEURT at final
eval.

## 5. Deliverables

| File | Change |
|---|---|
| `signsmd2d/realization_parts.py` | new — per-part segmentation + four standalone coarse-style text realizations (reuses evidence.py categories/thresholds) |
| `generate_signsmd2d.py` | register `--variant parts`; write `texts/<part>/<clip>.txt`, `events/<clip>.json`, and manifest |
| `datasets_signsmd.py` | load four part texts + events JSON; assert `num_steps` matches pose length; verify TXT/JSON agreement |
| `models_signsmd_paired.py` | `Uni_Sign_SignSMD_Paired`: independent phrase embed, scatter, live zero-gated injection in `encode_pose` |
| `models.py` | additive hook only: `encode_pose` accepts optional per-part `text_feat` dict (default `None` → byte-identical baseline path) |
| `fine_tuning_signsmd.py` | `--fusion paired` selecting the new model; flags `--signsmd_event_root`, `--text_event_dropout`, `--eval_mask_text`, `--events_required` |
| `script/train_openasl_then_h2s_signsmd_paired_parts.sh` | OpenASL pretrain → How2Sign finetune → normal/masked evaluation; stages `pretrain | finetune | eval | all` |
| `script/generate_parts_and_train_stage1.sh` | one-command OpenASL+How2Sign parts generation/preflight followed by the `script/train_stage1.sh` OpenASL Stage-1 recipe with dataset-specific evaluation roots |

## 6. Verification status

Completed component checks:

1. Compiler on one real clip from each dataset: every event span within
   `[0, num_steps)`; text rendering and JSON agree on event counts;
   forbidden-vocabulary check passes. A full-corpus granularity audit is still
   required; report mean intervals-per-part —
   hand parts averaging ≈1 interval on multi-second clips means the
   segmentation collapsed (see §2.1) and thresholds must be revisited
   before any corpus-scale generation.
2. Dataloader: a real 256-step How2Sign sample loaded/collated four TXT files
   and 37 JSON events; a deliberately mismatched `num_steps` is rejected.
3. Zero-init invariant: with the local mT5 weights and real events, paired and
   plain pose embeddings were bit-identical at initialization (`[1,256,768]`).
4. Unit test confirms the final gate convolution receives a nonzero gradient
   while its initial fused output remains exactly the pose input.
5. Unit test confirms overlapping events are averaged only on their matching
   part/span and all unrelated part/steps remain zero.

Still runtime-pending: a CUDA/DeepSpeed forward-backward job and evaluation of
a trained checkpoint with `--eval_mask_text`.

## 7. Fallback / extensions (not in the first run)

- **Token-level interleave** `[part pose tokens][part text tokens]`:
  rejected as primary (cannot exploit step sync, ~4× encoder length, no
  warm start). Only revisit if F1 is positive and the bottleneck is
  provably the 256-d injection capacity.
- **Dual-stream combination**: F1 (fine-grained, inside pose encoder) and
  the dual-stream SMD segment (global summary, decoder-side) are
  orthogonal and can be combined once each has a standalone readout.
- **Phrase encoder upgrade**: replace embed_tokens mean-pool with a full
  mT5 encoder pass over phrases if the phrase vocabulary proves too
  coarse.
