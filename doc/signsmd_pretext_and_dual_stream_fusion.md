# SignSMD as an Auxiliary Modality for Sign Language Translation

## 1. Motivation

The goal is to use **SignSMD (SMD)** as an auxiliary semantic modality for sign language translation (SLT), rather than simply concatenating SMD tokens with pose tokens at the input of T5.

The previous early-fusion formulation was:

\[
[\text{SMD};\text{Pose}]
\rightarrow
\text{T5 Encoder}
\rightarrow
\text{T5 Decoder}
\rightarrow
\text{Translation}.
\]

This formulation allows pose and SMD tokens to interact from the first self-attention layer. However, pose and SMD represent the same signing sequence at very different abstraction levels:

- **Pose**: continuous, fine-grained geometric and temporal information.
- **SMD**: discrete, structured, semantic description of motion.

Naive early fusion may therefore introduce negative interference, especially when SMD is noisy, incomplete, or overly compressed.

A better direction is to first explicitly teach the model the relationship between the two representations, and then fuse them at a later stage.

### 1.0 SMD is a derived representation, not an independent modality

One framing point shapes everything below. SignSMD-2D is compiled **deterministically from the same RTMPose keypoints** that form the pose input (`signsmd2d/compiler.py` even resamples with exactly the trainer's frame rule). Three consequences:

1. **The vision-language-pretraining analogy is only partial.** Image–caption pretraining aligns two *independently sourced* signals. Pose→SMD here teaches the encoder to reproduce a known function of its own input. That is still valuable — it distills the compiler's abstraction hierarchy (handshape, location, movement, symmetry) into the encoder as explicit supervision — but it cannot inject information the pose does not already contain. The correct framing is **structured self-supervision / distillation of a programmatic motion abstraction**, not cross-modal grounding.

2. **Randomly sampled matching negatives are nearly free to reject.** Compiled texts carry statistical fingerprints — step counts correlated with clip length, `[step a-b]` span structure aligned to the pose timeline — so a model can call "no" from surface cues without reading motion semantics. Hard negatives are therefore **not optional** (see §1.4).

3. **The compiler is an asset for supervision.** Because SMD is generated programmatically from evidence, we can produce *exact counterfactuals* by perturbing the evidence before realization — guaranteed-wrong, minimally different, surface-statistics-matched negatives with controllable difficulty. Caption-based multimodal work cannot do this; it should be an explicit contribution (§1.4).

---

# Idea 1: SMD and Pose-SMD Matching as Pretext Tasks

## 1.1 Core Idea

Before directly optimizing:

\[
(\text{Pose},\text{SMD})
\rightarrow
\text{Translation},
\]

use auxiliary pretext tasks to teach T5:

\[
\boxed{
\text{Pose}
\leftrightarrow
\text{SMD}
}
\]

The intuition: before expecting the model to benefit from the SMD input, explicitly train it to understand how continuous pose corresponds to its structured description. Per §1.0 this is structured self-supervision — the encoder is forced to make the compiler's abstraction level linearly available in its representations, rather than hoping SLT supervision induces it implicitly.

## 1.2 Pretext Task A: Pose-to-SMD Generation

Given a pose sequence, generate its SignSMD description.

\[
\boxed{
\text{Pose}
\rightarrow
\text{SMD}
}
\]

Example:

```text
Input:
Describe the signing motion: [POSE SEQUENCE]

Target:
The right hand is located near the chest and moves upward while
the left hand remains relatively stationary.
```

This forces the T5 encoder to extract motion-semantic concepts (handshape, hand location, palm orientation, movement direction, relative hand configuration, temporal structure, symmetry, hand interaction) instead of learning only the final translation target.

Conceptually: \(P \rightarrow H_P \rightarrow D_{\text{SMD}}\), encouraging \(H_P\) to contain structured motion semantics.

**Target-length control (required, not cosmetic).** The SLT path caps translation labels at 50 mT5 tokens, while full SMD texts run up to 512. With per-token cross-entropy, a modest share of describe batches would dominate the total gradient signal by sheer token count. Concretely:

- cap describe targets with `--describe_max_target_tokens` (start at 128), preferring compile-time compaction (the `coarse` variant) over hard truncation;
- optionally use **span-infilling** (T5-native): put most of the SMD in the input and generate only a masked interval/sentence, which shortens targets while keeping full-text conditioning;
- log per-task loss separately so the effective task weighting is visible (see §1.5).

## 1.3 Pretext Task B: Pose-SMD Matching

The second pretext task explicitly teaches pose–description alignment.

Input: \((\text{Pose},\text{SMD})\). Output: \(\{\texttt{yes},\texttt{no}\}\).

Positive example:

```text
Input:
Does the following description match the signing? The right hand
moves upward near the face. [POSE SEQUENCE]

Target:
yes
```

Negative example:

```text
Input:
Does the following description match the signing? Both hands move
downward from the chest. [POSE SEQUENCE]

Target:
no
```

Design constraints:

- **Balance positives/negatives 50/50.** The task contributes roughly one supervised token per sample; an unbalanced prior makes the tiny loss signal degenerate.
- **Random negatives (descriptions of other clips) are a weak baseline only.** Because compiled texts encode step counts and span structure tied to clip length (§1.0), other-clip negatives are frequently detectable from surface statistics. When used at all, sample them length-matched (similar step count). The version of this task worth running uses the counterfactual negatives of §1.4.

Thus:

\[
\boxed{(\text{Pose},\text{SMD}^{+}) \rightarrow \texttt{yes}}
\qquad
\boxed{(\text{Pose},\text{SMD}^{-}) \rightarrow \texttt{no}}
\]

## 1.4 Compiler-Generated Counterfactual Hard Negatives (core)

Because SMD is realized from an intermediate evidence representation
(`signsmd2d`: `compute_evidence` → `segment` → `realize`), hard negatives
should be produced by **perturbing the evidence and re-realizing**, not by
editing text. Perturbation operators:

| Operator | Evidence perturbation | Text effect |
|---|---|---|
| `swap_hands` | exchange left/right evidence subtrees | hand identity flips |
| `flip_vertical` | negate y-component of wrist trajectories | upward ↔ downward |
| `mirror_horizontal` | mirror x about the body midline | left/right side of space flips |
| `shift_location` | offset wrist positions by one location zone | chest ↔ face, etc. |
| `alter_handshape` | permute finger-curl channels | fist ↔ open flat, etc. |

Correct SMD:

```text
right hand moves upward near the head
```

Counterfactual negatives:

```text
right hand moves downward near the head     (flip_vertical)
left hand moves upward near the head        (swap_hands)
right hand moves upward near the chest      (shift_location)
```

Properties random negatives lack:

- **guaranteed wrong** — the perturbation is applied to the evidence, so the realized text describes motion that verifiably did not happen;
- **surface-statistics matched** — same header, same step spans, same segment count, so the only usable signal is motion semantics;
- **attribute-controllable** — the perturbed attribute is recorded in a manifest, enabling per-attribute matching-accuracy diagnostics (which attributes does the model actually read?);
- passes the same `assert_clean` vocabulary gate, plus an assertion that the realized negative differs from the positive text.

The model must distinguish hand identity, location, direction, and handshape — exactly the attributes SLT needs.

## 1.5 Multi-Task Pretraining

The multimodal pretraining stage combines three tasks:

**Task 1 — Translation.** `Translate sign language video to English: ` + Pose (+ SMD) → Translation.

**Task 2 — SMD Generation.** `Describe the signing motion: ` + Pose → SMD.

**Task 3 — Pose-SMD Matching.** `Does the following description match the signing? ` + SMD + Pose → yes/no.

All tasks share the **same T5 encoder, decoder, and LM head**.

**Task prefixes are plain vocabulary words, not new special tokens.** Adding `<describe>`/`<match>` special tokens would resize the embedding table and break the zero-new-parameters property this path relies on (baseline checkpoints load with `strict=True`, and pretrained checkpoints load back into the baseline). Plain-word prefixes achieve the same routing with no vocabulary change. (Verify once that the mT5 tokenizer maps the target words "yes"/"no" to short, stable pieces.)

```text
                         Pretrained T5
                              │
                              ▼
                  Multimodal Pretraining
                 /            |            \
                ▼             ▼             ▼
        Pose → Translation  Pose → SMD  SMD + Pose → yes/no
                \             |             /
                 └── Shared T5 Parameters ──┘
```

The unified objective is standard autoregressive generation:

\[
\mathcal{L}
=
-\sum_t
\log
p_\theta(y_t\mid y_{<t},X,\text{task}).
\]

Task importance is controlled through **batch-level sampling ratios** (each batch carries one task), e.g.

\[
p_{\text{SLT}} = 0.70,
\qquad
p_{\text{SMD}} = 0.20,
\qquad
p_{\text{Match}} = 0.10,
\]

treated as hyperparameters. Two caveats the ratios alone do not solve:

1. **Token-count imbalance.** Sampling ratios control *batch* frequency, not *label-token* frequency. A describe batch can contain 5–10× the label tokens of an SLT batch (§1.2), silently overweighting Task 2. Mitigate with the describe target cap plus optional per-task loss scales (`--task_loss_scales`), and always log per-task loss meters.
2. **Matching share.** At \(p=0.10\) with ~1 supervised token per sample, Task 3 may need a larger share or an explicit loss scale; treat this as a first-class hyperparameter, monitored via held-out matching accuracy.

## 1.6 Training Pipeline

```text
T5 language-pretrained weights (+ baseline Uni-Sign SLT checkpoint)
             │
             ▼
   Pose/SMD multimodal pretraining on OpenASL
      ├── Pose (+ SMD) → Translation      (retained, dominant share)
      ├── Pose → SMD
      └── SMD + Pose → yes/no  (counterfactual negatives)
             │
             ▼
      SLT-focused finetuning on How2Sign
             │
             ▼
     Pose + SMD → Translation
```

Two commitments the earlier draft left open:

- **Stage 1 retains SLT batches** (the 0.70 share above). Dropping translation during pretraining risks drifting the capability Stage 2 warm-starts from; retention also lets Stage 1 keep the existing dev-BLEU-4 checkpoint selection unchanged.
- **Pretraining runs on OpenASL, finetuning on How2Sign**, matching the established pipeline convention. Note the matching-negative statistics (clip length, segment counts) differ between corpora; the counterfactual construction of §1.4 makes this mostly moot because negatives are always derived from the same clip.

The main hypothesis:

> Before SMD can function effectively as an auxiliary input, the model should explicitly learn the correspondence between continuous pose and its structured semantic description.

---

# Idea 2: Shared-Weight Dual-Stream Representation-Level Fusion

## 2.1 Motivation

The current model uses **single-stream early fusion** (`models_signsmd.Uni_Sign_SignSMD`):

```text
SMD ── mT5 embedding ──┐
                       ├── Concat ── T5 Encoder ── T5 Decoder ── Translation
Pose ── ST-GCN ────────┘
```

Mathematically: \(H = E_\theta([S;P])\). Pose and SMD interact from the first self-attention layer.

Instead, use a **dual-stream representation-level fusion** strategy.

## 2.2 Shared-Weight T5 Encoding

Pose and SMD are encoded independently with the **same T5 encoder parameters**:

\[
H_{\text{pose}} = E_\theta(P_{\text{stream}}),
\qquad
H_{\text{SMD}} = E_\theta(S_{\text{stream}}).
\]

**Stream inputs (previously unspecified, now fixed):**

- **Pose stream = exactly the baseline input**: `[translate prompt][pose embeddings]`. This preserves the pretrained prompt–pose adjacency, so any baseline mT5 Uni-Sign checkpoint warm-starts the primary stream unchanged (`strict=True`).
- **SMD stream**: `"Motion description: "` (plain words) + SMD tokens. The tag tells the shared encoder which mode it is in, at zero new parameters.

```text
Pose ──[translate prompt]──→ T5 Encoder ──→ H_pose ──┐
                                ↑ shared θ           │
SMD ──["Motion description:"]→ T5 Encoder ──→ H_SMD ─┤
                                                     ▼
                                                   Fuse
                                                     ▼
                                                T5 Decoder
                                                     ▼
                                               Translation
```

The distinction:

\[
\text{early fusion: } E_\theta([S;P])
\qquad\text{vs.}\qquad
\text{representation-level fusion: } Fuse(E_\theta(P), E_\theta(S)).
\]

Cost note: dual-stream is actually *cheaper* in encoder attention FLOPs
(\(L_P^2 + L_S^2 < (L_P+L_S)^2\)); the overhead is just a second encoder
launch.

## 2.3 Simple Concatenation Fusion

\[
H_{\text{fusion}} = [H_{\text{pose}};H_{\text{SMD}}],
\qquad
Y = D_\theta(H_{\text{fusion}}).
\]

This prevents SMD from altering the formation of the pose representation inside the encoder. **The trade-off must be stated symmetrically**: it also prevents the SMD encoding from being *pose-conditioned* — all cross-modal interaction is deferred to decoder cross-attention, a real capacity reduction relative to early fusion. This is exactly the interaction the matching pretext (Idea 1) trains into the shared weights, which is the main argument for the combined configuration (§3) over Idea 2 alone. (T5 decoder cross-attention uses no relative position bias across the encoder axis, so concatenating two independently encoded segments is well-formed.)

This variant adds **zero parameters**: baseline checkpoints load with `strict=True` and vice versa.

## 2.4 Weighted SMD Fusion

Because SMD is auxiliary, introduce a scalar weight:

\[
H_{\text{fusion}} = [H_{\text{pose}}; \alpha H_{\text{SMD}}],
\qquad
\alpha = 0.1 \text{ as a first experiment}.
\]

The model then begins close to the pose-only baseline. A fixed \(\alpha\) is a buffer hyperparameter, not a parameter — `strict=True` compatibility is preserved.

## 2.5 Learnable Gate

Global gate: \(\alpha = \sigma(a)\). Sample-dependent gate:

\[
g_i = \sigma\left(W[\operatorname{Pool}(H_{\text{pose}}^{(i)});\operatorname{Pool}(H_{\text{SMD}}^{(i)})]\right),
\qquad
H_{\text{fusion}}^{(i)} = [H_{\text{pose}}^{(i)}; g_i H_{\text{SMD}}^{(i)}].
\]

Useful because generated-SMD quality varies across samples. **Compatibility cost:** \(a\) and \(W\) are new parameters — these variants lose `strict=True` loading in both directions and need a non-strict warm-start path (baseline weights + explicitly initialized gate, \(a\) initialized so \(\sigma(a)\approx 0.1\)). Run them only after the zero-parameter variants establish the main effect (§5).

## 2.6 SMD Modality Dropout

To prevent the shortcut \(\text{SMD} \rightarrow \text{Translation}\) while ignoring pose, randomly remove the SMD stream during training:

\[
H_{\text{fusion}}
=
\begin{cases}
[H_P; gH_S], & 70\%\\
[H_P; \varnothing], & 30\%.
\end{cases}
\]

**Implementation: mask, don't zero.** Drop the SMD segment by zeroing its *attention mask* (decoder cross-attention then ignores it), not by feeding zero vectors — equivalent in effect, cleaner, and identical to how SMD-free inference works.

Bonus diagnostic for free: the same model evaluated with the SMD segment masked gives a direct measurement of whether the pose stream degraded relative to the pose-only baseline. Add this eval mode (`--eval_mask_smd`) to the ablation.

This preserves the hierarchy: **pose = primary modality, SMD = auxiliary**.

---

# 3. Combining the Two Ideas

## Stage 1 — Multimodal Pretraining (OpenASL)

Teach \(Pose \rightarrow SMD\) and \((SMD, Pose) \rightarrow \text{yes/no}\) with counterfactual negatives, retaining SLT batches at dominant share (§1.5–1.6).

## Stage 2 — Dual-Stream SLT (How2Sign)

\[
H_P=E_\theta(P),\quad
H_S=E_\theta(S),\quad
H_{\text{fusion}}=Fuse(H_P,H_S),\quad
Translation=D_\theta(H_{\text{fusion}}).
\]

```text
                 ┌──────────── PRETEXT TASKS ────────────┐
                 │  Pose → SMD                           │
                 │  SMD + Pose → yes/no (counterfactual) │
                 │  Pose (+SMD) → Translation (retained) │
                 └────────────────┬──────────────────────┘
                                  ▼
                         Multimodally-trained T5
                                  ▼
Pose ──[prompt]──→ T5 Encoder ──→ H_pose ──┐
                      ↑ shared θ           │
SMD ──[tag]──────→ T5 Encoder ──→ H_SMD ───┤
                                           ▼
                                         Fuse ──→ T5 Decoder ──→ Translation
```

The combination is more than additive: the matching pretext compensates for exactly the cross-modal interaction that dual-stream encoding removes (§2.3).

---

# 4. Main Research Hypothesis

> Structured motion descriptions compiled from pose are a useful auxiliary input for sign language translation, but directly concatenating low-level pose embeddings and high-level semantic descriptions may cause negative cross-modal interference. Distilling the compiler's motion abstraction into the encoder via generation and counterfactual-matching pretext tasks first establishes pose–description correspondence, while shared-weight dual-stream encoding preserves modality-specific representations before adaptive fusion.

Two complementary contributions:

1. **Pose-SMD structured self-supervision**
   - Pose-to-SMD generation (distillation of a programmatic abstraction).
   - Pose-SMD matching with **compiler-generated counterfactual negatives** — guaranteed-wrong, surface-matched, attribute-controllable.

2. **Shared-weight dual-stream fusion**
   - Same T5 encoder parameters for both streams; zero new parameters in the core variants.
   - Pose remains the primary modality (fusion weight, modality dropout).
   - SMD introduced only at the representation level.

---

# 5. Ablation Plan

| # | Method | Pretraining | Fusion | New params | Priority |
|---|---|---|---|---|---|
| E0 | Pose-only | No | None | — | done (baseline) |
| E1 | Pose + SMD | No | Early concat | 0 | done (negative result) |
| E2 | Pose + SMD | Yes | Early concat | 0 | **1** |
| E3 | Pose + SMD | No | Dual-stream concat | 0 | **1** |
| E4 | Pose + SMD | Yes | Dual-stream concat | 0 | **1** |
| E5 | + fixed α=0.1 | Yes | Weighted dual-stream | 0 | 2 |
| E6 | + gate + dropout | Yes | Gated dual-stream | gate only | 3 |

The most informative comparison is **E1 vs E2 vs E4**, which separates the two candidate causes of the E1 negative result: (1) missing pose-SMD alignment training, (2) inappropriate fusion architecture. E3 completes the 2×2. Each pretraining run is expensive — do not start E5/E6 until E2–E4 establish the main effect.

Diagnostics reported alongside BLEU/ROUGE:

- **matching accuracy** on held-out clips, broken down by perturbed attribute (§1.4 manifest) — measures what the encoder actually reads;
- **SMD-masked eval** of dual-stream models (§2.6) — measures pose-stream degradation vs E0;
- per-task training loss curves from Stage 1.

---

# 6. Implementation Plans

The two ideas are implemented as **two independent plans**, each with its own
phases, experiments, smoke tests, and deliverables:

- **Plan 1 — SignSMD pretext pretraining**: `signsmd_pretext_plan.md`
  (counterfactual negatives in the compiler, pretext dataset, multi-task
  model, `pretraining_signsmd.py`; evaluated standalone via **E2** =
  pretrain → early fusion).
- **Plan 2 — dual-stream fusion**: `signsmd_dual_stream_fusion_plan.md`
  (`models_signsmd_dual.py`, `--fusion early|dual` flag, fusion variants,
  SMD dropout, `--eval_mask_smd` diagnostic; evaluated standalone via
  **E3** = baseline warm-start → dual-stream).

Dependency structure:

```text
Plan 1 (pretext)            Plan 2 (dual-stream)
     │                            │
     ▼                            ▼
    E2                           E3          ← each plan readable alone
     └──────────┬─────────────────┘
                ▼
               E4   = Plan 2's code warm-started from Plan 1's checkpoint
                      (out/pretext_openasl/best_checkpoint.pth)
```

The plans share no files: Plan 1 delivers a checkpoint, Plan 2 delivers an
architecture, and E4 (§5) is the only coupling point. Either plan can be
built, smoke-tested, and run to its standalone result while the other is
still unstarted.
