# Experiment: SignSMD Dual-Stream Fusion

Self-contained experiment. No dependency on the pretext-pretraining line
(`signsmd_pretext_plan.md`) — everything here warm-starts from ordinary
baseline checkpoints and changes **only the fusion architecture**.

## 1. Motivation and Hypothesis

Early fusion (`models_signsmd.Uni_Sign_SignSMD`) concatenates SignSMD text
tokens with pose embeddings at the mT5 encoder input:

\[
H = E_\theta([S;P]),
\]

and gave a negative result on How2Sign. Pose and SMD describe the same
signing at very different abstraction levels (continuous geometry vs
discrete compiled semantics), so letting them interact from the first
self-attention layer may distort the formation of the pose representation —
the modality the baseline already translates well from.

**Hypothesis (H-dual):** the early-fusion negative result is caused by the
fusion architecture, not by the SMD content. Encoding pose and SMD in
separate streams with the *same* encoder weights and fusing only at the
representation level,

\[
Fuse(E_\theta(P), E_\theta(S)),
\]

protects the pose representation and recovers (or exceeds) the pose-only
baseline while still letting the decoder exploit SMD.

This experiment varies exactly one factor — fusion architecture — with the
training objective, data, and parameter count held fixed.

## 2. Method

### 2.1 Shared-weight dual-stream encoding

- **Pose stream = exactly the baseline input**: `[translate prompt][pose
  embeddings]`. The pretrained prompt–pose adjacency is preserved, so any
  baseline mT5 Uni-Sign checkpoint warm-starts the primary stream unchanged
  (`strict=True`).
- **SMD stream**: `"Motion description: "` (plain vocabulary words — no new
  special tokens) + SMD text tokens.
- Both streams go through the **same T5 encoder parameters** (two forward
  passes).

```text
Pose ──[translate prompt]──→ T5 Encoder ──→ H_pose ──┐
                                ↑ shared θ           │
SMD ──["Motion description:"]→ T5 Encoder ──→ H_SMD ─┤
                                                     ▼
                                                   Fuse
                                                     ▼
                                                T5 Decoder ──→ Translation
```

Trade-off, stated honestly: all cross-modal interaction moves to decoder
cross-attention (T5 cross-attention has no positional bias over the encoder
axis, so concatenating two independently encoded segments is well-formed).
Cost note: encoder attention FLOPs actually drop
(\(L_P^2+L_S^2 < (L_P+L_S)^2\)); the overhead is one extra encoder launch.

### 2.2 Fusion variant ladder

1. **Concat** (core variant): \(H_{\text{fusion}} = [H_P; H_S]\).
   Zero new parameters — `strict=True` checkpoint round-trip with the
   baseline in both directions.
2. **Fixed weight**: \(H_{\text{fusion}} = [H_P; \alpha H_S]\),
   \(\alpha=0.1\) first — starts the model near the pose-only baseline.
   Still zero parameters (α is a hyperparameter).
3. **Learnable gate**: global \(\alpha=\sigma(a)\), or sample-dependent
   \(g_i = \sigma(W[\operatorname{Pool}(H_P^{(i)});\operatorname{Pool}(H_S^{(i)})])\)
   — useful because SMD quality varies per clip. Adds parameters → loses
   `strict=True`; warm-start with `strict=False`, init \(\sigma(a)\approx0.1\).
   Run only after the zero-parameter variants read out.

### 2.3 SMD modality dropout

To block the shortcut SMD→Translation, drop the SMD stream during training
with probability ~0.3 — implemented by **zeroing the SMD segment's
attention mask** (not by feeding zero vectors). The same mechanism gives a
free diagnostic: `--eval_mask_smd` evaluates any trained dual model with
the SMD segment masked, measuring whether the pose stream degraded relative
to the pose-only baseline. Hierarchy preserved: pose = primary, SMD =
auxiliary.

## 3. Experiment Design

**The control run is `script/finetune_h2s_signsmd_from_openasl.sh`** (early
fusion, How2Sign finetune warm-started from the OpenASL pose-only
checkpoint `openasl_pose_only_slt.pth`, output
`out/signsmd_fusion_h2s_from_openasl`). E3 replicates that run exactly —
same warm-start checkpoint, same `--signsmd_text_root`, same
batch 8 / grad-accum 8 / 20 epochs / AdamW lr 1e-4 — changing **only**
`--fusion dual`, so any metric difference is attributable to the fusion
architecture alone.

| # | Config | Varies vs control | Run |
|---|---|---|---|
| E0 | pose-only baseline (same OpenASL warm start) | no SMD at all | done |
| E1 | early fusion | — (this IS the control) | done — `finetune_h2s_signsmd_from_openasl.sh` (negative result) |
| E3 | dual-stream concat | `--fusion dual` only | **primary run** |
| E5 | dual + fixed α=0.1 | + `--fusion_alpha 0.1` | after E3 |
| E6 | dual + sample gate + SMD dropout 0.3 | + gate/dropout | after E3/E5 |

Readout:

- **E3 vs E1**: the direct test of H-dual (same data, same objective, same
  parameter count, same schedule — only the fusion architecture differs).
- **E3 vs E0**: does dual-stream at least stop hurting relative to
  pose-only? If E3 ≥ E0 while E1 < E0, the architecture explanation holds
  even if SMD adds nothing yet.
- **`--eval_mask_smd` on E3/E5/E6 vs E0**: pose-stream integrity; a large
  unmasked-vs-masked gap indicates genuine SMD contribution rather than
  shortcut reliance.

Metrics: BLEU-4 / ROUGE (BLEURT at final `--eval`), baseline dev-BLEU-4
checkpoint selection unchanged.

(Optional future coupling, not part of this experiment: warm-starting the
same code from a pretext-pretrained checkpoint — see
`signsmd_pretext_plan.md` — would fill the remaining cell of the 2×2.)

## 4. Implementation

Follows the codebase's additive-subclass convention; baseline files
untouched; no tokenizer changes.

### D1. `models_signsmd_dual.py` (new): `Uni_Sign_SignSMD_Dual(Uni_Sign)`

`forward`:
1. pose stream: baseline `[translate prompt][pose]` embeds →
   `self.mt5_model.encoder(inputs_embeds=..., attention_mask=...)`;
2. SMD stream: `"Motion description: "` + SMD tokens (tail-truncated at
   `--signsmd_max_tokens`, as in early fusion) → same encoder;
3. fusion: scale `H_S` by `--fusion_alpha` (default 1.0) or gate
   (`--fusion_gate none|global|sample`); training-time `--smd_dropout p`
   zeroes the SMD segment's attention mask per sample;
4. concat hidden states + masks; loss via
   `self.mt5_model(encoder_outputs=(H_fusion,), attention_mask=fused_mask,
   labels=...)` (labels built exactly as in the baseline, max_length=50,
   label smoothing / fp32-loss flags inherited);
5. `stack_out` carries `encoder_outputs` + `attention_mask` (not
   `inputs_embeds`).

`generate` **must be overridden** — the inherited implementation passes
`inputs_embeds`; wrap `H_fusion` in
`transformers.modeling_outputs.BaseModelOutput` and call
`self.mt5_model.generate(encoder_outputs=..., attention_mask=...)`.

Flags: `--fusion_alpha`, `--fusion_gate`, `--smd_dropout`,
`--eval_mask_smd`.

Checkpoint compatibility: `--fusion_gate none` adds zero parameters →
`strict=True` both ways with baseline checkpoints; `global|sample` gates
add `a`/`W` → load warm starts with `strict=False`, init gate ≈ 0.1.

### D2. `fine_tuning_signsmd.py` (modified)

Add `--fusion early|dual` (default `early`, existing behavior unchanged)
selecting `Uni_Sign_SignSMD` vs `Uni_Sign_SignSMD_Dual`; pass through the
D1 flags; `validate_args`: gate/dropout/alpha/mask flags require
`--fusion dual`. Dataset (`S2T_Dataset_SignSMD`), DeepSpeed setup, train
loop, eval loop, and BLEU-4 selection are all reused unchanged.

## 5. Run Script

**`script/finetune_h2s_signsmd_dual_from_openasl.sh`** (new) — a line-for-line
mirror of the control script `script/finetune_h2s_signsmd_from_openasl.sh`,
differing only in `--fusion dual` and output dirs:

```bash
# E3 (matches the control run exactly except --fusion dual)
ckpt_path=/data/hwu/Uni_sign_repo/out/openasl_pose_pretrain/openasl_pose_only_slt.pth
signsmd_root=data/How2Sign/signsmd2d_v0_1/texts

deepspeed --include $gpus --master_port $port fine_tuning_signsmd.py \
   --batch-size 8 \
   --gradient-accumulation-steps 8 \
    --epochs 20 \
    --opt AdamW \
    --lr 1e-4 \
    --fusion dual \
    --output_dir out/signsmd_dual_h2s_from_openasl \
    --finetune $ckpt_path \
    --dataset How2Sign \
    --task SLT \
    --dtype fp32 \
    --signsmd_text_root $signsmd_root

# eval stage: same as control's eval block + --fusion dual
# extra diagnostic eval: add --eval_mask_smd

# E5: append --fusion_alpha 0.1        (output_dir out/signsmd_dual_alpha01_...)
# E6: append --fusion_gate sample --smd_dropout 0.3
```

The script keeps the control's `finetune | eval | smoke | all` stage
interface; its `smoke` stage evaluates the OpenASL checkpoint with
`--fusion dual --signsmd_on_missing empty`, which (like the control's
smoke) should land ≈ the plain baseline eval — a masked/empty SMD stream
reduces the dual model to the baseline computation.

## 6. Smoke Tests (before any GPU time; user runs all Python)

1. Baseline checkpoint loads into `--fusion dual --fusion_gate none` with
   `strict=True`; the resulting checkpoint loads back into baseline
   `models.Uni_Sign` the same way (zero-new-params invariant).
2. One forward+backward on a 2-sample batch; check fused hidden/mask shapes
   (`[B, L_prompt+L_pose+L_smd, 768]`).
3. One `--eval` pass on a few batches: `encoder_outputs`-based generation
   returns sensible text; `--eval_mask_smd` runs and changes the output.
4. Gate variants: `strict=False` warm start reports only gate params as
   missing; \(\sigma(a) \approx 0.1\) at init.
5. `"Motion description: "` tokenizes to short mT5 pieces (no `<unk>`).

## 7. Deliverables

| File | Status |
|---|---|
| `models_signsmd_dual.py` | new |
| `fine_tuning_signsmd.py` | modified (`--fusion early|dual` + flags) |
| `script/finetune_h2s_signsmd_dual_from_openasl.sh` | new (mirror of the control script) |
