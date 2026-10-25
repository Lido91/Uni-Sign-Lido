# Diagnostic: GT-vs-SMD Contrastive Probe

Standalone text-only diagnostic for the SignSMD line. It answers the
question the fusion experiments keep raising: **does the compiled SMD text
actually contain clip-specific information about the translation, or is it
generic articulation boilerplate?** No pose, no fusion model — just the two
texts of each clip.

```text
GT translation ── encoder ── pool ── head ── z_A ─┐
                    (shared weights, siamese)     ├─ cosine similarity ↑ for
SignSMD text   ── encoder ── pool ── head ── z_B ─┘  matched pairs (InfoNCE)
```

## 1. Method

- **Pairs**: `(GT translation, SMD text)` per clip, straight from the labels
  file + `--signsmd_text_root` (same file contract as the fusion path).
  `--dataset How2Sign,OpenASL` pools both corpora for training (text root
  becomes a `{}` template resolved per dataset); evaluation always stays
  per-dataset — retrieval inside a mixed pool would reward corpus-style
  cues rather than GT↔SMD alignment.
- **Encoder**: the repo's local mT5-base encoder (`config.mt5_path`, same
  tokenizer as the fusion models; offline). `--encoder hf --hf_model
  bert-base-uncased` switches to a HF encoder when downloads are available.
- **Embedding**: masked mean-pool → projection head (`--head mlp`, 256-d) →
  L2 normalize.
- **Loss**: symmetric InfoNCE over in-batch negatives with a learnable
  CLIP-style temperature (init 0.07). Batch size = #negatives + 1.
- **Regimes**: full encoder fine-tune (default; the strongest "is the
  information there at all" probe) or `--freeze_encoder` (linear-probe
  regime). Zero-shot floor: `--eval --head none`, no checkpoint.
- **Selection**: datasets with a dev split (OpenASL) validate on it;
  How2Sign has none, so a deterministic `--val_fraction 0.05` slice of
  train selects `best_checkpoint.pth` by average R@1.

## 2. Readout — "is the SMD correct?"

Retrieval over the **full test split** (not per-batch), chance R@1 = 1/N:

| Result | Interpretation |
|---|---|
| trained R@1 ≈ chance | SMD carries no clip-specific semantic signal — fusion cannot be expected to help |
| R@1 well above chance, low absolute | SMD is informative but noisy/partial |
| high R@1 | compiler preserves translation-relevant content; fusion negatives are an architecture/optimization problem, not a content problem |

Per-clip: `per_sample_scores_test.csv` (worst margin first) ranks every clip
by matched cosine and margin over its hardest in-split negative —
low-margin clips are candidate wrong/generic SMDs to inspect manually, and
the score can later gate per-sample fusion (cf. the sample gate in
`signsmd_dual_stream_fusion_plan.md`).

Caveat: a fine-tuned probe measures *extractable information*, not
human-style semantic similarity — that is exactly what we want for "can the
translator possibly benefit from this text".

## 3. Files

| File | Status |
|---|---|
| `contrastive_gt_smd.py` | new — standalone train/eval (plain PyTorch, 1 GPU) |
| `script/contrastive_gt_smd.sh` | new — `<stage> <How2Sign\|OpenASL>`, stage = `zeroshot \| train \| eval \| smoke \| all` |
| `script/contrastive_gt_smd_h2s.sh` | thin forwarder to the above (How2Sign) |

## 4. Smoke (user runs all Python)

1. `bash script/contrastive_gt_smd.sh smoke` — 128 pairs, 1 epoch:
   checks pairing, forward/backward, checkpoint save/load, CSV/JSON output.
2. `bash script/contrastive_gt_smd.sh zeroshot` — should land ≈ chance.
3. Full run per dataset: `bash script/contrastive_gt_smd.sh all How2Sign`,
   `bash script/contrastive_gt_smd.sh all OpenASL` (OpenASL train is ~3×
   larger — expect a correspondingly longer run), or joint
   `bash script/contrastive_gt_smd.sh all both` (pooled training,
   per-dataset test readout; selection = mean of the two val R@1s).
