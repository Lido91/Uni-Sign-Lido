# Baseline Uni-Sign + SignSMD Text Fusion

Fuses the **baseline** Uni-Sign SLT path (four-part ST-GCN pose encoder + mT5,
`models.Uni_Sign`) with offline-compiled **SignSMD structured sign
descriptions** (deterministic rule-based text derived from the same RTMPose
keypoints; contract `signsmd2d_v0_1`, see
`motion-smd/HOW2SIGN_JOINT_SMD_DESIGN_2D.md`).

Motivation: MMSLT (ICCV 2025, arXiv 2411.16789) showed that textual sign
descriptions and continuous visual features are complementary
(PHOENIX14T BLEU-4: descriptions only 8.99, video only 13.19, fused 25.73).
This path tests the same hypothesis with deterministic SignSMD text instead of
per-frame MLLM prose, and with the description entering the LLM as **real
discrete tokens** rather than a pooled sentence embedding.

## 1. Architecture

```text
src_input (unchanged baseline batch: body/left/right/face_all [B,T,V,3])
        |
        +--> four-part ST-GCN --> Linear(1024,768) --> pose embeddings [B,T,768]
        |
src_input["signsmd_text"] (list[str], loaded by the dataset)
        |
        | mT5 tokenizer, truncation to --signsmd_max_tokens
        v
description token embeddings [B,L_d,768]

mT5 encoder input:
  [description tokens][translate prompt][pose embeddings]
attention mask:
  [description mask   ][prompt mask     ][frame mask     ]
```

Everything downstream (labels, label smoothing, cross-entropy, beam
generation, BLEU-4 selection) is inherited from the baseline unchanged.

Segment order: the description comes FIRST so that (a) the pretrained
adjacency `[translate prompt][pose]` from baseline checkpoints is preserved,
and (b) truncating an over-long description can never cut the prompt.

## 2. Zero new parameters — checkpoint compatibility

`Uni_Sign_SignSMD` adds no parameters over `models.Uni_Sign`: the description
reuses the mT5 embedding table. Consequences:

- any baseline mT5 Uni-Sign checkpoint (stage-3 How2Sign/OpenASL, or the
  released pose-only checkpoints) warm-starts this path with `strict=True`;
- with an empty description the model is numerically close to the baseline
  (only the extra `</s>` token differs), so the fusion starts from baseline
  behavior and must *learn* to use the text;
- fusion checkpoints load back into the plain baseline the same way.

## 3. Data contract

```text
--signsmd_text_root <dir>
    <dir>/<clip-name>.txt        # clip-name = labels video_path minus ".mp4"
```

- Text files are the sanitized compact SignSMD realizations (target-free by
  construction — the compiler must assert no captions/gloss leak into them).
- Missing files: `--signsmd_on_missing error` (default) fails fast;
  `empty` substitutes `""` for smoke tests before full compilation.
- Two step-aligned variants, selected purely by which root you pass
  (see `docs/SIGNSMD_TWO_VARIANT_PLAN.md`):
  - coarse interval prose: `data/<ds>/signsmd2d_v0_2/texts`
  - stats numeric timelines (motion-smd angle_text_v5 style):
    `data/<ds>/signsmd2d_stats_v0_1/texts`
- The old `signsmd2d_v0_1` roots (percent spans, full-frame timeline) are
  superseded; keep them only for comparison.

### Time-axis alignment (steps = pose tokens)

The pose loader defaults to `--frame_sampling linear`: clips longer than
`--max_length` are subsampled by **deterministic linear indexing**
(`np.round(np.linspace(0, duration-1, max_length))`, mirroring the pose2d
KL-VAE path), and shorter clips are kept whole exactly like the baseline.

The compiler (`signsmd2d/compiler.py`) applies **the exact same rule**
before computing any evidence, then renders every span as an inclusive
`[step a–b]`. Step i of the text therefore IS pose token i of the model
input — coverage, reference, and per-step rates all match what the model
sees, by construction. Invariants:

- trainer `--max_length` == compile-time `--max_length` (both default 256,
  recorded in the generation manifest);
- trainer must use `--frame_sampling linear` (the default);
- both sides slice the same optional `[start, end)` pickle window first.

Long-clip evaluation also becomes deterministic as a side effect.
`--frame_sampling unisign_random` restores the baseline's random subsampling
(breaking exact step alignment) for ablations.

## 4. Files

| File | Role |
| --- | --- |
| `datasets_signsmd.py` | baseline description loading plus four-part TXT/JSON event loading for paired fusion |
| `models_signsmd.py` | `Uni_Sign_SignSMD`: baseline model + description segment in the encoder input |
| `models_signsmd_paired.py` | event phrase embedding, part/span scatter, and zero-gated pose-part injection |
| `fine_tuning_signsmd.py` | train/eval entrypoint (mirrors `fine_tuning.py` flow) |
| `script/train_signsmd_fusion.sh` | OpenASL pretrain -> How2Sign finetune -> eval runner |
| `script/train_openasl_then_h2s_signsmd_paired_parts.sh` | paired OpenASL -> How2Sign -> normal/masked evaluation runner |
| `script/generate_parts_and_train_stage1.sh` | generate both parts corpora, validate coverage, then launch paired Stage-1 with `train_stage1.sh` hyperparameters |

The early/dual baseline paths retain their original behavior. `models.py` has
one optional part-text injection hook; its default `None` path is unchanged.

## 5. Usage

```bash
# How2Sign finetune, warm-started from a baseline checkpoint
deepspeed fine_tuning_signsmd.py \
    --dataset How2Sign --task SLT --dtype fp32 \
    --finetune <baseline_checkpoint.pth> \
    --signsmd_text_root data/How2Sign/signsmd2d_v0_1/texts \
    --output_dir out/signsmd_fusion_h2s

# Evaluation
deepspeed fine_tuning_signsmd.py \
    --dataset How2Sign --task SLT --dtype fp32 --eval \
    --finetune out/signsmd_fusion_h2s/best_checkpoint.pth \
    --signsmd_text_root data/How2Sign/signsmd2d_v0_1/texts \
    --output_dir out/eval_signsmd_fusion_h2s
```

New CLI arguments (all others identical to the baseline):

| Argument | Default | Meaning |
| --- | --- | --- |
| `--signsmd_text_root` | required | directory of `<clip>.txt` descriptions |
| `--signsmd_max_tokens` | 512 | hard cap for the description segment (tail-truncated) |
| `--signsmd_on_missing` | `error` | `error` or `empty` on a missing text file |
| `--frame_sampling` | `linear` | `linear`: deterministic subsampling, text spans align exactly with the pose timeline; `unisign_random`: baseline random sampling (ablation) |
| `--eval_signsmd_text_root` | training text root | text root for a different evaluation dataset under `--eval-on-how2sign-test` |
| `--eval_signsmd_event_root` | training event root | paired event root for that different evaluation dataset |

Constraints: `--dataset How2Sign|OpenASL`, `--task SLT`, `--llm_type mt5`,
no `--rgb_support`.

## 6. Ablation matrix (mirrors MMSLT)

| Run | Command delta | Corresponds to |
| --- | --- | --- |
| pose only | plain `fine_tuning.py` | MMSLT "video only" |
| fusion | this path | MMSLT full model |
| fusion, empty text | `--signsmd_on_missing empty` + no texts | sanity: should match pose only |
| text only | future work (mask pose segment) | MMSLT "SLD only" |

## 7. Generating the descriptions (SignSMD-2D compiler)

The compiler is implemented IN THIS REPO as the numpy-only package
`signsmd2d/` plus the bulk CLI `generate_signsmd2d.py` (no torch, no GPU).
It reads the same label pickles and RTMPose pose pickles the trainer uses,
so IDs align with zero coverage gaps. Thresholds are versioned per variant
(`configs/signsmd2d_thresholds.yaml` for coarse,
`configs/signsmd2d_stats_thresholds.yaml` for stats) and hashed into every
manifest.

Pretrain + finetune needs BOTH corpora. Add `--variant stats` for numeric
timelines, or `--variant parts` for four standalone part texts plus JSON event
spans; default is coarse:

```bash
# OpenASL (train/dev/test share one flat pose directory)
python generate_signsmd2d.py --dataset OpenASL --splits train dev test

# How2Sign (per-split pose directories)
python generate_signsmd2d.py --dataset How2Sign --splits train test

# Part-paired contract (must use the trainer's max_length)
python generate_signsmd2d.py --dataset OpenASL --splits train dev test \
  --variant parts --max_length 256
python generate_signsmd2d.py --dataset How2Sign --splits train test \
  --variant parts --max_length 256

# smoke first: a few samples per variant, inspect the output text
python generate_signsmd2d.py --dataset How2Sign --splits test --limit 10
python generate_signsmd2d.py --dataset How2Sign --splits test --limit 10 --variant stats
```

Useful flags: `--limit N`, `--overwrite`, `--num_shards N --shard K`
(parallel generation), `--fail_fast`, `--fps` (default 24, recorded as
"assumed" in the manifest). Resume is the default — existing `.txt` files
are skipped. For `parts`, a clip is skipped only when all four texts and its
JSON exist. Every generated text passes a banned-vocabulary assertion (no
`forward`/`backward`/`contact` depth claims); interval spans are inclusive
trainer-aligned step indices.

## 8. Current status / prerequisites

- Early, dual, and paired fusion code paths are implemented. Real one-clip
  OpenASL/How2Sign parts generation, real paired dataloader collation,
  part/span scatter, zero-gate identity, and gate-gradient component checks
  have passed. A CUDA/DeepSpeed training step remains runtime-pending.
- Recommended paired order: (1) full OpenASL + How2Sign `parts` generation;
  (2) inspect manifest interval granularity; (3) run
  `script/train_openasl_then_h2s_signsmd_paired_parts.sh all` in the activated
  training environment.
- For the original `script/train_stage1.sh` hyperparameter recipe and its
  How2Sign-test checkpoint selection, run the combined
  `script/generate_parts_and_train_stage1.sh all` launcher instead.
- Thresholds are heuristic initial values; calibrate on the train split
  only if needed, then bump the contract version.
