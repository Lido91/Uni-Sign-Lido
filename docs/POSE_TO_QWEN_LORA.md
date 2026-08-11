# Uni-Sign with Qwen3-4B (LoRA): pose → decoder-only LLM

Companion to [POSE_TO_MT5_PRETRAIN_FINETUNE.md](POSE_TO_MT5_PRETRAIN_FINETUNE.md).
`--llm_type qwen` replaces mT5-Base with **Qwen3-4B-Instruct-2507**
(`/data/hwu/models/qwen3-4b-instruct-2507`, set in `config.py::qwen_path`,
overridable via `--llm_path`). The default `--llm_type mt5` path is unchanged.

## Architecture

The ST-GCN pose encoder is byte-for-byte the same (`models.py::Uni_Sign.encode_pose`).
Differences (`models_qwen.py::Uni_Sign_Qwen`):

```text
pose frames [B, T, 1024]
        ↓ pose_proj: Linear(1024, 2560)   # Qwen hidden size instead of 768
pose embeds [B, T, 2560]
```

Qwen is decoder-only, so there is no encoder to feed. Each sample is packed as
one embedding sequence:

```text
training:   [prompt embeds; pose embeds (valid frames only); target embeds + EOS]
            right-padded, labels = -100 except target tokens, shifted causal CE
            with label smoothing 0.2 (same target truncation: 50 tokens)
generation: [left pad; prompt embeds; pose embeds]  → generate(inputs_embeds=...)
            beam search 4, max_new_tokens 100 (unchanged evaluate() loop)
```

Pose padding frames are dropped at packing time (the mT5 path kept them and
masked them out; here only valid frames enter the sequence).

## What trains

| Module | mT5 path | Qwen path |
|---|---|---|
| ST-GCN + part_para + pose_proj | full | full |
| LLM | full (580M) | frozen base + LoRA (r=16, α=32, dropout 0.05, all attention+MLP projections) |

Because the base is frozen, checkpoints contain **only trainable tensors**
(`get_requires_grad_dict` now filters on `requires_grad`) — a few hundred MB
instead of ~8 GB — and `fine_tuning.py` loads them with `strict=False` when
`--llm_type qwen` (missing keys = frozen base weights, expected).

## Running

Training uses the dedicated **`train_qwen.py`** entry: a plain PyTorch loop
(AdamW over trainable params, bf16 autocast, cosine schedule, optional torchrun
DDP) that runs in the motion-smd **`smd` conda env with no extra installs** —
no deepspeed/timm/decord/sacrebleu needed. For that, the heavy imports in
`utils.py` / `datasets.py` / `models.py` (deepspeed, decord, torchvision,
einops) were made lazy: they are only imported by the original
`fine_tuning.py` path or the `--rgb_support` branches.

```bash
conda activate smd
# Stage 1: OpenASL pose-only pre-training (effective batch 384)
bash script/train_stage1_qwen.sh
# Stage 2: How2Sign fine-tuning from the stage-1 best checkpoint (effective batch 24)
bash script/train_openasl_then_how2sign_qwen.sh
# Single-GPU smoke test
python train_qwen.py --batch-size 2 --epochs 1 --dataset OpenASL --task SLT \
    --dtype bf16 --output_dir out/smoke_qwen --eval-on-how2sign-test
```

New CLI args: `--llm_type {mt5,qwen}`, `--llm_path`, `--lora_rank`,
`--lora_alpha`, `--lora_dropout`, `--llm_grad_ckpt` (gradient checkpointing,
recommended for the 4B model). The scripts use `--dtype bf16`; the base model
is loaded in the training dtype. `fine_tuning.py --llm_type qwen` also still
works if deepspeed/timm are available.

## Requirements / caveats

- The `smd` env covers everything (`transformers >= 4.51` for Qwen3, `peft`,
  `pycocoevalcap`). BLEU-1..4/ROUGE-L are computed with **pycocoevalcap** —
  the same package as the motion-smd How2Sign evaluation — not the vendored
  SLRT sacrebleu, so scores are directly comparable to your motion-smd
  baselines but may differ slightly from Uni-Sign's published sacrebleu
  numbers.
- The prompt is the same plain string as the mT5 path ("Translate sign language
  video to English: ") — no Qwen chat template. Every target ends with
  `<|im_end|>` so the fine-tuned model learns to stop.
- The mT5-era caveats still apply: `--eval-on-how2sign-test` selects the
  pre-training best checkpoint on How2Sign *test*, and How2Sign has no dev
  split, so fine-tuning also model-selects on test. Use a proper validation
  split for paper-grade numbers.
- Old full-model mT5 checkpoints load exactly as before (`strict=True`,
  everything was trainable so nothing is filtered at save time).
