# How2Sign Trainable VAE Encoder Integration

## Scope

This is an **additive** sign-to-text path for Uni-Sign. It replaces the original
2D-keypoint/ST-GCN encoder with a pretrained, trainable MoMADiff KL-VAE encoder
for canonical How2Sign 133D motion.

The original implementation remains the default and is unchanged:

- `models.py`
- `datasets.py`
- `fine_tuning.py`
- `config.py`
- all existing `script/*.sh` files

The new path does not introduce LoRA, a LLaVA-style alignment stage, cached
latents, VAE reconstruction, or a VAE decoder. It preserves the original
Uni-Sign mT5 prompt, seq2seq cross-entropy training, DeepSpeed infrastructure,
beam generation, and validation BLEU-4 checkpoint selection.

## Runtime flow

```text
unnormalized 3D sign motion [B,T,133]
  -> normalize with the trained VAE's mean/std
  -> head-truncate each sample to a multiple of 4
  -> trainable KL-VAE encoder + quant_conv
  -> posterior moments [B,T/4,1024]
  -> posterior mean [B,T/4,512]
  -> one Linear(512, mT5 hidden=768), replacing the old pose_proj
  -> motion embeddings [B,T/4,768]
  -> prepend the ordinary text prompt embeddings [B,P,768]
  -> mT5 inputs_embeds [B,P+T/4,768]
  -> token-level cross-entropy against the English reference
```

The translation gradient flows through mT5, the replacement `pose_proj`,
`quant_conv`, and the VAE encoder. The VAE decoder is neither loaded nor used.
The posterior log-variance half is retained for checkpoint compatibility but
the sign-to-text forward uses the deterministic posterior mean.

## Files

| File | Purpose |
|---|---|
| `vae_encoder.py` | Trainable encode-only MoMADiff KL-VAE port and checkpoint loader |
| `datasets_vae.py` | Canonical `[T,133]` How2Sign motion/reference dataset |
| `models_vae.py` | Independent `Uni_Sign_VAE` mT5 model |
| `fine_tuning_vae.py` | Independent train/eval entrypoint |
| `configs/how2sign_vae_encoder.cfg` | Dataset, VAE, mT5, train, and generation settings |
| `tests/test_vae_encoder_path.py` | Shape, split, mask, and gradient tests |

## Data and checkpoint contract

The configured processed root is:

```text
/home/student/hwu/Workplace/MotionGPT3/datasets/how2sign
```

Required layout:

```text
new_joint_vecs/{id}.npy  # unnormalized [T,133]
texts/{id}.txt
train.txt
val.txt
test.txt
Mean.npy
Std.npy
```

The current config intentionally uses the normalization statistics persisted by
the trained VAE:

```text
/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/kl_vae_h2s/meta/mean.npy
/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/kl_vae_h2s/meta/std.npy
```

VAE initialization checkpoint:

```text
/home/student/hwu/Workplace/MoMADiff/checkpoints/how2sign/kl_vae_h2s/lightning_ckpt/min-MPVPE_PA_hand-epoch=109.ckpt
```

All paths are configuration fields. They are not hardcoded in dataset or model
code.

Verified local split contract:

| Split | Samples | Inspected sample shape before unit truncation |
|---|---:|---:|
| train | 30,684 | `[365,133]` |
| val | 1,717 | `[189,133]` |
| test | 2,308 | `[197,133]` |

The loader head-truncates these examples to `[364,133]`, `[188,133]`, and
`[196,133]`, respectively, so the VAE produces an integer number of latent
steps.

## Configuration and commands

The config is JSON syntax stored with a `.cfg` suffix because this repository's
`.gitignore` excludes all `*.json` files. It is parsed with Python's standard
`json` module and adds no dependency.

Training:

```bash
deepspeed fine_tuning_vae.py \
  --config configs/how2sign_vae_encoder.cfg
```

Evaluation:

```bash
deepspeed fine_tuning_vae.py \
  --config configs/how2sign_vae_encoder.cfg \
  --finetune out/how2sign_vae_encoder/best_checkpoint.pth \
  --output_dir out/how2sign_vae_encoder_eval \
  --eval
```

CLI values override defaults from the config's `training` section. Dataset,
VAE, normalization, mT5, prompt, target length, and generation geometry remain
declared in the same config.

## Original-vs-new isolation

The original command continues to instantiate:

```text
datasets.S2T_Dataset -> models.Uni_Sign -> ST-GCN -> mT5
```

Only the new command instantiates:

```text
datasets_vae.How2SignVaeDataset
  -> models_vae.Uni_Sign_VAE
  -> TrainableKlVaeEncoder
  -> mT5
```

No original import is monkey-patched. A VAE-path checkpoint includes VAE encoder
parameters and cannot be loaded into the original ST-GCN class with
`strict=True`; the two checkpoint families are intentionally separate.

## Validation completed

- Python syntax compilation passed for all new Python modules.
- The new modules also parsed successfully with Python 3.8, which is older
  than the Python 3.9 version recommended by Uni-Sign.
- All three focused unit tests passed.
- `git diff --check` passed.
- Actual epoch-109 Lightning checkpoint loaded strictly into the encode-only
  port.
- All loaded VAE encoder parameters had `requires_grad=True` and the module was
  in training mode.
- One real sample from train, val, and test loaded as finite 133D motion.
- A two-sample real batch produced moments `[2,100,1024]` and posterior mean
  `[2,100,512]`.
- A backward pass produced finite gradients in both the first encoder
  convolution and `quant_conv`.
- Valid lengths `[364,400]` generated latent mask lengths `[91,100]`.

## Remaining environment requirement

The current active `soke` environment has PyTorch and Transformers but does not
have `timm`, while Uni-Sign's original `fine_tuning.py` and the new entrypoint
both import `timm.optim.create_optimizer`. Therefore the encoder/data/gradient
path was validated, but a complete mT5+DeepSpeed training smoke was not launched
in that environment.

The configured `./pretrained_weight/mt5-base` directory is also not present in
this checkout. Supply the original Uni-Sign mT5-Base weights there or update
`model.mt5_path` in the config before launching training.
