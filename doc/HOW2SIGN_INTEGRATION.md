# How2Sign + YouTube3D VQ Embedding to mT5 Integration

## Scope

This is an additive sign-to-text path for Uni-Sign. It consumes cached SOKE
single-stage, three-branch VQ codebook embeddings instead of RTMPose keypoints
or online VAE latents. The original pose, Qwen, and trainable KL-VAE paths are
unchanged and use separate checkpoint families.

The VQ entrypoint does not import or instantiate the original ST-GCN model or
training loop. Projection is entirely frame-wise MLP computation; temporal
context is modeled by the mT5 encoder after projection.

## Data and provenance contract

Configured embedding root:

```text
/home/student/hwu/Workplace/SOKE/data/How2Sign/EMBEDDINGS_h2s_youtube3d
```

Per-sample files:

```text
{how2sign|youtube3d}/{sample}.npy
shape: [1,T',3,512]
dtype: floating point, normally float32
part order: body/rest, left hand, right hand
```

These are quantized codebook lookup vectors, not pre-quantization encoder
features, decoded 133D motion, or multi-stage RVQ embeddings. The loader checks
the corresponding SOKE token manifest:

```text
/home/student/hwu/Workplace/SOKE/data/How2Sign/TOKENS_h2s_youtube3d/_manifest.json
```

The actual filename is `_manifest.json`, not `manifest.json`. Its checkpoint
fingerprint must match the configured VQ-VAE checkpoint. The manifest must also
declare train/test coverage and codebook sizes `(96,192,192)`. The embedding
`split_index.json` controls split membership, preventing train/test mixing even
though each source stores both splits in one directory.

Annotations remain external configuration fields:

- How2Sign: `{root}/{split}/re_aligned/how2sign_realigned_{split}_preprocessed_fps.csv`
- YouTube3D: `{root}/{split}/re_aligned/youtube_asl_{split}.csv`

## Model path

```text
cached embedding [B,T',3,512]
  -> separate body/left/right Linear(512,256) + GELU
  -> concatenate the three branches [B,T',768]
  -> fusion Linear(768,mT5 hidden=768)
  -> prepend "Translate sign language video to English: "
  -> mT5 encoder inputs_embeds
  -> mT5 decoder text cross-entropy
```

Padding is zero-filled and excluded by a VQ-step attention mask. The three
branches are not summed and their codebook IDs are not passed through the text
tokenizer. This path does not load the VQ encoder or decoder during training;
the cache and its manifest provide the tokenizer boundary.

## Configuration and commands

One config drives both training and evaluation:

```text
configs/how2sign_youtube3d_vq_embeddings.cfg
```

Default training uses How2Sign + YouTube3D and evaluates on How2Sign test. The
output directory is:

```text
/data/hwu/Uni_sign_repo/out/t5_h2s_youtube3d_vq_embeddings
```

For staged training, two additional configs keep source selection out of Bash:

| Stage | Train/eval source | Config | Output |
|---|---|---|---|
| Pretrain | YouTube3D train+test / How2Sign test | `configs/youtube3d_vq_embeddings_pretrain.cfg` | `/data/hwu/Uni_sign_repo/out/t5_youtube3d_vq_pretrain` |
| Finetune | How2Sign / How2Sign | `configs/how2sign_vq_embeddings_finetune.cfg` | `/data/hwu/Uni_sign_repo/out/t5_youtube3d_to_how2sign_vq_finetune` |

Thin launch wrappers:

```bash
bash script/train_vq_t5_pretrain.sh
bash script/train_vq_t5_finetune.sh
```

The fine-tuning wrapper requires the pretraining run's
`best_checkpoint.pth` and passes it through the native `--finetune` runtime
override. It fails before launch when that checkpoint is absent.

The pretraining config intentionally combines all three YouTube3D annotation
splits. The current cache manifest and `split_index.json` cover only train/test,
so val must first be generated with the same configured SOKE VQ-VAE checkpoint.
The Uni-Sign loader refuses to start until the val split exists in both files;
it does not silently consume untracked `.npy` files.

Training:

```bash
deepspeed --include localhost:0,1,3 --master_port 29513 \
  fine_tuning_vq_embeddings.py \
  --config configs/how2sign_youtube3d_vq_embeddings.cfg
```

Evaluation:

```bash
deepspeed --include localhost:0 --master_port 29513 \
  fine_tuning_vq_embeddings.py \
  --config configs/how2sign_youtube3d_vq_embeddings.cfg \
  --finetune /data/hwu/Uni_sign_repo/out/t5_h2s_youtube3d_vq_embeddings/best_checkpoint.pth \
  --eval
```

Continue fine-tuning model weights:

```bash
deepspeed --include localhost:0,1,3 --master_port 29513 \
  fine_tuning_vq_embeddings.py \
  --config configs/how2sign_youtube3d_vq_embeddings.cfg \
  --finetune /path/to/a/vq_embedding_checkpoint.pth
```

`--finetune` restores model weights strictly but initializes a new optimizer,
scheduler, and epoch counter. Saved checkpoints contain the mT5/adapter model
state plus VQ provenance and a config snapshot.

## Validation performed

- Python compilation passed for dataset, model, entrypoint, and tests.
- Three focused unit tests passed: cache/provenance/collate, adapter gradients,
  and invalid geometry rejection.
- Real cache loading produced 196,103 unique train samples from How2Sign +
  YouTube3D and 2,308 How2Sign test samples.
- Real samples loaded as float32 `[T',3,512]`; a mixed two-source batch produced
  `[2,91,3,512]` with valid mask lengths `[91,21]`.
- A real How2Sign sample completed an mT5 CPU forward with input shape
  `[1,57,768]` and finite loss.

## Known caveats

- The current cache contains train/test but no val embeddings. The staged
  pretraining config therefore combines only YouTube3D train+test and does not
  require val. Pretraining checkpoint selection uses How2Sign test; reserve a
  separate How2Sign validation set for a publication-quality comparison.
- `split_index.json` contains 51 duplicate train entries and 5 duplicate test
  entries. Annotation-driven construction keeps one record per unique sample.
- A full multi-GPU DeepSpeed epoch was not launched as part of integration
  validation.
- VQ embeddings are checkpoint-specific. Changing the VQ-VAE checkpoint or
  codebook geometry requires a distinct cache/config and a matching manifest.
