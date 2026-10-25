# Uni-Sign Model Architecture and Data Flow

This document introduces the Uni-Sign architecture used for the OpenASL and
How2Sign inputs described in
[`OPENASL_HOW2SIGN_INPUT_FORMAT.md`](OPENASL_HOW2SIGN_INPUT_FORMAT.md). It
focuses on the default **pose-only sign-language translation (SLT)** path in the
current checkout:

```text
datasets.py -> models.py::Uni_Sign -> mT5 -> English translation
```

The description below follows the implementation rather than treating the
paper diagram as an exact specification. The optional hand-RGB and pose2d
KL-VAE paths are separated near the end.

## 1. Architecture at a glance

Uni-Sign is an encoder-decoder model with two main components:

1. A part-aware ST-GCN pose encoder converts each video frame into a continuous
   pose embedding.
2. An mT5 encoder-decoder reads a textual task prompt followed by those pose
   embeddings and autoregressively generates the translation.

```text
OpenASL / How2Sign label and pose files
                  |
                  v
RTMPose whole-body sequence
coordinates [B,T,133,2] + confidence [B,T,133]
                  |
                  | select, normalize, and confidence-mask 69 joints
                  v
     +------------+------------+------------+------------+
     |                         |                         |
 body [B,T,9,3]       hands [B,T,21,3] each   face [B,T,18,3]
     |                         |                         |
     +------------ four part-aware ST-GCN branches -----+
                  |
                  v
four part features [B,T,256] each
                  |
                  | concatenate on the feature axis
                  v
fused pose features [B,T,1024]
                  |
                  | Linear(1024, mT5 hidden size)
                  v
continuous pose tokens [B,T,768] by default
                  |
                  | prepend prompt embeddings [B,P,768]
                  v
mT5 encoder input [B,P+T,768] + attention mask [B,P+T]
                  |
                  v
mT5 autoregressive decoder
                  |
                  v
English sentence
```

Notation:

- `B`: batch size.
- `T`: longest pose sequence in the batch after sampling and padding.
- `P`: number of tokens in the textual prompt.
- `L`: number of tokens in the target sentence.
- `V`: number of joints in one body-part graph.

With the default local `mt5-base`, the mT5 hidden size is 768. The current code
reads this width from the selected mT5 configuration when it creates the pose
projection.

## 2. From files to model inputs

### 2.1 Labels locate pose sequences and supply targets

`S2T_Dataset` reads a gzip-compressed pickle label file. Each record supplies
the clip name, `video_path`, and target English `text`. The loader replaces the
`.mp4` suffix in `video_path` with `.pkl` and opens the corresponding pose
file.

For OpenASL and How2Sign, each pose pickle provides:

```text
keypoints: [T_raw,1,133,2]
scores:    [T_raw,1,133]
```

The singleton dimension represents one detected signer. The 133 entries are
COCO-WholeBody RTMPose joints. They are **not** a 133-dimensional SMPL-X motion
feature.

### 2.2 Temporal sampling

The default maximum length is 256 frames:

```text
T_raw <= 256: keep every frame
T_raw >  256: randomly choose 256 distinct frames, then sort their indices
```

Sorting preserves temporal order, but the spacing is not uniform. Because the
same loader is used for evaluation, predictions for clips longer than 256
frames can vary between runs unless randomness is controlled or the sampling
policy is changed.

### 2.3 Part selection and normalization

`datasets.py::load_part_kp()` reduces the 133-point input to four semantic
parts:

| Model branch | RTMPose selection | Joints | Per-sample output |
|---|---|---:|---:|
| `body` | `[0] + [3, ..., 10]` | 9 | `[T,9,3]` |
| `left` | `[91:112]` | 21 | `[T,21,3]` |
| `right` | `[112:133]` | 21 | `[T,21,3]` |
| `face_all` | 9 face-contour points, 8 mouth points, and point 53 | 18 | `[T,18,3]` |

The last dimension is:

```text
[normalized x, normalized y, confidence]
```

The body determines a sequence-level square crop and scale. Hand coordinates
are made wrist-relative; face coordinates are made relative to point 53. Hand
and face offsets are divided by the body scale. Coordinates are clipped to
`[-1,1]`, and a joint whose confidence is at most `0.3` is replaced by
`[0,0,0]`.

### 2.4 Batch construction

`Base_Dataset.collate_fn()` pads every sequence to the longest sample in its
batch by repeating that sample's last frame:

```text
body:           [B,T, 9,3]
left:           [B,T,21,3]
right:          [B,T,21,3]
face_all:       [B,T,18,3]
attention_mask: [B,T]
```

The repeated frames pass through the pose encoder, but their attention-mask
values are zero, so mT5 does not treat the corresponding pose tokens as valid
encoder positions. `fine_tuning.py::move_src_input_to_device()` moves the
tensors to the GPU and converts floating tensors from the loader's dtype to
the configured training dtype.

## 3. Part-aware ST-GCN pose encoder

`models.py::Uni_Sign.encode_pose()` processes the four parts in this order:

```python
["body", "left", "right", "face_all"]
```

### 3.1 Per-joint input projection

Each joint's `(x,y,confidence)` vector is projected independently:

```text
[B,T,V,3]
    -> Linear(3,64)
[B,T,V,64]
    -> permute
[B,64,T,V]
```

### 3.2 Spatial graph encoder

Each part has a graph matching its topology. The graph uses distance
partitioning, a maximum hop of 1, and trainable adaptive adjacency matrices.
Three ST-GCN blocks change the channel width as follows:

```text
64 -> 64 -> 128 -> 256
```

The temporal kernel in this spatial chain is 1, so its principal role is
within-frame graph reasoning. Its output is `[B,256,T,V]`.

### 3.3 Shared hand parameters and body-anchor injection

The left and right hands are separate input streams but share their input
projection, spatial ST-GCN, and temporal ST-GCN parameters.

Before temporal modeling, the model adds a detached body feature to every node
of each non-body part:

```text
left hand  += body left-wrist feature
right hand += body right-wrist feature
face       += body head/nose feature
```

The anchor is broadcast over the joints in the receiving part. The `.detach()`
operation prevents these cross-part additions from sending gradients back into
the body branch.

### 3.4 Temporal encoder and joint pooling

Every part then passes through three temporal ST-GCN blocks. They preserve 256
channels and use a temporal convolution kernel of 5:

```text
[B,256,T,V]
    -> temporal ST-GCN x 3
[B,256,T,V]
    -> mean over V
[B,256,T]
    -> transpose
[B,T,256]
```

Thus, the pose encoder emits one 256-dimensional vector per part and frame.

## 4. Pose fusion and the mT5 interface

### 4.1 Feature fusion

The four part vectors are concatenated along the feature dimension:

```text
body [B,T,256] + left [B,T,256] + right [B,T,256] + face [B,T,256]
    -> concatenate(dim=-1)
[B,T,1024]
```

A learned zero-initialized vector, `part_para`, is added to the fused feature.
There is no pose tokenizer, Q-Former, or resampler in this default path.

The fused feature is projected into the selected mT5 embedding space:

```text
[B,T,1024] -> Linear(1024,d_model) -> [B,T,d_model]
```

For the default mT5-Base configuration, `d_model = 768`. Each frame is now a
continuous pose token. It is not converted to a SentencePiece token ID.

### 4.2 Prompt plus pose sequence

OpenASL and How2Sign use this prompt:

```text
Translate sign language video to English:
```

The prompt is tokenized normally and embedded with mT5's encoder token table.
Prompt and pose embeddings are then concatenated along the sequence dimension:

```text
prompt: [B,P,d_model]
pose:   [B,T,d_model]
result: [B,P+T,d_model]
```

Their masks are concatenated in the same order. The result is passed to
`MT5ForConditionalGeneration` through `inputs_embeds`, so Uni-Sign uses the
full mT5 encoder-decoder rather than a decoder-only language model.

## 5. Training and inference flows

### 5.1 Training

For SLT, the label record's English `text` is the target. It is tokenized to a
maximum of 50 tokens, and padding token IDs are replaced with `-100`.

```text
pose + prompt -> mT5 encoder
target prefix -> mT5 decoder with teacher forcing
decoder logits [B,L,vocabulary_size]
              -> label-smoothed cross-entropy
```

The default label-smoothing value is `0.2`. In the normal pose-only path, the
ST-GCN, pose projection, mT5 encoder, and mT5 decoder are all trainable.

### 5.2 Inference and evaluation

Evaluation first runs the same forward pass to obtain loss and the prepared
encoder inputs. `Uni_Sign.generate()` then invokes mT5 generation. The standard
evaluation loop uses beam search with:

```text
num_beams = 4
max_new_tokens = 100
```

For SLT, decoded hypotheses are compared with references using BLEU and ROUGE.
During training, `best_checkpoint.pth` is selected by BLEU-4 on the selection
loader.

OpenASL has a distinct development split. How2Sign does not have one in the
current configuration, so `fine_tuning.py` aliases its test loader as the
development/selection loader. Consequently, selecting the best How2Sign
checkpoint in this setup uses test BLEU-4 and leaks test-set information.

## 6. Why OpenASL pre-training transfers to How2Sign

The two datasets differ in their label splits and pose-directory layout, but
their model-facing schema is identical:

```text
four normalized part tensors + frame mask -> same Uni_Sign model -> English
```

Therefore, a pose-only OpenASL checkpoint can initialize the same ST-GCN,
projection, and mT5 parameter structure for How2Sign. This is transfer learning
across datasets, not a change to the forward architecture.

## 7. Optional architecture paths

### 7.1 Hand-RGB support

Passing `--rgb_support` adds an EfficientNet-B0 branch for sampled hand crops.
Projected hand-image features interact with the hand pose features through
deformable attention and a learned gate before the body-anchor addition and
temporal ST-GCN. This changes the model structure and checkpoint contract.

The OpenASL/How2Sign workflow described in the input-format document is
pose-only when `--rgb_support` is omitted.

### 7.2 Pose2d KL-VAE replacement

The separate `fine_tuning_pose2d_klvae.py` path replaces all four ST-GCN
branches with a frozen pose2d KL-VAE encoder:

```text
four normalized pose parts
    -> keep xy and flatten 69 joints
[B,T,138]
    -> z-score normalization
frozen KL-VAE encoder
    -> posterior mean [T_i/4,512]
Linear(512,d_model)
    -> mT5 pose embeddings
```

It uses deterministic VAE-compatible temporal sampling and a downsampled
attention mask. The downstream prompt, mT5 objective, and generation logic are
conceptually the same, but its checkpoints are not structurally compatible
with the original ST-GCN checkpoints. See
[`POSE2D_KLVAE_T5.md`](POSE2D_KLVAE_T5.md) for that path's full contract.

## 8. Shape reference

| Stage | Shape |
|---|---:|
| Raw coordinates | `[B,T_raw,133,2]` |
| Raw confidence | `[B,T_raw,133]` |
| Body input | `[B,T,9,3]` |
| Left/right hand input | `[B,T,21,3]` each |
| Face input | `[B,T,18,3]` |
| Projected part input | `[B,64,T,V]` |
| Spatial ST-GCN output | `[B,256,T,V]` |
| Temporal ST-GCN output | `[B,256,T,V]` |
| Joint-pooled part output | `[B,T,256]` |
| Four-part concatenation | `[B,T,1024]` |
| Default mT5 pose tokens | `[B,T,768]` |
| Default combined encoder input | `[B,P+T,768]` |
| Decoder logits | `[B,L,vocabulary_size]` |

## 9. Code map

| Responsibility | Implementation |
|---|---|
| Dataset paths | [`config.py`](../config.py) |
| Pose loading and temporal sampling | [`datasets.py`](../datasets.py) `S2T_Dataset.load_pose()` |
| Joint selection and normalization | [`datasets.py`](../datasets.py) `load_part_kp()` and `crop_scale()` |
| Padding and attention masks | [`datasets.py`](../datasets.py) `Base_Dataset.collate_fn()` |
| Part graphs | [`stgcn_layers/gcn_utils.py`](../stgcn_layers/gcn_utils.py) `Graph` |
| Spatial/temporal ST-GCN blocks | [`stgcn_layers/stgcn_block.py`](../stgcn_layers/stgcn_block.py) |
| Pose fusion and mT5 interface | [`models.py`](../models.py) `Uni_Sign` |
| Training, generation, and checkpoint selection | [`fine_tuning.py`](../fine_tuning.py) |

For a more detailed discussion of the base pose encoder, including individual
graph stages and configuration constraints, see
[`POSE_ENCODER_FEATURE_FLOW.md`](POSE_ENCODER_FEATURE_FLOW.md).
