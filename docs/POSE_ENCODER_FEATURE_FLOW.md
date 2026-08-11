# Uni-Sign Pose Encoder and Feature-Dimension Flow

This document describes the **pose-only path implemented in the current
Uni-Sign checkout**. It follows the tensors from the 133-keypoint RTMPose input,
through the four ST-GCN branches, to the embeddings consumed by mT5-Base. It
also explains how Stage-1 pre-training changes the initialization of this path.

The main implementation files are:

- [`datasets.py`](../datasets.py): pose selection, normalization, temporal
  sampling, padding, and masks.
- [`models.py`](../models.py): pose branches, cross-part connections, feature
  concatenation, projection, and mT5 input construction.
- [`stgcn_layers/gcn_utils.py`](../stgcn_layers/gcn_utils.py): skeleton graphs.
- [`stgcn_layers/stgcn_block.py`](../stgcn_layers/stgcn_block.py): spatial and
  temporal ST-GCN blocks.
- [`script/train_stage1.sh`](../script/train_stage1.sh): released Stage-1
  training command.

## 1. End-to-end overview

For a pose-only run, `--rgb_support` is omitted. The active path is:

```text
RTMPose whole-body keypoints
    [B, T, 133, 2] coordinates
    [B, T, 133]    confidence scores
                  |
                  v
Select and normalize 69 keypoints
    body:       [B, T,  9, 3]
    left hand:  [B, T, 21, 3]
    right hand: [B, T, 21, 3]
    face:       [B, T, 18, 3]
                  |
                  v
Four part-aware ST-GCN branches
    each branch: [B, T, 256]
                  |
                  v
Feature-dimension concatenation
    [B, T, 4 x 256] = [B, T, 1024]
                  |
                  v
Linear projection: 1024 -> 768
    pose tokens: [B, T, 768]
                  |
                  v
Sequence-dimension concatenation with prompt embeddings
    [B, P + T, 768]
                  |
                  v
mT5-Base encoder-decoder
                  |
                  v
Generated target sentence
```

Here:

- `B` is batch size.
- `T` is the padded pose sequence length in the current batch, capped at 256
  frames by default.
- `P` is the number of mT5 tokens in the language prompt.
- The last input coordinate is `(x, y, confidence)`, so its feature dimension
  is 3.

## 2. Input pose contract

### 2.1 Raw representation

The loader expects RTMPose-style whole-body data stored in a pickle file with:

```python
pose["keypoints"]  # per-frame coordinates, based on 133 whole-body points
pose["scores"]     # per-frame confidence for the same 133 points
```

The code expects a leading detected-person dimension and removes it with
`skeleton = skeleton[0]` and `conf = conf[0]`. Conceptually, one sample before
part selection is therefore:

```text
coordinates: [T, 133, 2]
scores:      [T, 133]
```

Coordinates and confidence are combined after part selection to form a
three-channel joint representation.

### 2.2 Temporal sampling

The default `--max_length` is 256. If a video contains more than 256 frames,
the loader:

1. randomly samples 256 frame indices;
2. sorts the indices;
3. preserves the temporal order of the selected frames.

If the sequence has at most 256 frames, all frames are used.

This is sparse random temporal sampling, not interpolation or uniform
resampling.

### 2.3 Selection of 69 keypoints

The 133 points are reduced to four code-level branches:

| Branch | Source indices | Joint count | Tensor before batching |
|---|---:|---:|---:|
| Body | `[0] + [3, ..., 10]` | 9 | `[T, 9, 3]` |
| Left hand | `[91:112]` | 21 | `[T, 21, 3]` |
| Right hand | `[112:133]` | 21 | `[T, 21, 3]` |
| Face | 9 contour points + 8 mouth points + point 53 | 18 | `[T, 18, 3]` |

The total is `9 + 21 + 21 + 18 = 69` keypoints.

Although the paper may group both hands conceptually, the released code uses
four separate modes:

```python
self.modes = ["body", "left", "right", "face_all"]
```

### 2.4 Coordinate normalization and confidence masking

The confidence threshold is `0.3`.

- The body coordinates define a sequence-level square crop and scale.
- Body coordinates are normalized to approximately `[-1, 1]`.
- Left- and right-hand coordinates are first made wrist-relative by subtracting
  the first hand joint.
- Face coordinates are made relative to the final selected facial anchor.
- Hand and face coordinates are divided by the body-derived scale and clipped
  to `[-1, 1]`.
- A point with confidence `<= 0.3` is replaced by zeros, including its
  coordinate and confidence channels.

The final input for every selected point is:

```text
[normalized_x, normalized_y, confidence]
```

## 3. Batch construction and attention mask

Sequences in a batch are padded to the longest `T` in that batch. Padding is
implemented by repeating the last pose frame, rather than by inserting an
all-zero pose.

For each part, the collated shape is:

```text
body:       [B, T,  9, 3]
left:       [B, T, 21, 3]
right:      [B, T, 21, 3]
face_all:   [B, T, 18, 3]
```

The loader also creates:

```text
attention_mask: [B, T]
```

Real frames receive mask value 1 and padded frames receive 0. The ST-GCN sees
the repeated padding frames, while the mask is later passed to mT5 so that mT5
does not attend to padded pose tokens.

## 4. ST-GCN branch architecture

Each branch starts with a point-wise projection:

```text
[B, T, V, 3]
    Linear(3, 64)
-> [B, T, V, 64]
    permute
-> [B, 64, T, V]
```

`V` is 9, 21, 21, or 18 depending on the part.

### 4.1 Spatial encoder

Each part has a skeleton graph matching its joint topology. The graph uses:

- distance partitioning;
- maximum hop distance 1;
- adaptive adjacency, meaning the initialized graph adjacency is trainable.

The three spatial ST-GCN blocks have channel widths:

```text
64 -> 64 -> 128 -> 256
```

The temporal kernel in this first chain is 1, so this stage primarily performs
per-frame spatial graph modeling.

Output:

```text
[B, 256, T, V]
```

### 4.2 Parameter sharing between hands

The left- and right-hand modules are explicitly tied:

```python
self.gcn_modules["left"] = self.gcn_modules["right"]
self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
self.proj_linear["left"] = self.proj_linear["right"]
```

Therefore, both hands use the same input projection, spatial GCN, and temporal
GCN parameters.

### 4.3 Cross-part body anchors

Before temporal encoding, Uni-Sign injects a body anchor into each non-body
part:

```text
left-hand nodes  += detached left-wrist body feature
right-hand nodes += detached right-wrist body feature
face nodes       += detached head/nose body feature
```

The body feature is broadcast to all nodes in the corresponding branch. The
use of `.detach()` means these additions do not send gradient back into the
body branch through the cross-part connection.

Consequently, the pose-only model is not simply four fully independent GCNs,
even though their final outputs are concatenated directly.

### 4.4 Temporal encoder

Each part then passes through another three ST-GCN blocks:

```text
256 -> 256 -> 256 -> 256
```

These blocks use a temporal convolution kernel of 5 and residual connections.
The output shape remains:

```text
[B, 256, T, V]
```

### 4.5 Intra-part joint pooling

The code averages over the joint dimension `V` and transposes the result:

```text
[B, 256, T, V]
    mean over V
-> [B, 256, T]
    transpose
-> [B, T, 256]
```

Thus every time step is represented by one 256-dimensional vector per part.

## 5. Four-part feature fusion

The four `[B, T, 256]` tensors are concatenated along the **feature dimension**:

```text
cat([body, left, right, face], dim=-1)
    -> [B, T, 1024]
```

This is an important implementation detail: the comment in `models.py` says
"concat ... across token dimension," but `dim=-1` actually concatenates the
feature channels. The sequence length remains `T`.

A learned zero-initialized vector is then added:

```python
self.part_para = nn.Parameter(torch.zeros(1024))
```

It broadcasts across batch and time:

```text
[B, T, 1024] + [1024] -> [B, T, 1024]
```

There is no attention, Q-Former, resampler, or other learned interaction module
between the four pooled pose features in the pose-only path. Their learned
interaction after concatenation begins with the following linear projection.

## 6. Projection into the mT5 embedding space

The concatenated pose feature is mapped to the mT5-Base hidden width:

```python
self.pose_proj = nn.Linear(256 * 4, 768)
```

Shape transformation:

```text
[B, T, 1024] -> [B, T, 768]
```

Every pose frame is now treated as one continuous 768-dimensional input token.
The pose is **not** quantized into text token IDs.

### Configuration constraint

The CLI exposes `--hidden_dim`, whose default is 256. However, the pose branch
output and `pose_proj` are hard-coded around `4 x 256 = 1024`. In addition,
`part_para` is created with `hidden_dim * 4` elements. Changing `hidden_dim`
away from 256 without updating the model will cause a shape mismatch.

Likewise, replacing mT5-Base with a model whose hidden width is not 768 requires
changing `pose_proj`.

## 7. Prompt and mT5 sequence construction

The prompt depends on the dataset language:

```text
CSL datasets:  "Translate sign language video to Chinese: "
other datasets: "Translate sign language video to English: "
```

The prompt is tokenized and converted to embeddings using the mT5 encoder's
token embedding table:

```text
prompt token IDs -> [B, P, 768]
```

The prompt and pose are then concatenated along the **sequence dimension**:

```text
prompt embeddings: [B, P, 768]
pose embeddings:   [B, T, 768]
cat(dim=1):        [B, P + T, 768]
```

The matching mask is:

```text
prompt mask: [B, P]
pose mask:   [B, T]
combined:    [B, P + T]
```

The complete tensor is passed to `MT5ForConditionalGeneration` via
`inputs_embeds`. Therefore, the full mT5 encoder and autoregressive decoder are
used; mT5 is not acting as a decoder-only model.

## 8. Training output and objective

The reference sentence is tokenized to at most 50 tokens. Padding token IDs are
replaced by `-100` so they are ignored by the loss.

If the target length is `L`, the decoder logits conceptually have shape:

```text
[B, L, vocabulary_size]
```

The released model computes token-level cross-entropy with label smoothing:

```python
CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=-100)
```

The default label-smoothing value is 0.2.

At evaluation time, the released scripts call mT5 generation with:

```text
num_beams = 4
max_new_tokens = 100
```

## 9. Stage-1 initialization and pre-training

At initial construction:

- the ST-GCN pose encoder is randomly initialized;
- the 1024-to-768 pose projection is randomly initialized;
- mT5-Base is loaded from its language-model pre-trained checkpoint;
- no pose or mT5 parameters are explicitly frozen.

Stage 1 performs supervised generative sign-to-text training. The released
command uses pose-only CSL-News training for 20 epochs. Gradients update the
pose encoder, pose projection, and mT5 together.

After Stage 1, the checkpoint contains a sign-pre-trained ST-GCN and its learned
alignment to mT5. For a downstream pose-only experiment, Stage 2 can be skipped
and the downstream run can initialize from:

```text
out/stage1_pretraining/best_checkpoint.pth
```

Training a downstream dataset without that checkpoint is possible, but it
leaves the ST-GCN and pose projection randomly initialized. mT5 still remains
text-pre-trained, so this setting is more precisely "without sign-language
pre-training" rather than a fully random model.

The paper states that Stage-1/Stage-2 pre-training uses CSL-News for Chinese
Sign Language tasks and YouTube-ASL for American Sign Language tasks. The
current released Stage-1 shell script only contains the CSL-News command; the
complete YouTube-ASL pre-training loader is not present in this checkout.

## 10. Shape reference table

| Location | Tensor shape | Feature width |
|---|---:|---:|
| Raw selected body | `[B, T, 9, 3]` | 3 |
| Raw selected left hand | `[B, T, 21, 3]` | 3 |
| Raw selected right hand | `[B, T, 21, 3]` | 3 |
| Raw selected face | `[B, T, 18, 3]` | 3 |
| Input projection | `[B, 64, T, V]` | 64 |
| Spatial GCN output | `[B, 256, T, V]` | 256 |
| Temporal GCN output | `[B, 256, T, V]` | 256 |
| Joint-pooled part feature | `[B, T, 256]` | 256 |
| Four-part concatenation | `[B, T, 1024]` | 1024 |
| Projected pose tokens | `[B, T, 768]` | 768 |
| Prompt embeddings | `[B, P, 768]` | 768 |
| Combined mT5 input | `[B, P + T, 768]` | 768 |
| mT5 encoder hidden states | `[B, P + T, 768]` | 768 |
| Decoder logits | `[B, L, vocabulary_size]` | vocabulary size |

## 11. Minimal pose-only configuration

Do not pass `--rgb_support`. A downstream pose-only command has the following
shape:

```bash
deepspeed --include localhost:0,1,2,3 fine_tuning.py \
  --batch-size 8 \
  --epochs 20 \
  --lr 3e-4 \
  --output_dir out/pose_only_finetuning \
  --finetune out/stage1_pretraining/best_checkpoint.pth \
  --dataset DATASET_NAME \
  --task SLT
```

Omitting `--rgb_support` prevents construction and execution of the
EfficientNet-B0 hand-image branch, score-aware RGB sampling, deformable
attention, and PGF gate. None of those components changes the pure-pose feature
dimensions described above.
