# SignSMD-2D Generation Rules

## 1. Scope and purpose

`generate_signsmd2d.py` is a deterministic, rule-based compiler that converts
RTMPose COCO-WholeBody 2D keypoint sequences into structured English
descriptions of signing motion.

It does **not** use:

- the ground-truth translation;
- gloss annotations;
- caption text;
- an LLM or another learned captioning model.

The label files are used only to obtain `video_path`, which determines the
input pose pickle and the output text filename.

The complete pipeline is:

```text
label video_path
    -> RTMPose 133-keypoint pickle
    -> optional [start:end) temporal crop
    -> trainer-aligned linear temporal sampling
    -> confidence filtering and short-gap interpolation
    -> 2D geometric evidence extraction
    -> rule-based temporal analysis
    -> structured SignSMD text
```

The bulk driver is [`generate_signsmd2d.py`](../generate_signsmd2d.py). The
geometric and realization rules are implemented in the
[`signsmd2d`](../signsmd2d) package.

## 2. Supported data and file mapping

The current driver supports the following datasets and splits:

| Dataset | Splits | Pose directory |
| --- | --- | --- |
| How2Sign | `train`, `test` | `data/How2Sign/<split>-pose-rtmpose-192` |
| OpenASL | `train`, `dev`, `test` | `data/OpenASL/pose-rtmpose-192` |

Each pose pickle must contain:

```python
pose["keypoints"]  # [T, 1, 133, 2] or [T, 133, 2]
pose["scores"]     # [T, 1, 133] or [T, 133]
```

If `pose["start"]` and `pose["end"]` exist, the compiler first selects the
half-open interval:

```python
keypoints = keypoints[start:end]
scores = scores[start:end]
```

The clip ID is `video_path` with only the final `.mp4` removed. The output
contract is:

```text
<output_root>/texts/<clip-id>.txt
```

This makes the generated text IDs identical to the IDs used by the Uni-Sign
dataloader.

## 3. Exact temporal-alignment rule

Before computing any geometric evidence, the generator resamples the cropped
sequence using the same rule as the SignSMD training dataloader.

For a sequence containing `T` cropped frames:

```python
if T > max_length:
    indices = np.round(
        np.linspace(0, T - 1, max_length)
    ).astype(np.int64)
else:
    indices = np.arange(T)
```

All evidence, segmentation, and textual spans are computed **after** this
operation. Therefore:

```text
text step i = model pose token i
```

A span such as `[step 20-35]` refers directly to pose tokens 20 through 35,
inclusive, in the sequence seen by the model.

Generation-time `--max_length` must consequently equal training-time
`--max_length`. If the two values differ, descriptions for long clips refer to
a different temporal sequence from the model input.

The generator default is:

```text
--max_length 256
```

The default `--fps` is 24. FPS is used only for the approximate duration in the
text header and for the manifest. Motion thresholds are measured per sampled
step, not per second.

Relevant implementations:

- [`signsmd2d/compiler.py`](../signsmd2d/compiler.py), `resample_like_trainer`;
- [`datasets_signsmd.py`](../datasets_signsmd.py), `load_pose`.

## 4. RTMPose landmark layout

The compiler uses the following zero-based COCO-WholeBody indices:

- body:
  - nose: 0;
  - shoulders: 5 and 6;
  - elbows: 7 and 8;
  - wrists: 9 and 10;
- face: 23-90;
- left hand: 91-111;
- right hand: 112-132.

Each hand contains 21 points:

```text
wrist
thumb:  CMC, MCP, IP, tip
index:  MCP, PIP, DIP, tip
middle: MCP, PIP, DIP, tip
ring:   MCP, PIP, DIP, tip
pinky:  MCP, PIP, DIP, tip
```

The exact mapping is defined in [`signsmd2d/layout.py`](../signsmd2d/layout.py).

## 5. Confidence and missing-data rules

A keypoint is valid only when:

```python
score > 0.3
```

A score equal to or below 0.3 is treated as missing, rather than as a
zero-valued coordinate.

For each keypoint independently:

- an internal missing run of at most three sampled steps is linearly
  interpolated;
- a gap longer than three steps remains missing;
- missing values at the beginning or end are not extrapolated;
- invalid geometric measurements are represented by `NaN`.

A hand is geometrically valid at a step only if:

- the shoulder-based body coordinate frame is valid;
- the hand wrist is valid;
- all four non-thumb MCP points are valid;
- at least 13 of the 21 hand keypoints are valid.

If the shoulder-based body frame is valid for less than 20% of the clip, the
compiler emits an unreliable-detection message instead of a normal
articulation description.

These rules are implemented in
[`signsmd2d/evidence.py`](../signsmd2d/evidence.py).

## 6. Body-relative 2D coordinate system

The body coordinate frame is calculated independently at each step. Let
`LS` and `RS` denote the left and right shoulders:

```text
origin       = (LS + RS) / 2
lateral axis = normalize(RS - LS)
scale        = distance(LS, RS)
vertical axis = the upward-oriented perpendicular to the lateral axis
```

The nose, when valid, is used to confirm which direction of the perpendicular
axis is upward.

For a point `p`:

```text
x = dot(p - origin, lateral axis) / shoulder width
y = dot(p - origin, vertical axis) / shoulder width
```

Therefore:

- positive x means signer-right;
- negative x means signer-left;
- positive y means upward;
- body-space distances are expressed in shoulder-width units.

Hand-local distances are additionally normalized by palm scale. This reduces
sensitivity to image position, body size, and the apparent scale of the hand.

The representation remains strictly image-plane geometry. It does not recover
anatomical depth or true 3D joint angles.

## 7. Geometric evidence

### 7.1 Arm evidence

For each arm, the compiler calculates:

- wrist position in the body coordinate frame;
- elbow bend;
- forearm direction in the body coordinate frame.

Elbow bend is defined as:

```text
elbow_bend = pi - angle(shoulder - elbow, wrist - elbow)
```

An approximately straight arm therefore has a value near zero. Larger values
represent greater apparent 2D elbow flexion.

### 7.2 Per-finger curl

For each finger, the compiler constructs a five-point chain:

```text
hand wrist -> joint 1 -> joint 2 -> joint 3 -> fingertip
```

The curl value is the sum of three apparent 2D joint bends:

```text
curl = bend_1 + bend_2 + bend_3

bend_j = pi - angle(previous_point - joint_j,
                    next_point - joint_j)
```

Consequently:

- a straight finger has curl near 0 degrees;
- a more bent finger has a larger curl;
- the value can exceed 180 degrees because it is the sum of three bends.

This value is an image-plane articulation proxy, not an anatomical finger
joint angle.

### 7.3 Other hand evidence

For each hand, the compiler also calculates:

- mean adjacent-finger spread;
- thumb opposition, defined as the minimum thumb-tip-to-other-fingertip
  distance normalized by palm scale;
- index-to-pinky aperture;
- hand center in shoulder-normalized body coordinates;
- finger direction;
- a palm-orientation proxy.

The palm proxy is the signed 2D cross product between:

- the wrist-to-middle-MCP hand axis;
- the index-MCP-to-pinky-MCP knuckle line.

It is mirrored between the left and right hands to provide a consistent
camera-relative sign. It is not a true 3D palm normal.

### 7.4 Face and non-manual evidence

The compiler calculates:

- eyebrow height relative to the eyes;
- inner-mouth vertical opening;
- outer-mouth width;
- head roll/tilt from the eye line relative to the shoulder line;
- face position in body coordinates.

Brow height and mouth opening are normalized by interocular distance. The
median value across the clip is used as the neutral reference for brow and
mouth features.

## 8. Output variants

The generator supports three versioned contracts:

| Variant | Contract | Output |
| --- | --- | --- |
| `coarse` | `signsmd2d_v0_2` | Interval-level descriptive prose |
| `stats` | `signsmd2d_stats_v0_1` | Numeric per-channel timelines |
| `fingers` | `signsmd2d_fingers_v0_1` | Compact per-finger curl events |

The command-line default is `coarse`. The fingers-only dual-stream experiment
must explicitly pass:

```bash
--variant fingers
```

## 9. Coarse variant

### 9.1 Activity signals

The coarse variant measures the following per-step changes:

- wrist speed;
- summed five-finger curl change;
- palm-orientation change;
- elbow-bend and forearm-direction change;
- inter-hand separation change;
- brow-height change;
- mouth-opening change;
- head-tilt change.

Each change is divided by its configured activity threshold:

| Feature | Activity threshold per step |
| --- | ---: |
| Wrist speed | 0.045 shoulder widths |
| Handshape rate | 0.10 radians |
| Palm-orientation rate | 0.12 |
| Arm rate | 0.05 radians |
| Inter-hand separation rate | 0.05 shoulder widths |
| Brow rate | 0.020 interocular units |
| Mouth-opening rate | 0.035 interocular units |
| Head-tilt rate | 0.030 radians |

The maximum normalized rate across all articulators becomes the sequence
activity score. This score is smoothed with a five-step moving average:

```text
smoothed activity > 1.0  -> active
smoothed activity <= 1.0 -> inactive
```

Consecutive active and inactive runs form intervals. An interval shorter than
eight steps is merged into its longer neighboring interval.

An interval is labeled as:

- `hold`: inactive;
- `movement`: positional motion dominates or exceeds its threshold;
- `transition`: handshape or orientation change exceeds its threshold;
- `mixed`: active, but neither positional nor configuration evidence clearly
  dominates.

The implementation is in
[`signsmd2d/segmentation.py`](../signsmd2d/segmentation.py), and the thresholds
are in [`configs/signsmd2d_thresholds.yaml`](../configs/signsmd2d_thresholds.yaml).

### 9.2 Handshape categories

Each per-finger curl is categorized using:

```text
curl < 0.9 rad -> extended
curl > 2.2 rad -> closed
```

The rule set can produce:

- open flat handshape;
- open flat handshape with fingers spread;
- fist;
- fist with the index finger extended;
- handshape with index and middle fingers extended;
- pinching or rounded handshape;
- curved or cupped handshape.

If no category is sufficiently supported, the compiler omits the handshape
label instead of emitting an uncertainty category.

### 9.3 Spatial categories

Using the shoulder-normalized hand-center coordinates:

```text
y > 0.45  -> head height
y < -0.35 -> lower signing space
otherwise -> chest height

x > 0.35  -> signer-right side
x < -0.35 -> signer-left side
otherwise -> near the body midline
```

If the hand-to-face distance is below 0.45 shoulder widths, the compiler adds
`close to the face`.

### 9.4 Motion categories

Mean wrist speed is classified as:

```text
speed <= 0.020 -> hold
speed < 0.045  -> slow
speed < 0.10   -> moderate
speed >= 0.10  -> fast
```

Path straightness is:

```text
net displacement / total path length
```

A straightness value of at least 0.75 produces `straight`; otherwise the path
is described as `curved`. At least two direction reversals produce `with
repeated direction reversals`.

### 9.5 Bilateral roles

Whole-clip left and right wrist path lengths are compared. If one path is more
than 1.6 times the other, that hand is described as dominant and the other as
a supporting or base hand. If both have nonzero path length without such an
imbalance, both are described as actively articulating.

## 10. Stats variant

The stats variant produces numeric timelines for 28 channels:

- one bimanual channel:
  - hand separation;
- twelve channels per side:
  - wrist x;
  - wrist y;
  - elbow bend;
  - total five-finger curl;
  - finger spread;
  - thumb opposition;
  - palm-orientation proxy;
  - five individual finger curls;
- three non-manual channels:
  - brow height;
  - mouth opening;
  - signed head tilt.

For each channel, the compiler:

1. Splits the timeline into contiguous finite and missing runs.
2. Keeps finite runs only when they contain at least eight steps.
3. Reports missing gaps only when they contain at least eight steps.
4. Classifies the complete channel as static when the finite value range is
   below the channel's `min_change`.
5. Otherwise smooths the signal with a five-step window.
6. Computes local standard deviation over a ten-step activity window.
7. Divides the signal into static and active regions.
8. Splits active regions at peaks and valleys.
9. Reports holds, directional changes, or repeated cycles.

The main thresholds are:

| Channel | Minimum change | Static standard-deviation threshold |
| --- | ---: | ---: |
| Hand separation | 0.30 sw | 0.040 sw |
| Wrist x/y | 0.30 sw | 0.040 sw |
| Elbow bend | 20 degrees | 5 degrees |
| Total five-finger curl | 60 degrees | 15 degrees |
| Individual finger curl | 40 degrees | 10 degrees |
| Finger spread | 10 degrees | 3 degrees |
| Thumb opposition | 0.50 | 0.12 |
| Palm-orientation proxy | 0.50 | 0.12 |
| Brow height | 0.06 | 0.015 |
| Mouth opening | 0.15 | 0.045 |
| Head tilt | 8 degrees | 2.5 degrees |

A cyclic event requires:

- at least four moving segments;
- alternating positive and negative directions;
- mean amplitude at least equal to `min_change`;
- at least 60% of the amplitudes to differ from the mean amplitude by less
  than 60%;
- at least two complete cycles.

The output may contain statements such as:

```text
holds at ...
rises: ... -> ...
lowers: ... -> ...
repeats N cycles (... to ...)
not detected [step a-b]
```

The channel analysis is implemented in
[`signsmd2d/stats_segments.py`](../signsmd2d/stats_segments.py), the text
realization is in
[`signsmd2d/realization_stats.py`](../signsmd2d/realization_stats.py), and the
thresholds are in
[`configs/signsmd2d_stats_thresholds.yaml`](../configs/signsmd2d_stats_thresholds.yaml).

## 11. Fingers variant

The fingers variant uses the same per-finger geometry and channel-analysis
algorithm as the stats variant, but removes:

- wrist position;
- elbow bend;
- total hand curl;
- hand spread;
- thumb opposition;
- palm orientation;
- bilateral separation;
- facial and other non-manual features.

It emits only ten channels:

```text
Left:  thumb, index, middle, ring, pinky
Right: thumb, index, middle, ring, pinky
```

For each finger:

- minimum valid run: 8 steps;
- smoothing window: 5 steps;
- minimum static hold: 8 steps;
- global static criterion: finite value range below 40 degrees;
- local static threshold: standard deviation below 10 degrees;
- positive curl change: `curls`;
- negative curl change: `extends`;
- maximum retained non-static events: 2.

Ordinary moving events are ranked by absolute angle change. The two strongest
events are retained and then restored to chronological order.

The nominal `min_change` is 40 degrees. After peak/valley segmentation, the
implementation retains an individual moving segment if its absolute delta is
at least half of that value, i.e. 20 degrees. However, a channel whose complete
finite range is below 40 degrees is classified as static before segmentation.

A cyclic run is represented as one event:

```text
repeats N cycles (minimum angle to maximum angle) [step a-b]
```

If the channel has a sufficiently long valid run but no retained substantial
movement, the output is:

```text
no substantial change (~mean angle)
```

If no valid run of at least eight steps exists, the output is:

```text
not detected throughout
```

Unlike the full stats variant, the compact fingers variant does not print
intermediate missing gaps. It reports only the completely unusable-channel
case.

An example output format is:

```text
Motion: 256 steps (linearly downsampled from 322 frames, ~13.4s at 24 FPS)

[Left Fingers]
  Thumb curl:
    no substantial change (~35 degrees)
  Index finger curl:
    extends: 158 degrees -> 61 degrees [step 0-9]
    curls: 28 degrees -> 132 degrees [step 138-146]

[Right Fingers]
  Index finger curl:
    repeats 9 cycles (19 degrees to 182 degrees) [step 0-255]
```

The realization is implemented in
[`signsmd2d/realization_fingers.py`](../signsmd2d/realization_fingers.py), and
the thresholds are in
[`configs/signsmd2d_fingers_thresholds.yaml`](../configs/signsmd2d_fingers_thresholds.yaml).

## 12. Restrictions imposed by 2D geometry

The compiler rejects generated text containing the following terms:

```text
forward
backward
contact
toward the body
away from the body
```

These terms would imply depth or physical contact that cannot be established
from 2D keypoints.

The implementation instead uses restricted language such as:

- `palm appears to face the camera`;
- `palm appears to face away from the camera`;
- `the hands overlap in the image`;
- `toward signer-left` and `toward signer-right`.

The descriptions should therefore be interpreted as structured 2D visual
evidence, not as complete anatomical, semantic, or linguistic descriptions of
the sign.

The vocabulary assertion is implemented in
[`signsmd2d/compiler.py`](../signsmd2d/compiler.py).

## 13. Generation, resume, and failure behavior

The default output directories are:

```text
data/<dataset>/signsmd2d_v0_2
data/<dataset>/signsmd2d_stats_v0_1
data/<dataset>/signsmd2d_fingers_v0_1
```

For each clip:

1. Text is written to `<clip>.txt.tmp`.
2. The temporary file is atomically renamed to `<clip>.txt`.
3. Existing text files are skipped by default.
4. `--overwrite` is required to regenerate existing files.

The generator supports deterministic sharding:

```text
--num_shards N --shard K
```

After sorting the label IDs, shard `K` receives:

```python
ids[K::N]
```

Per-sample exceptions are collected without stopping the remaining generation
unless `--fail_fast` is supplied. The process exits with code 1 if any sample
failed.

Each manifest records:

- contract version and variant;
- dataset and split;
- label and pose paths;
- `max_length`;
- assumed FPS;
- threshold-file path;
- SHA-256 of the threshold file;
- completed, skipped, and failed counts;
- up to 200 failure records;
- sharding information;
- elapsed time.

## 14. Current temporal-contract mismatch

The existing generated `signsmd2d_fingers_v0_1` manifests for How2Sign
train/test and OpenASL train/test record:

```text
max_length = 512
```

The current dual-stream fingers launcher instead trains with:

```text
--frame_sampling linear
--max_length 256
```

For clips longer than 256 frames, the existing text describes a sequence of up
to 512 steps while the model receives only 256 pose steps. The textual
`[step a-b]` spans are therefore not aligned with the current training input.
Clips containing at most 256 frames are unaffected.

If the intended training length remains 256, both corpora should be
regenerated with overwrite:

```bash
python generate_signsmd2d.py \
  --dataset OpenASL \
  --splits train dev test \
  --variant fingers \
  --max_length 256 \
  --overwrite

python generate_signsmd2d.py \
  --dataset How2Sign \
  --splits train test \
  --variant fingers \
  --max_length 256 \
  --overwrite
```

Running these commands without `--overwrite` would skip the existing
512-step files.

There is currently no `manifest_OpenASL_dev.json` under the fingers contract.
This does not by itself prove that all dev text files are absent, but it means
there is no independent generation record for that split.

## 15. Stale percentage-based documentation

Some older comments in `datasets_signsmd.py`, `fine_tuning_signsmd.py`, and
`docs/POSE_SIGNSMD_FUSION.md` still refer to percentage-based temporal spans.
That wording describes an earlier contract and is stale for the current
implementation.

The current compiler produces inclusive, trainer-aligned `[step a-b]` spans.
The executable compiler and the versioned threshold manifests should be
treated as authoritative.

