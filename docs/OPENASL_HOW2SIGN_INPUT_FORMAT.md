# Uni-Sign 中 OpenASL 与 How2Sign 的输入数据格式

本文基于当前 checkout：

```text
/home/student/hwu/Workplace/Uni-Sign
```

内容来自当前 `config.py`、`datasets.py`、`models.py` 以及本机实际
OpenASL/How2Sign label 和 pose 文件。

## 1. 先澄清：这里不是 OpenPose，也不是 133D motion

当前 OpenASL 和 How2Sign 使用的是 **RTMPose 提取的
COCO-WholeBody 133 个二维关键点**：

```text
每帧 keypoints: [1, 133, 2]
每帧 scores:    [1, 133]
```

- `1`：当前文件中保留的一个检测人物。
- `133`：COCO-WholeBody 133 个关键点。
- `2`：图像坐标 `(x, y)`。
- `scores`：每个关键点的检测置信度。

虽然部分 ST-GCN 源码注释中保留了 `openpose` 字样，但当前数据不是
OpenPose JSON，也没有加载 OpenPose 18/25 点格式。

这里也不是其他 motion-generation 仓库使用的 How2Sign `[T,133]`
SMPL-X motion feature。两者的 `133` 含义完全不同：

```text
本仓库:       133 个二维 RTMPose joints，每帧原始 shape [1,133,2]
Motion/VAE:   133 个 SMPL-X motion feature，每个时间步 shape [133]
```

Uni-Sign 原始 ST-GCN 路径不读取 `Mean.npy`、`Std.npy`，也不使用 SMPL-X。
（新增的 pose2d KL-VAE 路径会读取 VAE 训练时的 `mean.npy`/`std.npy` 做
z-score，见第 14 节。）

## 2. 磁盘目录

本次训练所使用的数据绝对根目录是：

```text
/home/student/hwu/Workplace/Uni-Sign/data
```

### 2.1 OpenASL

```text
/home/student/hwu/Workplace/Uni-Sign/data/OpenASL/
├── labels.train
├── labels.dev
├── labels.test
└── pose-rtmpose-192/
    ├── <clip-id>.pkl
    └── ...
```

当前数量：

| Split | Label 数量 | 缺失 pose |
|---|---:|---:|
| train | 96,476 | 0 |
| dev | 967 | 0 |
| test | 976 | 0 |

三个 split 共用同一个平铺的 pose 目录：

```text
/home/student/hwu/Workplace/Uni-Sign/data/OpenASL/pose-rtmpose-192/
```

### 2.2 How2Sign

```text
/home/student/hwu/Workplace/Uni-Sign/data/How2Sign/
├── labels.train
├── labels.test
├── train-pose-rtmpose-192/
│   ├── <clip-id>.pkl
│   └── ...
└── test-pose-rtmpose-192/
    ├── <clip-id>.pkl
    └── ...
```

当前数量：

| Split | Label 数量 | 缺失 pose |
|---|---:|---:|
| train | 31,086 | 0 |
| test | 2,349 | 0 |

How2Sign 的 pose 按 split 放在不同目录。当前没有 `labels.dev`，代码会令：

```python
dev_dataloader = test_dataloader
```

因此 How2Sign 的 `best_checkpoint.pth` 实际按 test BLEU-4 选择，存在
test-set leakage。

## 3. Label 文件格式

`labels.train`、`labels.dev` 和 `labels.test` 虽然没有 `.gz` 后缀，
但实际是 **gzip 压缩的 Python pickle**。加载后是：

```python
dict[str, dict]
```

每条记录包含：

```python
{
    "name": "<clip-name>.mp4",
    "gloss": "",
    "text": "<English sentence>",
    "video_path": "<clip-name>.mp4",
}
```

OpenASL 实例：

```python
{
    "name": "Mci9oyb5V2E-00:00:06.000-00:00:06.589.mp4",
    "gloss": "",
    "text": "Hello!",
    "video_path": "Mci9oyb5V2E-00:00:06.000-00:00:06.589.mp4",
}
```

How2Sign 实例：

```python
{
    "name": "--7E2sU6zP4_10-5-rgb_front.mp4",
    "gloss": "",
    "text": "And I call them decorative elements ...",
    "video_path": "--7E2sU6zP4_10-5-rgb_front.mp4",
}
```

Loader 使用 `video_path`，把结尾的 `.mp4` 换成 `.pkl`：

```text
<clip-name>.mp4 -> <pose-root>/<clip-name>.pkl
```

当前 OpenASL 和 How2Sign 的 `gloss` 都为空，SLT 训练目标来自 `text`。

## 4. 单个 pose pickle 的原始格式

### 4.1 OpenASL

实际文件包含：

```python
{
    "keypoints": np.ndarray,  # [T,1,133,2], float64
    "scores": np.ndarray,     # [T,1,133],   float32
}
```

例如：

```text
keypoints: (18, 1, 133, 2), float64
scores:    (18, 1, 133),    float32
```

### 4.2 How2Sign

实际文件包含：

```python
{
    "keypoints": np.ndarray,  # [T,1,133,2], float64
    "scores": np.ndarray,     # [T,1,133],   float32
    "w_h": np.ndarray,        # [2], int64
}
```

例如：

```text
keypoints: (322, 1, 133, 2), float64
scores:    (322, 1, 133),    float32
w_h:       (2,),             int64
```

`w_h` 记录图像宽高相关信息，但当前 pose-only `S2T_Dataset.load_pose()`
没有读取它。

## 5. 帧数处理

设原始片段长度为 `T_raw`，默认：

```text
max_length = 256
```

处理规则：

```text
T_raw <= 256:
    保留全部帧

T_raw > 256:
    从 range(T_raw) 随机抽取 256 个不同帧
    然后把帧索引排序，保持时间先后顺序
```

对应代码行为是：

```python
sorted(random.sample(range(duration), k=256))
```

因此它不是等间隔采样。长视频在不同 epoch 可能抽到不同帧，这相当于一种
temporal augmentation。

如果 pose pickle 提供可选的 `start` 和 `end` 字段，loader 只在
`[start,end)` 内采样；当前检查到的 OpenASL/How2Sign代表文件没有这两个字段。

## 6. 从 133 点中选择四个局部

每帧先去掉人物维度：

```text
[1,133,2] -> [133,2]
[1,133]   -> [133]
```

然后提取四个局部：

| 部位 | 原始 RTMPose 索引 | joint 数量 | 局部坐标处理 |
|---|---|---:|---|
| `body` | `[0,3,4,5,6,7,8,9,10]` | 9 | 全片段统一 bbox 归一化 |
| `left` | `[91:112]` | 21 | 减去左手腕坐标 |
| `right` | `[112:133]` | 21 | 减去右手腕坐标 |
| `face_all` | `[23,25,...,39] + [83:91] + [53]` | 18 | 减去索引 53 的面部 anchor |

每个输出 joint 都是：

```text
[x_normalized, y_normalized, confidence]
```

所以单样本最终为：

```python
{
    "body":     Tensor[T,  9, 3],
    "left":     Tensor[T, 21, 3],
    "right":    Tensor[T, 21, 3],
    "face_all": Tensor[T, 18, 3],
}
```

## 7. 坐标归一化与置信度过滤

置信度阈值固定为：

```text
threshold = 0.3
```

### 7.1 Body

在所选全部帧和 9 个 body joints 中，收集 `score > 0.3` 的 `(x,y)`：

```text
xmin, xmax, ymin, ymax
scale = max(xmax - xmin, ymax - ymin)
```

使用覆盖整个片段的正方形范围，把坐标映射到 `[-1,1]`。

如果有效 body 坐标少于 4 个，或者 `scale == 0`，整个样本的归一化
body 输出为零。

### 7.2 Hands 和 face

```text
left hand  = (joint - left wrist) / body_scale
right hand = (joint - right wrist) / body_scale
face       = (joint - face anchor) / body_scale
```

坐标随后裁剪到 `[-1,1]`。

对于所有部位：

```text
score <= 0.3 -> 整个 [x,y,score] 设为 0
```

OpenASL 与 How2Sign 使用完全相同的四部位选择和归一化代码。

## 8. Batch collate 格式

同一 batch 内不同样本的长度可能不同。对每个部位：

1. 找到 batch 内最大长度 `T_max`。
2. 短样本重复自己的最后一帧，补到 `T_max`。
3. 生成 attention mask：有效位置为 1，padding 为 0。

最终 `src_input`：

```python
{
    "body":             Tensor[B,T_max, 9,3],
    "left":             Tensor[B,T_max,21,3],
    "right":            Tensor[B,T_max,21,3],
    "face_all":         Tensor[B,T_max,18,3],
    "attention_mask":   LongTensor[B,T_max],
    "src_length_batch": LongTensor[B],
    "name_batch":       list[str],
}
```

最终 `tgt_input`：

```python
{
    "gt_sentence": list[str],
    "gt_gloss": list[str],
}
```

一个真实 OpenASL 两样本 batch：

```text
原长度: 18, 33

body:           [2,33, 9,3]
left:           [2,33,21,3]
right:          [2,33,21,3]
face_all:       [2,33,18,3]
attention_mask: [2,33]
lengths:        [18,33]
```

由于原始 keypoints 是 `float64`，dataset/collate 的 pose tensor 也是
`float64`。进入模型前，`fine_tuning.py::move_src_input_to_device()` 会按
运行配置转换为：

```text
--dtype bf16 -> torch.bfloat16
--dtype fp16 -> torch.float16
--dtype fp32 -> torch.float32
```

attention mask 和长度 tensor 保持整数类型。

## 9. 进入 ST-GCN 和 mT5 后的 shape

（以下是原始默认路径；ST-GCN 被 KL-VAE 整体替换的新路径见第 14 节。）

四个部位分别经过相同结构的 pose encoder：

```text
[B,T,V,3]
  -> Linear(3,64)
  -> [B,64,T,V]
  -> spatial ST-GCN: 64 -> 128 -> 256
  -> temporal ST-GCN: 256 -> 256，三层
  -> 对 V 个 joints 求平均
  -> [B,T,256]
```

左手和右手共享：

- `Linear(3,64)`
- spatial ST-GCN
- temporal ST-GCN

在 temporal ST-GCN 前，模型还把 body 的对应 anchor feature 加入
left、right 和 face feature。

四个部位输出拼接：

```text
body      [B,T,256]
left      [B,T,256]
right     [B,T,256]
face_all  [B,T,256]
          |
          v
concat    [B,T,1024]
          |
          v
Linear(1024,768)
          |
          v
pose embeddings [B,T,768]
```

OpenASL 和 How2Sign 都使用英文 prompt：

```text
Translate sign language video to English:
```

Prompt embedding 放在 pose embedding 前面：

```text
[prompt embeddings, pose embeddings]
```

随后作为 `inputs_embeds` 输入 mT5 encoder-decoder。目标英文句子最多
tokenize 到 50 tokens，padding label 设为 `-100`，训练使用带
label smoothing 的 cross-entropy。

## 10. 完整数据流

```text
labels.{split} (gzip pickle)
        |
        | name / video_path / text
        v
<video_path>.mp4 -> <pose_path>.pkl
        |
        | keypoints [T,1,133,2]
        | scores    [T,1,133]
        v
最多随机抽取 256 帧
        |
        v
RTMPose 133 joints
        |
        +--> body     [T, 9,3]
        +--> left     [T,21,3]
        +--> right    [T,21,3]
        `--> face_all [T,18,3]
        |
        | batch 内重复最后一帧 padding
        v
[B,T_max,V,3] + attention_mask
        |
        v
四路 ST-GCN
        |
        v
pose embeddings [B,T_max,768]
        |
        | prepend English prompt embeddings
        v
mT5 encoder-decoder
        |
        v
English translation + cross-entropy loss
```

## 11. OpenASL 与 How2Sign 的核心区别

| 项目 | OpenASL | How2Sign |
|---|---|---|
| Label splits | train/dev/test | train/test |
| Pose 目录 | 三个 split 共用平铺目录 | train/test 分目录 |
| Raw joints | RTMPose 133 | RTMPose 133 |
| Raw keypoint shape | `[T,1,133,2]` | `[T,1,133,2]` |
| Raw score shape | `[T,1,133]` | `[T,1,133]` |
| 额外字段 | 通常无 | `w_h [2]`，当前未使用 |
| Loader 输出 | 四个 `[T,V,3]` 局部 | 完全相同 |
| Prompt | English | English |
| Pose encoder | 相同 | 相同 |
| 目标 | English text | English text |

因此，OpenASL 预训练得到的整个 ST-GCN + mT5 checkpoint 可以通过
`strict=True` 加载到 How2Sign：两者在进入模型后的输入 schema 和模型结构
完全一致，变化的主要是样本内容、数据量和 split。

## 12. 当前需要注意的限制

1. **How2Sign 没有独立 dev**：当前代码用 test 选择 best checkpoint。
2. **长序列随机采样**：验证和测试也会调用同一个 `load_pose()`，所以超过
   256 帧的样本在评估时也可能随机变化。
3. **Raw pose 是 float64**：必须在送入模型前显式转换到训练 dtype。
4. **`w_h` 未使用**：pose-only 路径不依赖图像宽高。
5. **当前没有 RGB 输入**：只要不传 `--rgb_support`，loader 不读取原始视频。
6. **这里不是 133D SMPL-X motion**：不能直接接入 How2Sign VAE/motion
   generation loader。

## 13. 主要代码位置

```text
config.py
  label 路径和 pose 根目录

datasets.py
  load_part_kp()       133 点拆成 body/left/right/face
  crop_scale()         坐标归一化和置信度过滤
  Base_Dataset.collate_fn()
                       padding、mask 和 batch
  S2T_Dataset.load_pose()
                       pose pickle 加载与最多 256 帧采样

fine_tuning.py
  move_src_input_to_device()
                       float64 -> bf16/fp16/fp32，移动到 GPU

models.py
  Uni_Sign.forward()   四路 ST-GCN、特征拼接、mT5 和 loss
```

## 14. 新增：ST-GCN 整体替换为 pose2d KL-VAE（2026-08）

新增了一条加法式路径，把第 9 节的四路 ST-GCN 编码器**完整替换**为在
MoMADiff 中用 How2Sign+OpenASL 联合训练的 138D pose2d KL-VAE
（`MoMADiff/configs/klvae_h2s_openasl_pose2d.yaml`），VAE latent 经
**单层 Linear head** 直接投射到 mT5 embedding space。原始路径不受影响，
仍是默认。

### 14.1 数据输入：仅抽帧策略不同

第 2–4、6–8 节（label、pose pickle、四部位拆分、归一化、collate）对新
路径**原样适用**。唯一不同是第 5 节的抽帧：新路径使用
`datasets_pose2d_klvae.S2T_Dataset_Pose2D_KLVAE`，默认
`--frame_sampling vae_linear` —— 与 VAE 语料构建规则一致的**确定性
线性索引重采样**，把片段重采样进 `[40, max_length]`（区间内的片段整段
保留）。原始的随机非均匀抽帧会改变局部速度和时间连续性（时间卷积
encoder 对此敏感），保留为 `--frame_sampling unisign_random` 供消融。
线性重采样同时使评估变为确定性（原路径长片段评估会随 run 变化）。

### 14.2 模型内的新数据流

```text
src_input（第 8 节的 batch 格式）
        |
        | 取 body/left/right/face_all 的 xy（丢弃 confidence 通道）
        | 按 body(9) | left(21) | right(21) | face_all(18) 顺序摊平
        v
[B,T,138]   （69 joints x [x,y]，与 MoMADiff utils/pose2d_repr.py 同构）
        |
        | z-score：kl_vae_h2s_openasl_pose2d/meta/{mean,std}.npy
        | 按每个样本真实长度截断到 4 的倍数，同长度分组
        v
KL-VAE encoder + quant_conv（**始终冻结**，torch.no_grad 前向，与
MoMADiff checkpoint 数值恒等；**逐样本编码**——VAE 是非因果时间卷积，
若整批带 repeat-last padding 编码，padding 会回流污染有效 latent，且
同一样本的表示会依赖 batch 内其他样本的长度）
        |
        v
posterior mean [T_i/4,512]，再 pad 成 [B,T_max/4,512]
        |
        v
单层 Linear(512,768) head   <-- 替换原 Linear(1024,768) pose_proj
        |
        v
pose embeddings [B,T/4,768]
        |
        | prepend 同样的英文 prompt embeddings
        v
mT5 encoder-decoder（objective/generate 与原路径一致）
```

`attention_mask` 相应从逐帧下采样到 latent 步：`帧长 // 4`（最小 1）。

### 14.3 与 VAE 训练特征的对齐

`datasets.load_part_kp` 与 MoMADiff `rtmpose_to_pose2d` 的数学完全相同
（body 正方形 bbox 归一到 [-1,1]、手减腕、face 减 53 号 anchor、除以
body scale、conf<=0.3 整点置零、clip 到 [-1,1]），部位顺序也一致，
因此 dataset 输出摊平后就是 VAE 的训练分布。时间维上，默认的
`vae_linear` 抽帧复用了 VAE 语料的线性重采样规则（见 14.1），消除了
随机抽帧带来的局部速度/连续性偏移。残余差异只剩 bbox 的计算时机
（Uni-Sign 在重采样后的帧上算，VAE 语料在全帧上算后再重采样）——线性
重采样均匀覆盖整个片段，bbox 极值几乎一致。

### 14.4 文件与用法

| 文件 | 说明 |
|---|---|
| `models_pose2d_klvae.py` | `Uni_Sign_Pose2D_KLVAE`（摊平 → z-score → 逐样本 KL-VAE → 单 head → mT5） |
| `datasets_pose2d_klvae.py` | `S2T_Dataset_Pose2D_KLVAE`：VAE 语料一致的线性重采样抽帧 |
| `fine_tuning_pose2d_klvae.py` | 训练/评估入口，复用 `fine_tuning.py` 的 train/evaluate |
| `vae_encoder.py` | 泛化为可配输入维度（默认 133 不变，此路径传 138） |
| `script/train_openasl_then_how2sign_klvae.sh` | 自动 pipeline：OpenASL 预训练（`--eval-on-how2sign-test` 选点）→ How2Sign 微调 → 最终 eval |
| `script/train_pose2d_klvae.sh` | 手动分阶段命令 |
| `docs/POSE2D_KLVAE_T5.md` | 该路径的完整英文文档 |

新增 CLI 参数：`--vae_checkpoint`（必填，MoMADiff stage-1 checkpoint，
`net_best_hand.pth` 或 lightning `.ckpt` 均可）、
`--vae_mean_path`/`--vae_std_path`（默认指向
`kl_vae_h2s_openasl_pose2d/meta/`）、
`--frame_sampling vae_linear|unisign_random`（默认 `vae_linear`，见
14.1）。VAE **始终冻结**（只训 head + mT5），因此本路径保存的
checkpoint 只含 head + mT5（+ mean/std buffer），`--finetune` 以
`strict=False` 加载，VAE 权重永远来自 `--vae_checkpoint`。

限制：仅支持 `--dataset How2Sign|OpenASL`、`--task SLT`、
`--llm_type mt5`，不支持 `--rgb_support`。该路径的 checkpoint 与原
ST-GCN checkpoint 参数族不同，互相不能 `strict=True` 加载；OpenASL 预训练
的新路径 checkpoint 可以 `strict=True` 加载到 How2Sign 新路径（结构一致，
与第 11 节结论同理）。
