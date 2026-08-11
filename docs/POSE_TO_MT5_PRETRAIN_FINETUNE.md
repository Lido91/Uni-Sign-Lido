# Uni-Sign：2D Pose 如何输入 mT5，以及 Pre-training / Fine-tuning 流程

本文基于当前 checkout 中的 `datasets.py`、`models.py`、
`fine_tuning.py`、`config.py` 和训练脚本，说明 Uni-Sign 如何把 RTMPose
二维关键点转换成 mT5 可以接收的连续 embedding，以及预训练、微调之间
究竟有什么区别。

> **名称说明**：代码实际使用的是 **mT5-Base**
> (`MT5ForConditionalGeneration`)，不是原始英文 T5。下文统一写作 mT5。

主要代码入口：

- [`datasets.py`](../datasets.py)：读取 pose、抽帧、选点、归一化、padding。
- [`models.py`](../models.py)：ST-GCN pose encoder、投影层、mT5 输入和 loss。
- [`fine_tuning.py`](../fine_tuning.py)：当前存在且可追踪的训练/评估入口。
- [`config.py`](../config.py)：mT5、label、pose 和 RGB 路径。
- [`script/train_stage1.sh`](../script/train_stage1.sh)：当前本地 OpenASL pose-only 预训练脚本。
- [`script/train_openasl_then_how2sign.sh`](../script/train_openasl_then_how2sign.sh)：当前本地 OpenASL → How2Sign 流程。

## 1. 一句话总结

Uni-Sign **不会把 2D pose 转成 mT5 词表里的 token ID**。它先用四个
part-aware ST-GCN 分别编码 body、left hand、right hand 和 face，然后把
每帧 pose 投影成一个 768 维连续向量；这些向量作为 `inputs_embeds`，接在
文本 prompt embedding 后面送入 mT5 encoder：

```text
RTMPose 133 joints
        ↓ 选取并归一化 69 joints
4 个 ST-GCN 分支：body / left / right / face
        ↓
每帧 4 × 256 = 1024 维
        ↓ Linear(1024, 768)
每帧一个 pose embedding：[B, T, 768]
        ↓ 与文本 prompt 沿时间/token 维拼接
[B, P + T, 768]
        ↓ inputs_embeds
mT5 encoder → mT5 autoregressive decoder → 目标句子
```

这里 `B` 是 batch size，`T` 是 batch 内 padding 后的 pose 帧数，`P` 是
prompt 的 mT5 token 数。

## 2. 磁盘上的 2D pose 输入

OpenASL/How2Sign 的 pose pickle 采用 RTMPose COCO-WholeBody 格式：

```python
pose["keypoints"]  # [T_raw, 1, 133, 2]
pose["scores"]     # [T_raw, 1, 133]
```

- `T_raw`：原始帧数。
- `1`：一个被检测到的人。
- `133`：COCO-WholeBody 的 133 个二维关键点。
- `2`：图像平面坐标 `(x, y)`。
- `scores`：每个关键点的检测置信度。

这个“133”表示 **133 个二维 joints**，不是 motion-generation/VAE 代码中
每帧 `[133]` 的 SMPL-X motion feature。

### 2.1 时间采样

默认 `--max_length 256`：

```text
T_raw <= 256：保留全部帧
T_raw >  256：随机抽取 256 个不同帧，再排序以保持时间顺序
```

因此长视频采用的是随机稀疏抽帧，不是插值，也不是固定间隔重采样。

### 2.2 从 133 点选出 69 点

`load_part_kp()` 把 whole-body pose 分为四个分支：

| 分支 | RTMPose 索引 | 点数 | 单样本 shape |
|---|---:|---:|---:|
| body | `[0] + [3, ..., 10]` | 9 | `[T, 9, 3]` |
| left | `[91:112]` | 21 | `[T, 21, 3]` |
| right | `[112:133]` | 21 | `[T, 21, 3]` |
| face_all | 9 个脸部轮廓点 + 8 个嘴部点 + 点 53 | 18 | `[T, 18, 3]` |

每个点最终不是只有 `(x, y)`，而是：

```text
[normalized_x, normalized_y, confidence]
```

所以最后一维是 3；总点数为 `9 + 21 + 21 + 18 = 69`。

### 2.3 坐标归一化与低置信度 masking

代码使用置信度阈值 `0.3`：

- body 用整个序列的有效 body 点计算方形范围，并归一化到约 `[-1, 1]`；
- left/right hand 先减去各自 wrist 点，变成 wrist-relative 坐标；
- face 减去选中 face 点中的最后一个 anchor；
- hand 和 face 再除以 body 得到的尺度并 clip 到 `[-1, 1]`；
- confidence `<= 0.3` 的点，三个通道全部置零。

### 2.4 Batch padding 和 mask

一个 batch 内，不同长度的 pose 会 padding 到该 batch 的最长长度。padding
不是补零，而是重复该样本的最后一帧：

```text
body:       [B, T,  9, 3]
left:       [B, T, 21, 3]
right:      [B, T, 21, 3]
face_all:   [B, T, 18, 3]
attention_mask: [B, T]
```

重复出来的 padding 帧仍会经过 ST-GCN，但对应 `attention_mask` 为 0，进入
mT5 后不会作为有效 encoder token 被 attention 使用。

## 3. 2D pose encoder：从 joints 到每帧 1024 维

四个 part 分别进入 ST-GCN。每个 part 首先执行：

```text
[B, T, V, 3]
    ↓ Linear(3, 64)
[B, T, V, 64]
    ↓ permute
[B, 64, T, V]
```

其中 `V` 分别是 9、21、21 和 18。

### 3.1 Spatial ST-GCN

每个 part 使用自己的 skeleton graph，空间 GCN 通道变化为：

```text
64 → 64 → 128 → 256
```

输出：

```text
[B, 256, T, V]
```

图邻接矩阵由对应 part 的骨架连接初始化，并设置为 adaptive，也就是在训练
过程中可以学习。

### 3.2 左右手共享参数

代码显式让 left/right hand 共享：

- `Linear(3, 64)`；
- spatial ST-GCN；
- temporal ST-GCN。

因此左右手是两个输入分支，但不是两套独立参数。

### 3.3 Body feature 注入其他 part

进入 temporal encoder 前：

```text
left hand  += detached body left-wrist feature
right hand += detached body right-wrist feature
face       += detached body head/nose feature
```

这里使用 `.detach()`，所以这三条跨 part 注入不会通过该连接反向更新 body
分支。

### 3.4 Temporal ST-GCN 与 joint pooling

每个 part 再经过三个保持 256 通道的 temporal ST-GCN block，时间卷积 kernel
为 5：

```text
[B, 256, T, V]
    ↓ temporal ST-GCN × 3
[B, 256, T, V]
    ↓ 对 V 个 joints 求平均
[B, 256, T]
    ↓ transpose
[B, T, 256]
```

最后，四个 `[B, T, 256]` 沿 **feature 维** 拼接：

```text
cat([body, left, right, face], dim=-1)
    → [B, T, 1024]
```

`models.py` 的注释写了 “token dimension”，但真实代码是 `dim=-1`，所以
这里拼接的是 feature，不是时间/token 数；`T` 保持不变。

模型还会加上一个可学习、初始为零的 `[1024]` 向量 `part_para`。它广播到
所有 batch 和 frame。

## 4. Pose 如何真正输入 mT5

### 4.1 投影到 mT5-Base hidden size

`pose_proj` 把每帧的 1024 维 pose feature 投影到 mT5-Base 的 768 维
embedding space：

```python
self.pose_proj = nn.Linear(256 * 4, 768)
```

```text
[B, T, 1024] → [B, T, 768]
```

此时每个 pose frame 相当于一个连续的 “pose token”，但它没有 token ID，
也没有进入 SentencePiece tokenizer。

### 4.2 文本 prompt

prompt 根据数据集语言构造：

```text
CSL_* 数据集：Translate sign language video to Chinese:
其他数据集： Translate sign language video to English:
```

prompt 正常经过 mT5 tokenizer 和 encoder embedding table：

```text
prompt token IDs → [B, P, 768]
```

然后与 pose embedding 沿 sequence 维拼接：

```text
prompt embeddings [B, P, 768]
pose embeddings   [B, T, 768]
        ↓ cat(dim=1)
inputs_embeds     [B, P + T, 768]
```

mask 同样拼接为：

```text
prompt attention mask [B, P]
pose attention mask   [B, T]
        ↓
combined mask         [B, P + T]
```

最终调用方式是：

```python
self.mt5_model(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    labels=labels,
)
```

也就是说，pose 直接作为 mT5 encoder 的连续输入；随后仍然使用完整的 mT5
encoder-decoder，而不是把 mT5 当成 decoder-only 模型。

## 5. Target 和训练 loss

对于 SLT，target 是 label 中的 `text`。target 经过 mT5 tokenizer：

```text
最大 target 长度：50 tokens
padding token：替换为 -100
```

mT5 decoder 使用 teacher forcing 产生 `[B, L, vocab_size]` logits。代码最终
重新计算 token-level cross entropy：

```python
CrossEntropyLoss(
    label_smoothing=args.label_smoothing,
    ignore_index=-100,
)
```

默认 `label_smoothing=0.2`。如果设置 `--compute_fp32_loss`，logits 会在 loss
计算前转为 FP32。

当前代码没有冻结 mT5、ST-GCN 或 `pose_proj`。因此正常训练时，下列模块会
一起 end-to-end 更新：

```text
ST-GCN pose encoder
pose_proj: 1024 → 768
mT5 encoder
mT5 decoder
```

## 6. “Pre-training”和“Fine-tuning”的本质区别

在这份代码里，pre-training 和 fine-tuning 并没有不同的 pose→mT5 forward
或不同的 loss。二者都执行 supervised sign-to-text generation：

```text
pose/RGB + target text → token-level cross entropy
```

主要差别是：

| 阶段 | 数据 | 初始化 | 模态 | 目的 |
|---|---|---|---|---|
| Sign pre-training | 大规模上游 sign-text 数据 | mT5 已有语言预训练；pose encoder 随机 | pose-only 或 RGB-pose | 学会 sign feature，并对齐到 mT5 空间 |
| Downstream fine-tuning | 目标数据集 | 加载 sign-pretrained checkpoint | 必须与 checkpoint 结构兼容 | 适配目标数据和任务 |

所以“没有 sign pre-training”也不代表整个模型完全随机：mT5 仍由
`./pretrained_weight/mt5-base` 初始化，只是 ST-GCN 和 `pose_proj` 没经过
上游 sign-text 训练。

## 7. 官方三阶段设计

仓库 README 和原始 stage 脚本描述的是：

```text
Stage 1: CSL-News pose-only pre-training
       ↓ out/stage1_pretraining/best_checkpoint.pth
Stage 2: CSL-News RGB-pose pre-training
       ↓ out/stage2_pretraining/best_checkpoint.pth
Stage 3: downstream fine-tuning
```

### Stage 1：pose-only pre-training

- 不传 `--rgb_support`；
- 随机初始化 ST-GCN 和 `pose_proj`；
- 从本地 mT5-Base 初始化语言模型；
- 所有模块一起训练；
- dev BLEU-4 最高的权重保存为 `best_checkpoint.pth`。

### Stage 2：RGB-pose pre-training

- 从 Stage-1 pose-only checkpoint 初始化；
- 增加 `--rgb_support`，启用 EfficientNet-B0 hand RGB、PGF deformable
  attention 和 fusion gate；
- pose-only checkpoint 没有新增 RGB 模块的参数，所以历史
  `pre_training.py` 使用 `strict=False` 加载 Stage-1 checkpoint；
- 已有 pose/mT5 参数继续训练，新增 RGB 参数从初始化状态开始训练。

### Stage 3：downstream fine-tuning

- pose-only downstream：从 Stage-1 checkpoint 初始化，不传
  `--rgb_support`；
- RGB-pose downstream：从 Stage-2 checkpoint 初始化，并传
  `--rgb_support`；
- 当前 `fine_tuning.py` 使用 `strict=True`，因此 checkpoint 和当前模型
  结构必须完全匹配。

## 8. 当前工作区实际使用的 OpenASL → How2Sign 流程

当前本地脚本已经偏离原始 CSL-News 三阶段脚本。

### 8.1 当前 Stage 1

[`script/train_stage1.sh`](../script/train_stage1.sh) 实际调用：

```text
fine_tuning.py
dataset = OpenASL
task = SLT
rgb_support = False
checkpoint initialization = 无
```

因此它在功能上是 **OpenASL pose-only sign pre-training**。输出目录为：

```text
out/stage1_pretraining_openasl/
```

脚本使用 3 张 GPU、per-GPU batch size 16、gradient accumulation 8；忽略
最后一个不足 batch 的部分后，名义 effective batch size 为：

```text
16 × 3 × 8 = 384
```

### 8.2 当前 How2Sign fine-tuning

[`script/train_openasl_then_how2sign.sh`](../script/train_openasl_then_how2sign.sh)
从下面的 checkpoint 初始化：

```text
out/stage1_pretraining_openasl/best_checkpoint.pth
```

然后在 How2Sign 上继续 pose-only SLT：

```text
fine_tuning.py
dataset = How2Sign
task = SLT
rgb_support = False
```

这里 OpenASL 与 How2Sign 使用相同的四分支 pose-only 模型，所以
`strict=True` 的结构是兼容的。

## 9. 每个 epoch 内发生什么

`fine_tuning.py` 的训练流程为：

1. `S2T_Dataset` 读取 label 和 pose pickle；
2. `DistributedSampler` 为各 GPU 分配训练样本；
3. collate 得到四个 part tensor、pose mask 和 target text；
4. tensor 按 DeepSpeed dtype 移到 GPU；
5. `Uni_Sign.forward()` 构建 `[prompt; pose]` mT5 输入；
6. 计算带 label smoothing 的 token-level cross entropy；
7. `model.backward(loss)` 和 `model.step()`；
8. cosine scheduler 随 optimizer update 推进；
9. 每个 epoch 保存 checkpoint，并生成句子做评估；
10. SLT 按 BLEU-4 选择 `best_checkpoint.pth`。

DeepSpeed 配置由 CLI 参数动态创建。默认使用 ZeRO stage 2；训练精度由
`--dtype fp16|bf16|fp32` 控制。

## 10. Checkpoint 内容与加载规则

保存格式为：

```python
{
    "model": model_state_dict
}
```

不包含 optimizer、scheduler 或当前 epoch。因此这里的 checkpoint 适合做
权重初始化或 inference，但不是一个完整的 optimizer-state resume checkpoint。

当前 fine-tuning 加载逻辑：

```python
state_dict = torch.load(args.finetune, map_location="cpu")["model"]
model.load_state_dict(state_dict, strict=True)
```

必须遵守：

```text
pose-only checkpoint → pose-only model：兼容
RGB-pose checkpoint  → RGB-pose model：兼容
pose-only checkpoint → RGB-pose model：当前 fine_tuning.py 不兼容
RGB-pose checkpoint  → pose-only model：当前 fine_tuning.py 不兼容
```

## 11. 当前 checkout 的可运行性说明

截至本文编写时，本机观察到：

- `./pretrained_weight/mt5-base` 存在，约 2.2 GB，包含
  `pytorch_model.bin`、`config.json`、tokenizer 文件；
- 当前工作区没有 `data/`、`dataset/` 和 `out/` 目录；
- `config.py` 虽然配置了相对数据路径，但当前磁盘上没有对应数据；
- `pre_training.py` 在当前 working tree 和 `HEAD` 中均不存在；它只存在于
  较早的 git 历史版本；
- 当前 [`script/train_stage2.sh`](../script/train_stage2.sh) 仍调用缺失的
  `pre_training.py`，而且使用小写 `--dataset openasl`，不符合 parser 允许的
  `OpenASL`，因此该脚本当前不能直接运行；
- `script/train_stage3.sh` 默认依赖尚不存在的
  `out/stage2_pretraining/best_checkpoint.pth`。

因此，当前 checkout 中能够完整追踪的训练入口是 `fine_tuning.py`，但正式
启动前仍需要恢复或挂载 `config.py` 所指向的数据。

## 12. 评估与数据泄漏注意事项

当前 `script/train_stage1.sh` 带有：

```text
--eval-on-how2sign-test
```

这会用 **How2Sign test BLEU-4** 选择 OpenASL pre-training 的
`best_checkpoint.pth`。此外，How2Sign 在当前 config 中没有 dev split，
`fine_tuning.py` 会令 `dev_dataloader = test_dataloader`，所以 How2Sign
fine-tuning 也按 test BLEU-4 选择 best checkpoint。

这两种行为都会把 test set 用于模型选择。如果结果要用于正式论文比较，
应准备独立 validation split，并只在训练与超参数选择全部结束后运行一次
test evaluation。

## 13. Shape 总表

| 位置 | Shape |
|---|---:|
| 原始坐标 | `[B, T_raw, 1, 133, 2]` |
| 原始置信度 | `[B, T_raw, 1, 133]` |
| body | `[B, T, 9, 3]` |
| left hand | `[B, T, 21, 3]` |
| right hand | `[B, T, 21, 3]` |
| face | `[B, T, 18, 3]` |
| part 输入投影 | `[B, 64, T, V]` |
| part spatial/temporal ST-GCN 输出 | `[B, 256, T, V]` |
| 每个 part joint pooling 后 | `[B, T, 256]` |
| 四个 part feature 拼接 | `[B, T, 1024]` |
| `pose_proj` 后 | `[B, T, 768]` |
| prompt embedding | `[B, P, 768]` |
| mT5 encoder 输入 | `[B, P + T, 768]` |
| decoder logits | `[B, L, vocab_size]` |

## 14. 最容易混淆的几点

1. Pose 不会被离散化为 mT5 token ID，而是通过 `inputs_embeds` 直接进入
   mT5 encoder。
2. 四个 part 是在 feature 维拼接，所以一个 frame 对应一个 pose token，
   不是四个 pose tokens。
3. mT5 在 sign pre-training 和 fine-tuning 中都没有被冻结。
4. 当前代码中的 pre-training 与 fine-tuning 使用相同的 forward 和 loss；
   “pre-training”主要表示先在较大的上游 sign-text 数据集训练。
5. `--hidden_dim` 实际必须保持 256：`part_para` 依赖它，而 ST-GCN 输出和
   `pose_proj` 又硬编码为 `4 × 256 → 768`。
6. 是否传 `--rgb_support` 会改变模型结构，因此也决定 checkpoint 能否用
   `strict=True` 加载。
