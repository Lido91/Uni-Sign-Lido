# SignJanus 联合训练框架说明

本文说明当前 Uni-Sign checkout 中的 **SLT + 手语姿态生成（T2P/SLG）+ Geo-Sign** 联合训练实现，重点对应推荐入口 `script/train_unified_merged_geosign.sh`。这是 Janus 的“理解/生成分路、共享语言主干”思路在 Uni-Sign 上的 sign-specific 实现，并结合 MoMADiff 的 masked latent diffusion 和 Geo-Sign 的双曲对比目标；它不是 Janus、MoMADiff 或 Geo-Sign 的原样复现。

## 一句话概括

模型用同一套 mT5 编码器支持两个任务：输入 2D 手语 pose 预测文本（SLT），以及输入英文句子生成 2D 手语 pose（T2P）。SLT 的交叉熵与 Geo-Sign 双曲对比损失先合成 SLT loss，再与 T2P 的 masked latent diffusion loss 相加。推荐的合并入口把 OpenASL 和 How2Sign 训练样本拼成一个自然比例的数据池，并按 Stage 3 的解冻规则训练 40 epochs；KL-VAE 始终冻结。

## 数据与采样

模型训练使用文本和 RTMPose-2D pose，不直接输入 RGB 视频。当前数据配置为：

| 数据集 | 训练标签 | pose 目录 | 仓库盘点的训练样本数 |
|---|---|---|---:|
| OpenASL | `data/OpenASL/labels.train` | `data/OpenASL/pose-rtmpose-192/` | 96,476 |
| How2Sign | `data/How2Sign/labels.train` | `data/How2Sign/train-pose-rtmpose-192/` | 31,086 |

`datasets_merged.py` 用 `ConcatDataset` 合并索引，不复制或搬动 pose 文件；两个子数据集保留各自的数据根目录和预处理。训练时使用普通 `DistributedSampler(shuffle=True)`，因此来源比例由数据量决定，约为 **OpenASL 75.6% / How2Sign 24.4%**，不是两个数据集 1:1 采样。样本 ID 会带上来源前缀以避免跨数据集重名。

合并后会分别构建一个 SLT loader 和一个 T2P loader。二者各自打乱，因此一个优化步骤中的 SLT batch 与 T2P batch **不保证来自相同样本**；它们共享相同的合并语料来源比例。每个优化步骤各取一个 SLT batch 和一个 T2P batch，按两个任务的权重共同更新模型。

```text
 OpenASL train (text + 2D pose)      How2Sign train (text + 2D pose)
                 +-------------------+
                           |
       ConcatDataset + DistributedSampler (natural 75.6/24.4 mix)
                           |
             +-------------+-------------+
             |                           |
        SLT batch                    T2P batch
             |                           |
     pose parts + confidence       GT pose -> frozen KL-VAE encoder
             |                           |
          ST-GCN                  512-D latents -> cosine masking
             |                           |
      pose projection          text embeds + masked latent slots
             |                           |
             +----------+   +------------+
                        |   |
                        v   v
                 shared mT5 encoder
                   /             \
                  /               \
     SLT decoder + Geo-Sign     latent-slot conditions
       hyperbolic branch              |
          /       \             LDM/MLP diffusion head
         v         v                   |
   CE loss      Geo loss        masked diffusion loss
         \         /                   /
          +-------+-------------------+
                  |
  weighted SLT loss + weighted T2P loss (one joint backward)
                  |
       gradient accumulation + cross-rank synchronization
                  |
     AdamW (Euclidean) + RiemannianAdam (manifold)
                  |
           updated unified model
                  |
   each epoch: evaluate OpenASL test and How2Sign test separately
                  |
  best_checkpoint: How2Sign BLEU-4; best_slg: How2Sign hand DTW
```

## 模型数据流

### SLT：pose 到文本

1. 输入按 body、左手、右手、face 等部位组织的 2D 关键点与置信度。
2. Uni-Sign 的 ST-GCN 提取部位 pose 特征，投影后与翻译 prompt embedding 拼接。
3. 共享 mT5 编码器—解码器生成目标文本，以 token-level cross-entropy（CE）监督。
4. Geo-Sign 分支将 body、左右手和 face pose 表征投影到 Poincaré ball；文本 token 表征也投影到该空间。双曲距离用于 pose-part 到文本 token 的注意力和对比目标。

Geo-Sign 分支只改变 SLT 目标；它不替换 T2P 的扩散生成器，也没有给生成路径另加一个 pose-text 对比损失。

### T2P/SLG：文本到 pose

1. 目标 pose 按部位拼成每帧 138 维的 XY 表示，使用预先训练的 KL-VAE 编码成 512 维连续 latent；编码器冻结，训练时从 posterior 采样。
2. 随机遮挡有效 latent 序列中的一部分位置。文本 token 与 latent slots 一起输入同一 mT5 encoder；被遮挡位置由可训练的 mask embedding 表示。
3. mT5 输出的逐 latent 条件向量交给 LDM/MLP diffusion head。扩散训练损失只在本次被 mask 的有效位置上统计。
4. 推理时通过逐步缩小的 MAR mask schedule 生成 latent，再由冻结 KL-VAE decoder 解码、反归一化并还原成 body/hand/face pose 部位。

验证生成采用 GT-length oracle：当前样本的真实 pose 帧数传入 `generate_pose`，输出长度按 VAE 的下采样倍率对齐。因此，当前验证不测文本到动作时长的预测能力；自由生成所需的 text-to-length predictor 尚未接入。

## 联合目标与参数更新

Geo-Sign 的 SLT 目标为：

\[
L_{\mathrm{SLT}}=\alpha_t L_{\mathrm{CE}}+(1-\alpha_t)L_{\mathrm{Geo}}.
\]

在当前入口配置中，`--alpha 0.7`，并采用 Geo-Sign 原实现中的阶段进度与可学习系数调度，最终将 \(\alpha_t\) 限制在 `[0.1, 1.0]`。联合目标为：

\[
L_{\mathrm{joint}}=\lambda_{\mathrm{SLT}}L_{\mathrm{SLT}}+\lambda_{\mathrm{T2P}}L_{\mathrm{diff}}.
\]

当前两个 loss weight 默认均为 1.0。实现会在一个 DDP-wrapped forward 中计算两个任务，再对加权总损失做一次 backward；这样共享的 mT5 参数在一个训练步骤里接收两个任务的梯度。梯度累积达到边界时，普通欧氏参数由 AdamW 更新，流形参数由独立的 RiemannianAdam 更新。代码也执行跨 rank 梯度同步和非有限梯度检查。

## 阶段与推荐入口

基础训练有三个解冻阶段：

| 阶段 | 训练参数 | 冻结参数 |
|---|---|---|
| Stage 1：生成 warmup | latent adapter、mask embedding、输出条件头、diffusion head | mT5、ST-GCN/pose projection、VAE |
| Stage 2：统一训练 | 上述生成模块 + mT5 + Geo-Sign 分支；SLT 与 T2P 联合 | ST-GCN/pose projection、VAE |
| Stage 3：联合微调 | ST-GCN/pose projection + mT5 + 生成模块 + Geo-Sign 分支 | VAE |

推荐合并入口 `script/train_unified_merged_geosign.sh` 是**单段 40-epoch Stage 3-policy 训练**：从 `unified_stage1_openasl/best_checkpoint.pth` 初始化，不要求先跑一个新的 Stage 2；由于该初始化没有 Geo-Sign 参数，入口允许新双曲分支随机初始化。`--slt_init` 先装载 SLT 权重，随后 `--finetune` 装载 unified Stage 1 权重并覆盖同名参数。VAE 权重和统计量从独立路径加载，不保存在 unified checkpoint 中。

如果希望采用渐进解冻，也可以用合并数据先跑 Stage 2，再从它的 checkpoint 跑合并 Stage 3；那是两个各 40 epochs 的训练阶段，不是推荐单段入口的默认行为。

## 默认运行配置

脚本默认使用物理 GPU `1,2,3`、FP32、DeepSpeed ZeRO-0、40 epochs、学习率 `3e-5`、每卡 batch 8、梯度累积 8，并启用 W&B 项目 `uni-sign-unified`。这对应三卡有效 batch：`3 × 8 × 8 = 192`。若单卡显存紧张，可用每卡 batch 4、累积 16，保持相同的有效 batch，同时降低单步激活占用：

```bash
cd /home/student/hwu/Workplace/Uni-Sign
GEOSIGN_GPUS=1,2,3 BATCH_SIZE=4 ACCUM_STEPS=16 \
INIT_CHECKPOINT=/data/hwu/Uni_sign_repo/out/unified_stage1_openasl/best_checkpoint.pth \
OUTPUT_DIR=/data/hwu/Uni_sign_repo/out/unified_stage3_merged_geosign \
bash script/train_unified_merged_geosign.sh
```

脚本默认输出目录是 `/data/hwu/Uni_sign_repo/out/unified_stage3_merged_geosign`。W&B run id 写入输出目录，按 epoch 保存完整 DeepSpeed 恢复状态；自动恢复要求保持相同 GPU 数量。脚本也提供 `DRY_RUN=1` 命令打印和 `--smoke_steps N` 短试跑入口。

## 验证与 checkpoint 选择

合并训练当前要求 `--val_split test`。每个 epoch 分别在 OpenASL test 和 How2Sign test 上计算 SLT 指标、T2P diffusion loss，并默认运行完整 text-to-pose 生成，计算置信度遮掩的 DTW-MPJPE2D。手部 checkpoint 指标为左、右手 DTW 分数的平均。

Stage 3 的 `--dataset` 默认是 How2Sign，因此：

- `best_checkpoint.pth` 按 **How2Sign test BLEU-4 最大**选择；
- `best_slg_checkpoint.pth` 按 **How2Sign test hand DTW-MPJPE2D 最小**选择；
- OpenASL 与 How2Sign 的指标分别记录，不拼成一个总体 BLEU 或 DTW 分数。

验证生成直接使用 GT 帧长，且每轮都查看 test 并据此选模型，所以这些 test 分数参与了模型选择，不能再称为未触碰的独立 final-test 结果。需要无偏最终报告时，应在模型选择之外保留独立测试集。Back-translation 不是这段每 epoch 验证的组成部分；它由 `back_translation_eval.py` / `script/eval_backtrans.sh` 单独运行。

## 实现文件

| 文件 | 职责 |
|---|---|
| `script/train_unified_merged_geosign.sh` | 推荐的合并数据 Stage 3 入口 |
| `script/train_unified_geosign.sh` | GPU、训练参数、checkpoint、W&B 与 Geo-Sign 参数配置 |
| `train_unified.py` | 数据加载、阶段冻结、joint forward/backward、验证、选模与恢复 |
| `datasets_merged.py` | OpenASL/How2Sign 自然比例拼接和来源命名 |
| `models_unified.py` | 共享 mT5、SLT/T2P 任务主体与 masked latent diffusion |
| `models_unified_geosign.py` | 在 unified 模型中调用原 Geo-Sign SLT forward |
| `vendor/geosign/` | Geo-Sign 源模型及训练 helper 副本；来源与哈希记录于 `provenance.json` |
| `geosign_training.py` | AdamW/RiemannianAdam、同步、有限性检查和流形优化器恢复 |
| `pose2d_metrics.py` | 2D pose 生成的置信度遮掩 DTW-MPJPE2D |
| `back_translation_eval.py` | 独立的生成结果回译评估 |

数据目录、字段和当前数据量盘点见 [`OPENASL_HOW2SIGN_INPUT_FORMAT.md`](OPENASL_HOW2SIGN_INPUT_FORMAT.md)。实现与运行状态应以本机代码和训练日志为准：单元/CPU 检查或 smoke run 只验证对应覆盖范围，不等于完整 40-epoch、多卡训练和评测已完成。
