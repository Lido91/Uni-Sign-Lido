# Geo-Sign 增强版 Stage 2 / Stage 3

2026-10-14：将本机 `/home/student/hwu/Workplace/Geo-Sign` 的双曲正则接入
Uni-Sign 的 SLT + SLG 联合训练。未启动正式训练，也未完成 CUDA/DeepSpeed 验证。

## 更新：合并两个训练集的推荐入口

按用户后续选择，新增一个自然合并训练入口：

```bash
cd /home/student/hwu/Workplace/Uni-Sign
bash script/train_unified_merged_geosign.sh
```

该入口直接复用现有 Stage 1 checkpoint，随机初始化新增双曲模块，
使用 Stage 3 的解冻规则联合训练 40 epochs：ST-GCN、mT5、生成模块和
双曲模块均训练，VAE 冻结。无需先跑独立的增强 Stage 2。

OpenASL train 96,476 条 + How2Sign train 31,086 条 = **127,562 条**。
SLT 和 SLG 各使用同一批合并样本，通过普通 DistributedSampler 随机打乱；
采样自然约为 75.6% / 24.4%，不对 How2Sign 过采样。
合并是逻辑索引拼接，不复制 label 或 pose；子数据集保留自己的数据根目录与
预处理。训练样本 ID 添加来源前缀，避免不同数据集同名 ID 覆盖。
沿用原 sampler 的分布式 padding 和训练 drop_last 规则。

两套 test 分别评测，W&B 记录 `test/openasl/...` 与 `test/how2sign/...`。
默认仍以 How2Sign test BLEU-4 选择 `best_checkpoint.pth`，以 How2Sign hand DTW
选择 `best_slg_checkpoint.pth`；两数据集的 BLEU 不混成一个语料分数。
输出目录为 `/data/hwu/Uni_sign_repo/out/unified_stage3_merged_geosign`。

默认物理 GPU 1、2、3，batch/GPU=8，累积=8，LR=3e-5，FP32/ZeRO-0，W&B 开启。
可沿用环境变量覆盖参数，例如：

```bash
BATCH_SIZE=4 ACCUM_STEPS=16 bash script/train_unified_merged_geosign.sh
```

若仍希望先冻结 ST-GCN 再解冻，两阶段都可合并训练：

```bash
MERGE_DATASETS=1 bash script/train_unified_stage2_geosign.sh
MERGE_DATASETS=1 bash script/train_unified_stage3_geosign.sh
```

此选择默认各 40 epochs，合计 80 个合并集 epochs，训练预算高于上述单段 40 epochs。
它们分别输出 `unified_stage2_merged_geosign` 与 `unified_stage3_merged_geosign`，
Stage 3 继承增强的合并 Stage 2。继续旧的单数据集 resume_state 会被拒绝；
数据变化应使用新目录和权重初始化。

合并训练增加 5 项 CPU 检查，共 **28 项通过**；实际数据样本读取及
pose 文件覆盖检查记录在 `MERGED_ASL_DATA_AUDIT_20261014.json`。
完整 GPU 训练、GPU 评测及多卡恢复仍未验证，未启动训练。

## 原实现如何保留

- `vendor/geosign/models.py` 是源 `models.py` 的逐字节副本，出处和 SHA256
  保存在 `vendor/geosign/provenance.json`。
- `models_unified_geosign.py` 复制原双曲初始化代码块，并直接调用副本的
  `Uni_Sign.forward`。Poincaré 投影、Fréchet mean、token attention、距离、
  margin InfoNCE 和动态 alpha 都执行原函数，不重新实现这些公式。
- 仅补 Uni-Sign loader 到 Geo-Sign `prefix_ids` / `labels_ids` 的接口；
  主干只有一套 ST-GCN / mT5，SLG 沿用现有 VAE / MAR / 扩散路径。
- 原 scalar gradient clipping、外部优化器梯度同步和梯度有限性检查函数
  逐字复制至 `vendor/geosign/training_helpers.py`。
- AdamW 和独立 RiemannianAdam 采用原分组与更新顺序。geoopt 0.5.1 的
  `manifold.isp_c` 仍在 AdamW 中，流形 bias 在 RiemannianAdam 中。

联合目标：

```text
slt_loss = alpha * CE + (1 - alpha) * Geo-Sign contrastive loss
total_loss = slt_loss_weight * slt_loss + t2p_loss_weight * diffusion_loss
```

alpha 沿用原 `clamp(alpha_arg + 0.1*progress + 0.2*sigmoid(loss_alpha_logit), 0.1, 1)`。
新阶段初始化时重置其进度计数；阶段内断点恢复时恢复计数。
这样 Stage 2 的较大更新数不会在 Stage 3 开始时把 alpha 推到 1、关掉对比损失。

## 训练配置

源 CSL-Daily 和 ASL 脚本均配置 40 epochs；本机 Geo-Sign 的 How2Sign
日志也记录至 epoch 39。增强版两阶段默认均为 40 epochs。

| 项目 | Stage 2 | Stage 3 |
| --- | --- | --- |
| 数据 | OpenASL | How2Sign |
| 初始化 | 原 Stage 1 best_checkpoint | 增强 Stage 2 best_checkpoint |
| ST-GCN / pose_proj | 冻结 | 解冻 |
| mT5 / SLG / 双曲分支 | 训练 | 训练 |
| VAE | 冻结 | 冻结 |
| LR | 1e-4 | 3e-5（保留原 Stage 3 设置） |
| epochs | 40 | 40 |
| batch / GPU | 8 | 8 |
| 梯度累积 | 8 | 8 |
| 精度 / ZeRO | FP32 / 0 | FP32 / 0 |
| init_c / alpha_arg | 1.0 / 0.7 | 1.0 / 0.7 |
| 文本特征 / 对比模式 | decoder / token | decoder / token |

采用 Geo-Sign 本机稳定配置 FP32 + ZeRO-0。三卡有效 batch 是
8 × 8 × 3 = 192；对比负样本池仍是每卡单个 microbatch，累积不扩大它。
增强分支有额外 hidden states / 对比图，原版显存试跑结果不能作为其显存保证。

## 启动

```bash
cd /home/student/hwu/Workplace/Uni-Sign
bash script/train_unified_stage2_geosign.sh
# Stage 2 产生 best_checkpoint.pth 后：
bash script/train_unified_stage3_geosign.sh
```

默认使用物理 GPU 1、2、3。增强脚本读取 `GEOSIGN_GPUS`，避免环境中旧
`GPUS=0,1,3` 意外选中已占用的 GPU 0。示例：

```bash
GEOSIGN_GPUS=1,2,3 BATCH_SIZE=4 ACCUM_STEPS=16 \
  bash script/train_unified_stage2_geosign.sh
```

调整 `EPOCHS`、`LR`、`BATCH_SIZE`、`ACCUM_STEPS`、`MASTER_PORT`、
`OUTPUT_DIR` 或 `INIT_CHECKPOINT` 可覆盖默认值。

若想从已训好的原 Stage 2 接着训练增强 Stage 2（而非从 Stage 1 重训）：

```bash
INIT_CHECKPOINT=/data/hwu/Uni_sign_repo/out/unified_stage2_openasl/best_checkpoint.pth \
  bash script/train_unified_stage2_geosign.sh
```

这属于权重初始化：新增双曲分支随机初始化，优化器重新创建，默认再训练 40 轮。
若直接在原 Stage 2 上接增强 Stage 3，显式允许新增参数初始化：

```bash
INIT_CHECKPOINT=/data/hwu/Uni_sign_repo/out/unified_stage2_openasl/best_checkpoint.pth \
  bash script/train_unified_stage3_geosign.sh --allow_plain_geosign_init
```

标准增强 Stage 3 要求初始化 checkpoint 包含双曲分支，避免误加载普通权重。

## 输出、评测和续训

输出分别为：

```text
/data/hwu/Uni_sign_repo/out/unified_stage2_openasl_geosign
/data/hwu/Uni_sign_repo/out/unified_stage3_h2s_geosign
```

保留现有普通实验目录。W&B 默认启用，记录 CE、margin_loss、alpha、曲率、
温度、SLG diffusion loss 和阶段末的评测指标。
每个 epoch 在 test 上评估 SLT BLEU / ROUGE 和置信度遮掩 DTW-MPJPE2D；
`best_checkpoint.pth` 选 SLT BLEU-4 最大，`best_slg_checkpoint.pth` 选 hand DTW 最小。

增强断点保存：模型（含双曲参数和 global_step）、DeepSpeed optimizer / scheduler、
外部 RiemannianAdam 状态。使用相同 GPU 数和配置重跑同一脚本恢复。

## 检查状态

已检查源文件哈希、初始化代码一致性、CPU 小模型的原 forward 输出、有限梯度、
联合损失、冻结规则、双优化器参数分组 / 更新 / 恢复，以及 plain checkpoint 白名单。
另有梯度累积和非有限梯度保护的 CPU 检查。
共 23 项 CPU 测试通过（6 项增强分支检查、17 项原验证 / DTW / 预测保存回归检查）。
脚本语法、Stage 2 路径和 Stage 3 显式 plain 初始化的 dry-run 已检查。

实际 CUDA / 多卡 DeepSpeed 测试未执行；用户要求仅准备代码。
`validation/smoke_geosign_unified.py` 保留了小模型三卡测试入口，未来可手动验证：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
MPLCONFIGDIR=/tmp/unisign-geosign-matplotlib \
TRITON_CACHE_DIR=/tmp/unisign-geosign-triton \
/home/student/hwu/miniconda3/envs/Uni-Sign/bin/python -B \
  -m torch.distributed.run --standalone --nproc_per_node=3 \
  validation/smoke_geosign_unified.py --output /tmp/unisign-geosign-ddp-smoke
```

该测试使用小 mT5 / pose fixture 和实际双曲分支，检查双优化器与存档恢复，
不能替代生产 VAE / diffusion、完整数据和显存的 GPU 验证。
