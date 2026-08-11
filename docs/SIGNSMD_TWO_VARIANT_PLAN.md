# SignSMD 双版本 + Step 对齐时间轴 —— 实施计划(待确认)

状态:**已实施(2026-08-11,用户确认后)**。第 8 节的开放问题仍待拍板;
冒烟命令见第 7 节第 6 步,由用户执行。

## 1. 需求(来自讨论,已确认的三条)

1. **pose 与文本完全对齐**:模型看到的文本必须恰好描述模型看到的那些 pose 帧
   (对标 RGB 分支:RGB 帧永远是 pose 已采样帧的子集,共享同一时间轴)。
2. **两个版本的描述并存**:
   - **stats 版**:motion-smd `angle_text_v5` 那种统计形式——逐信号通道的
     分段数值时间线(`holds at X` / `increases: X → Y` / `repeats N cycles`);
   - **coarse 版**:现在 signsmd2d 这种粗粒度结构化文本。
3. **时间轴用下采样后的 step 为单位**:不用秒、不用百分比;描述的是
   "下采样单位 step 中 motion 的变化",step 序号 = 模型 pose 序列的 token 位置。

## 2. 核心设计:一条 step 对齐的时间轴

### 2.1 已完成的前置(上一步已改)

`datasets_signsmd.py`:训练/评估默认 `--frame_sampling linear`——
超过 `--max_length`(256)的片段用 `np.round(np.linspace(0, duration-1, 256))`
确定性线性下采样,短片段整段保留;`unisign_random` 留作消融。

### 2.2 编译器采用完全相同的规则(本计划核心)

`signsmd2d/compiler.py` 的 `compile_clip` 在计算证据**之前**:

```python
# 与 datasets_signsmd.load_pose 逐字相同的规则
if duration > max_length:                      # max_length 默认 256
    idx = np.round(np.linspace(0, duration-1, max_length)).astype(np.int64)
else:
    idx = np.arange(duration)
keypoints, scores = keypoints[idx], scores[idx]   # 之后才 compute_evidence
```

由此得到三重对齐,全部是**构造性精确**的:

| 对齐维度 | 保证 |
|---|---|
| 覆盖 | 文本只依据模型将看到的那些帧计算(证据、分段、统计全部在重采样序列上) |
| 参照 | 文本里的 `step i` 就是模型 pose 序列第 i 个 token,一一对应 |
| 速率 | 阈值/速度按"每 step 变化量"衡量——正是模型输入端实际感受到的变化 |

头部格式(两个版本共用):

```
Motion: 256 steps (linearly downsampled from 1127 frames, ~47.0s at 24 FPS)
Motion: 114 steps (114 frames, ~4.8s at 24 FPS)          # 短片段,1 step = 1 帧
```

区间一律渲染为 `[step a–b]`(闭区间),hold 加 `(N steps)`。

### 2.3 必须维持的不变量

- 训练侧 `--max_length` 必须等于编译时 `--max_length`(都默认 256),
  且训练侧必须用 `--frame_sampling linear`(已是默认)。写入 manifest 供核对。
- 编译器与 loader 对 pose pickle 可选 `start`/`end` 窗口的处理保持一致
  (两边都先切窗口再重采样,现状已一致)。
- 编译器永不接触 caption/gloss;禁词断言(`forward/backward/contact/
  toward|away from the body`)对两个版本同样生效。

## 3. Variant A:coarse 版(contract 升级 `signsmd2d_v0_1` → `signsmd2d_v0_2`)

内容与现在基本一致,仅时间轴改造:

- `realization.py`:百分比区间 `[Initial Phase: 0%–100% | movement]` →
  `[Initial Phase: step 0–113 | movement]`;头部换成 2.2 的格式。
- `segmentation.py` 逻辑不变(它对帧序列无假设),但其速率阈值的语义从
  "每原始帧"变为"每 step";`configs/signsmd2d_thresholds.yaml` 数值暂不变
  (短片段 1 step = 1 帧,数值含义不变;长片段 per-step 变化变大,正是
  模型实际看到的),注释与 `version: signsmd2d_v0_2` 更新。
- 输出目录:`data/<ds>/signsmd2d_v0_2/texts`。v0_1 目录保留不动(可对照)。

已知的 coarse 质量问题(47 秒样本塌缩成 1 个区间、"mixed handshape" 兜底过多)
**不在本次范围**,后续单独做长区间再分段;本次只换时间轴。

## 4. Variant B:stats 版(新,contract `signsmd2d_stats_v0_1`)

### 4.1 形式:motion-smd `angle_text_v5` 的 2D 手语移植

逐通道输出分段数值时间线。通道全部来自现有 `evidence.py`(不加新几何):

| 分区 | 通道 | 单位 | 语义动词(+/−) |
|---|---|---|---|
| Bimanual | hand separation | sw(肩宽) | increases / decreases |
| [Left/Right Wrist] | horizontal position(+ = signer-right) | sw | moves toward signer-right / signer-left |
| | vertical position(+ = up) | sw | rises / lowers |
| [Left/Right Arm] | elbow bend | ° | bends / straightens |
| [Left/Right Hand] | total finger curl(手级汇总,5 指卷曲和) | ° | curls / extends |
| [Left/Right Fingers] | **逐指 curl × 5**(thumb/index/middle/ring/pinky,各为 3 关节弯曲和;<~51° 伸直,>~126° 握拢) | ° | curls / extends |
| | finger spread | ° | spreads / narrows |
| | thumb opposition(拇指-指尖距) | palm 单位 | opens / closes |
| | palm orientation proxy(+ = 背向相机) | 无量纲 | increases / decreases |
| Non-manual | brow height / mouth opening / head tilt(+ = 向 signer-left) | 眼距单位 / 眼距单位 / ° | rises / opens / tilts … |

### 4.2 分段算法:移植 `test_peak_valley.find_segments_with_static`

- 平滑 → 滑窗 std 活动度 → active/static 分类 → active 区内 peak/valley
  细分 → `detect_cyclic` 周期检测(手语重复动作多,保留该能力,最小振幅
  用通道各自的 `min_change`)。numpy-only,不引入 scipy。
- **NaN 适配**(motion-smd 假设信号稠密,我们的 2D 证据带有效性掩码):
  按有限值连续段(≥ `min_valid_run` steps)分别分段;≥ `gap_report_min`
  steps 的缺失段输出 `not detected [step a–b]`;全程缺失输出
  `not detected throughout`;全程变化 < `min_change` 输出
  `static at X throughout`。
- 不移植的部分:角度 unwrap(2D 通道无 ±180° 回绕)、egocentric 轨迹
  (与禁词冲突且 2D 不支持)。

### 4.3 输出样例(设计目标,以 47s 长片段为例)

```
Motion: 256 steps (linearly downsampled from 1127 frames, ~47.0s at 24 FPS)
Units: sw = shoulder width; steps index the model pose sequence (inclusive).

Bimanual:
  Hand Separation:
    holds at 1.10 sw [step 0–41] (42 steps)
    decreases: 1.10 → 0.42 sw [step 42–68]
    repeats 3 cycles (0.42 to 0.95 sw) [step 69–140]

[Right Wrist]
  Horizontal position (+ = signer-right):
    holds at 0.35 sw [step 0–30] (31 steps)
    moves toward signer-left: 0.35 → 0.05 sw [step 31–56]
  Vertical position (+ = up):
    rises: -0.40 → 0.10 sw [step 31–58]
[Right Arm]
  Elbow bend:
    bends: 45° → 95° [step 30–55]
[Right Hand]
  Total finger curl:
    repeats 4 cycles (130° to 310°) [step 60–140]
  Palm orientation proxy (+ = appears to face away from the camera):
    static at -0.35 throughout
...
Non-manual:
  Brow height:
    rises: 0.42 → 0.51 [step 100–120]
  Head tilt (+ = toward signer-left):
    not detected [step 200–255]
```

对比 coarse 版同一片段("both hands hold position" 的退化描述),stats 版
天然不受长区间塌缩影响——分段是逐通道、数据驱动的。

### 4.4 新文件

- `signsmd2d/stats_segments.py`:4.2 的分段/周期检测(NaN-aware 移植)。
- `signsmd2d/realization_stats.py`:通道抽取(复用 `compute_evidence`
  输出)+ 渲染。
- `configs/signsmd2d_stats_thresholds.yaml`(`version: signsmd2d_stats_v0_1`):
  逐通道 `min_change` / `static_threshold`(启发式初值,per-step 单位)+
  `smooth_window: 5`、`static_min_frames: 8`、`min_valid_run: 8`、
  `gap_report_min: 8`、`top_k: 0`(0 = 全通道;正整数 = 按活动度
  range/min_change 取 top-k,对标 motion-smd 的 top-k 消融,备用)。
- 输出目录:`data/<ds>/signsmd2d_stats_v0_1/texts`。

## 5. 生成入口与消费方式

`generate_signsmd2d.py` 增加:

- `--variant {coarse,stats}`(默认 coarse);
- `--max_length 256`(必须与训练一致,记入 manifest);
- thresholds/out_root 默认值随 variant 解析;manifest 记录
  variant、contract 版本、max_length、阈值 hash。

训练侧**零代码改动**:`--signsmd_text_root` 指向哪个目录就用哪个版本:

```bash
# coarse v0_2
--signsmd_text_root data/How2Sign/signsmd2d_v0_2/texts
# stats
--signsmd_text_root data/How2Sign/signsmd2d_stats_v0_1/texts
```

`script/train_signsmd_fusion.sh` 的文本根更新为 v0_2 并加注释(需先重新
生成)。stats 文本比 coarse 长,必要时训练加 `--signsmd_max_tokens 768`
(先生成后统计实际 token 长度再定)。

## 6. 与 motion-smd 下游的衔接(不在本次实施)

motion-smd 的 `dataset_caption.py` + `train_lora_llm.py` 是
"SMD 文本 → 自然语言 caption" 的 LoRA 管线。stats 版按同样的
`(structured text, caption)` 配对格式设计,后续若要做
"sign SMD → 译文" 的 LLM 实验可直接套用;本次只产出文本。

## 7. 实施顺序(确认后执行)

1. `compiler.py`:trainer 同款重采样 + `variant`/`max_length` 参数
   (已完成:`__init__.py` 的 `COARSE_VERSION`/`STATS_VERSION` 定义)。
2. `realization.py`:step 区间 + 新头部;`signsmd2d_thresholds.yaml`
   bump 到 v0_2。
3. `stats_segments.py` + `realization_stats.py` + stats 阈值 yaml。
4. `generate_signsmd2d.py` 插 variant/max_length。
5. 更新 `script/train_signsmd_fusion.sh`、`docs/POSE_SIGNSMD_FUSION.md`、
   记忆文件。
6. 交给你跑冒烟(我不执行 Python):

```bash
# 两个版本各生成 10 条 How2Sign test,人工检查文本
python generate_signsmd2d.py --dataset How2Sign --splits test --variant coarse --limit 10
python generate_signsmd2d.py --dataset How2Sign --splits test --variant stats  --limit 10
```

重点检查:①同一 clip 两版本头部 step 数一致且等于 loader 输出长度;
②长片段(>256 帧)stats 版分段合理、无禁词;③token 长度分布。

## 8. 待你拍板的问题

1. **stats 版通道集合**(4.1 表)够不够?要不要加"指尖间(指间距)"、
   "腕-鼻距离(近脸程度)"这类通道?
2. **两版本将来如何进模型**:各自单独训练对比,还是同一次训练拼接两段?
   (影响 `--signsmd_max_tokens`;当前按"分别指定、单独训练"设计。)
3. coarse 版长区间塌缩的修复(区间内再分段)排在本次之后单独做,OK?
4. OpenASL 的 v0_1 coarse 文本是否已生成过?v0_2 需要 train/dev/test 全量
   重新生成(编译很快,How2Sign test 2349 条约 15 秒)。
