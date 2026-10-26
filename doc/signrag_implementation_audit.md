# SignRAG 实现审计

检查日期：2026-10-26，America/Chicago。检查对象为当前 Uni-Sign checkout，HEAD `709495ca22`，以及工作区尚未提交的 `graded_retriever.py`、`script/twopass_csl.sh` 和设计/进度文档。没有修改训练代码，没有启动、停止或重跑 GPU 实验。

实际检查了检索评分、TF-IDF/draft 提示构造、RA-SFT 数据与模型输入、checkpoint 选择、四折标签、词级提示、队列及现有日志/权重。下面的数值来自当前文件或本次 CPU 检查；没有独立重新生成全量 SLT 预测。

整体实现路径是成立的，但实验协议、cross-fit 隔离和 Stage 2 的目标定义存在需要处理的问题。对照 §11.3 的具体“Code”说明，G1/G2 的初始化、冻结特征、候选构造、soft CE + exact loss、dev→train-gallery 选择及 checkpoint 导出均基本一致。Stage 2 当前只训练冻结特征上的 phi/text head；更大容量 sign encoder、可训练文本塔及 learned reranker 属于后续规划，其未实现不算当前 G1/G2 的缺陷。

1. **高：How2Sign 用 test 指标选择 checkpoint，结果不是独立测试集评估。**

   `fine_tuning_rag.py:78-84` 在 How2Sign 上把 `dev_dataloader` 直接指向 `test_dataloader`，`140-144` 再据该 split 的 BLEU-4 保存 best。`script/train_rag_h2s.sh:79` 也显式取 `max(test_bleu4)`。这与进度文档披露的现有协议一致，但协议本身会造成选择偏差。

   当前日志确认：`H2S_empty` 的 15.133 来自 epoch 0，`H2S_oracle_d` 的 17.036 来自 epoch 3，`H2S_real` 的 15.042 来自 epoch 1。这些是按 test 选出的最好分数，不能表述为 dev-selected independent test。应从 train 建立独立的、按源视频分组的验证集，冻结选择规则后再报告 test；旧结果保留并明确其协议即可。

2. **中，设计定义需统一：Stage 2 的候选子集 CE 符合“Code”说明，但不等价于前文的全库 KL。**

   设计 §11.3 的主体定义是训练句子全库上的 `q = softmax(TF-IDF / T)` 和 KL。`graded_retriever.py:184` 构造 batch 句子与各自 top-8 邻居的并集 U，`192-196` 只在 U 内重新归一化 q 和 p；没有采样概率或归一化校正。该候选子集目标已在同节“Code”说明中明确列出，所以不能把它判定为漏实现或写错 loss；应指出文档的两种目标定义不同，当前代码实现了后者。

   本次用实际 CSL-Daily train 标签重建相同的 TF-IDF（6578 个唯一句子，char 1–2 gram、sublinear TF、T=0.1），固定抽取 128 个真实 clip：U 有 1026 句；全库 q 在 U 内的平均概率质量只有 **17.4523%**，top-8 只有 **2.2930%**。因此“剩余目标接近零，可以忽略”并不成立：TF-IDF=0 的项在 softmax 前仍有 exp(0)=1 的质量。

   这不会使当前 loss 产生 NaN，本次检查的 loss、梯度及 own-mask 都正常。应明确选择：实现全库目标/有依据的近似，或把当前候选集目标正式写成 `q(u | U)` 并作为独立目标评估。候选集 CE 本身成立，不必仅为符合旧公式而强制改成全库计算；需要修正的是等价性说明及“剩余目标近零”的论据。当前 smoke test 验证数值与导出兼容性，尚未验证该目标是否提升检索质量。

3. **中：四折 translator 仅按句子 ID 分组，同文不同 ID 会跨折。**

   `script/twopass_csl.sh:69-73` 使用 `name[:7]`。本次直接读取已经生成的四折标签，clip ID 的 train/held-out 交集均为 0，四个 held-out 集的并集恰好覆盖全部 18401 个训练 clip；但按实际 `norm_text` 比较，仍有下表中的 held-out clip 在对应训练折有相同文本。

   | Fold | Train clips | Held-out clips | Held-out clips with identical train text |
   |---|---:|---:|---:|
   | 0 | 13809 | 4592 | 14 |
   | 1 | 13805 | 4596 | 19 |
   | 2 | 13798 | 4603 | 15 |
   | 3 | 13791 | 4610 | 15 |
   | Total | — | 18401 | **63 (0.3424%)** |

   例如 `S000007_P0000_T00` 与 `S000013_P0000_T00` 的文本都是“他是谁？”，却在 fold0 的两侧。这是规模较小但实际存在的同文泄漏，不能声称所有 held-out 句子都未见过。应按规范化文本分组；若还需保证同 sentence ID 一起出现，可对 sentence-ID/同文关系取连通分量再分折。已有四折实验可先保留，在分析时剔除并单列这 63 个 clip；严格的新 cross-fit 需要重新分组。

   `word_retrieval_csl.py:207-209` 使用相同的分组思路。另一个已披露限制是其冻结前端 `E_csl` 在全量 CSL train 上监督微调过，因此 CTC head 的 cross-fit 不等于整个预测链的严格 cross-fit。

4. **中：旧版 `--train_encoder` 的 GradCache 重复更新 BatchNorm，且不等价于完整大 batch。**

   `signrag_contrastive.py:365-370` 在 train 模式下无梯度编码每个 chunk；`400-403` 又在 train 模式重算。真实 `STGCN_block` 含多个 BatchNorm，前向即更新 running statistics，两遍会重复更新。这条路径确实用于已有 `S2fd_E1_h2s`（args 中 `train_encoder=true`，`encoder_chunk=16`，存在 done.json）。

   本次 CPU 使用仓库真实 `STGCN_block`，8 个样本、chunk=2：完整 batch 一遍的 BN 更新计数为 1，现有两遍分块路径为 8；参数梯度最大差为 17.1038。进一步与**相同微批次的一遍计算**比较，梯度误差约 3.8e-6，但 BN 更新仍为 4 对 8，running mean 最大差约 0.0992。这区分了分块 BN 本身的差异和额外重复更新的问题。

   应在重算时保存/恢复 BN buffers，或固定 BN；并明确对比基准采用 full-batch BN 还是 microbatch BN。当前 frozen-feature 的 S2dn/G1/G2 不走这条分支，不受此问题影响。

5. **中：two-pass 队列按文件存在判断完成，空文件/部分文件会误判。**

   `script/twopass_csl.sh:50-58` 的 `done_job`、`ready` 仅使用 `-f`。本次提取这两个原函数，在临时目录放置一个空的 `fold0_heldout/test_tmp_pres.txt`，两个判断均返回成功。`fine_tuning.py:458-464` 直接写最终文件名，未使用临时文件与原子发布。

   同时 `draft_retrieval.py:87-89` 静默只保留预测文件里有的样本，没有检查覆盖完整性。这会让残留文件被当成成功产物、跳过重译，或提前启动下游。`datasets_rag.py:49-52` 的覆盖检查通常会阻止缺失提示进入训练，因此本次没有证据表明现有完成实验使用了部分提示；风险主要是误调度和失败后的恢复。

   应在翻译成功、预测 ID 唯一且与预期标签完全匹配后原子发布，写 DONE/manifest；队列依赖该完成标记而非 `.txt` 存在。`draft_retrieval.py` 应显式区分全量输入与有声明的 fold/subset 输入。

6. **中：synthetic noise 是按 test 的参考指标校准的，test 参与了方法配置。**

   `doc/signrag_progress.md:498-501` 明确按 test hint retrieval BLEU 10.66 选择 noise=0.5，使 train 为 10.82；对应两个 `metrics_*.json` 也存在。test retrieval BLEU 的计算需要 test reference，因此该配置不是完全隔离 test 的流程。这里没有发现 test reference 被直接拼入非 oracle 模型输入，问题发生在超参数校准阶段。

   应按 dev 或训练内部留出集的 hint 指标校准，然后固定 noise 再测 test。当前 dev 的 hint BLEU 为 10.88，重新按 dev 选择可能仍是 0.5，但需要把选择规则落实到 dev，不能倒推宣称既有 test 从未参与。

7. **较低：RA-SFT 的 dev/test 长视频仍随机采帧，小幅增益会受到额外噪声影响。**

   `datasets.py:495-496` 对 duration > max_length 一律 `random.sample`，没有区分 train/dev/test；RA-SFT 和 rerank 使用该数据类。这是继承的原始行为。特征提取另有 deterministic uniform 版本，两个路径不同。

   当前 feature metadata 中，How2Sign test 有 474/2349 个 clip 达到 256 帧上限，CSL test 为 23/1176。达到上限包含原长恰好等于上限的情形，因此这些是可能受影响的上限计数，不是精确长视频数量。建议 dev/test 固定采样索引或使用 uniform，并确保基线/增强模型一致；不能把很小的差异仅归因于检索。

本次通过的检查：

- 12 个相关 Python 文件通过 AST 语法检查，5 个 shell runner 通过 `bash -n`。
- soft-alignment 两条评分实现的最大差约 7.45e-9；分块前后梯度最大差约 6.68e-6，梯度均有限。Stage 2 的 own sentence mask 没有梯度泄漏。
- 检查的 19 个 sentence-retrieval JSON 全部覆盖预期 query ID，没有重复 query；邻居均来自训练 gallery，score 有限；所有 train query 都通过 self/同文过滤；distinct 版本没有同一行重复文本。How2Sign oracle 少数行只有 1–2 个邻居，与现有文档一致。
- 10 个 CSL/H2S 词级 hint JSON 均完整覆盖对应标签 ID，无重复 ID；H2S WLASL hint 的 DONE 存在。
- `datasets_rag.py` 实际在训练阶段以 0.5 概率丢弃 hint，在 dev/test 保留；`models_signsmd.py` 确实把 `[hint][prompt][pose]` 输入 mT5。所核对的 real s2t、predgloss、WLASL 和 dev/test draft hint 路径没有把 query GT 当作检索 query；另有明确使用 GT 的 oracle、synthetic train draft、训练目标、过滤及分析路径。
- CSL 日志确实按 dev 选 checkpoint，主表数值匹配：empty 25.18、real_d 25.00、oracle_d 27.63、twopass_syn 25.50，均为对应最好 dev epoch 的 test BLEU-4。
- CSL empty、oracle_d、twopass_syn 的 best checkpoint 均可读，各含 627 个 tensor，key/shape 完全一致；其中 583 个 float32、44 个 int64。这是序列化及结构检查，没有对三份大权重做逐值有限性检查或重新运行推理。
- S2dn CSL phi 权重可加载，1024→768 phi 和 768→768 text head 均有限，tau/tau_attn 与导出检索代码的使用方式一致。历史 graded smoke 的 done.json 显示最好 epoch=0，test retrieval BLEU=5.301688；它证实初始模型兼容，不证明训练后的提升。

当前结论的边界：主结果数值是日志核对，未重跑全量推理；Stage 2 尚无正式 G1/G2 结果。优先处理独立验证协议与 Stage 2 目标定义，再修复 cross-fit 分组和完成标记。现有 CSL oracle 对照仍支持“好提示能被模型利用”，但当前证据不足以证明完整的强检索器方案已经实现或获得稳定增益。
