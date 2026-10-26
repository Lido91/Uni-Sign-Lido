# Experiment: SignRAG-Style Retriever on Uni-Sign (Steps 1–3)

Status: **implemented and smoke-tested (2026-10-21/22); the P0 full run
(`bash script/signrag_retrieval.sh p0`) was started on 2026-10-22.**
The code is `signrag_common.py`, `signrag_retrieval.py`,
`signrag_contrastive.py` and `script/signrag_retrieval.sh`. Smoke-test
results are in §8.6, and the semantic-positive audit on the full label sets
is in §3.4.

Self-contained experiment. It does four things:

1. Reproduces the retriever of SignRAG (Rao et al., arXiv 2610.11371, §3.2)
   on top of our pretrained pose-only Uni-Sign encoder.
2. Measures it with metrics SignRAG never reported.
3. Tests whether contrastive training of a **translation-initialised
   projection head** improves it. Optionally the ST-GCN encoder is trained
   too (a switch).
4. Treats sentences that repeat the same meaning as **positives** instead of
   negatives.

No translator is trained here; retrieval-augmented SFT (RA-SFT) comes after
this experiment (§11).

**Retrieval goal (bidirectional).** Similar signing should return similar
sentences, and similar sentences should return similar signing.

- sign→text serves retrieval-augmented SLT.
- text→sign serves retrieval-augmented sign production (our text-to-pose
  line).

Training is therefore symmetric: both directions use full-gallery negatives
when possible. Model selection averages both directions, and evaluation
reports both (§4.6, §5).

## 0. TL;DR

- **Step 1 (zero training, zero new parameters).** Extract frame-level
  ST-GCN features from a frozen Uni-Sign checkpoint. Retrieve the top-3
  training sentences with SignRAG's attention-weighted frame-level
  sign→sign score (Eq. 5–8).
- **Step 2 (contrastive training).** Train the projection head \(\phi\)
  against a frozen official mT5-base text tower, with \(\phi\)
  **initialised from the translation head `pose_proj`**, so step 0 of
  training *is* Step 1.
  - Positives are identical texts, plus (switch) paraphrases above a
    threshold.
  - The ST-GCN can be trained too (switch `--train_encoder`).
- **Step 3.** Re-run Step 1's retrieval with the trained \(\phi\), and add
  sign→text retrieval.
- **Scoring.** Every run is scored against **random** and a **TF-IDF text
  oracle** on the same queries and filters. Metrics: retrieval-only
  BLEU/ROUGE, token F1, exact match, same video, train–test gap, and the
  fraction of the gap to the oracle that is closed.

## 1. Motivation and Hypothesis

SignRAG adds a target-domain retrieval gallery to a pose-only MLLM: the
top-3 training sentences retrieved for a query are concatenated into the
LLM prompt during SFT. The evidence on retrieval is thin:

- Retrieval alone adds only **+1.5 BLEU-4** on CSL-Daily (29.72 → 31.22,
  their Table 2a). There is **no retrieval ablation on any English
  dataset**. The authors only say that "the gains of SignRAG over existing
  state-of-the-art methods are more moderate on the English datasets".
  How2Sign B@4 (their Table 1), with external data: SSVP-SLT 15.5,
  Uni-Sign 14.9, SignRAG-SFT 15.6 (+0.1), SignRAG-RFT 16.7. Without
  external data: RVLF 14.3, SignRAG-SFT 13.7 (−0.6), SignRAG-RFT 15.2.
  SignRAG-SFT already includes retrieval.
- A dedicated RL reward (RUG-RFT) is needed to "suppress harmful reliance"
  on retrieval.
- The paper **never measures retrieval quality directly**: no Recall@K, no
  random or oracle baseline, no helps/hurts statistics (§2.3).

At the same time, the model clearly *can* exploit retrieval. Without
identical-text filtering, SFT BLEU-4 collapses to **11.21** (vs 31.22),
because the model learns to copy retrieved sentences when they contain the
answer (their Table 5).

**Hypothesis (H-weak):** retrieval quality, not retrieval use, is the
bottleneck of retrieval-augmented SLT. SignRAG's retriever is weak because:

- (a) it retrieves sign→sign rather than in a text-anchored space;
- (b) the retrieval score is not the score it was trained with, and has no
  temperature (§8.6 shows this makes it prone to "hub" clips);
- (c) its contrastive stage uses a tiny batch (8, i.e. 7 in-batch
  negatives), with only the own pair as positive.

This experiment tests H-weak on our encoder in three moves: reproduce the
retriever (Step 1), measure it against bounds, and check whether a properly
trained head (Step 2/3) closes the gap.

## 2. Background

### 2.1 SignRAG retrieval score (§3.2, Eq. 5–8)

The gallery is the target training split only. Each entry is a frame-level
skeleton feature sequence \(Z^g\in\mathbb{R}^{L_g\times C}\) with its
translation. For a query \(Z^q\):

\[
Q=\operatorname{Norm}(\phi_q(Z^q)),\quad
G=\operatorname{Norm}(\phi_g(Z^g))\qquad\text{(row-wise } \ell_2\text{)}
\]
\[
E_{ij}=Q_i^\top G_j,\qquad
P_{ij}=\frac{\exp(E_{ij})}{\sum_{j'}\exp(E_{ij'})},\qquad
S(Q,G)=\frac{1}{L_q}\sum_{i=1}^{L_q}\sum_{j=1}^{L_g}P_{ij}E_{ij}
\]

- No temperature in the retrieval softmax.
- Top-\(K=3\), exhaustive over all gallery entries (sharded, chunked,
  merged top-K).
- Training queries exclude themselves and entries with identical
  translations (App. C.7).
- Retrieval dropout \(p=0.5\) during SFT.
- The origin of \(\phi_q,\phi_g\) is **not specified**.
- The sign encoder is fine-tuned on each downstream training set before the
  gallery is built.

### 2.2 SignRAG Stage I contrastive alignment (§3.1, App. B.2, C.2, C.3)

| Item | SignRAG Stage I |
|---|---|
| Keypoints | MMPose, 77 per frame (9 body, 42 hands, 8 mouth, 18 face), CoSign-style |
| Sign encoder | CoSign front end + 3-layer Transformer (8 heads, hidden 512, FFN 2048) |
| Text encoder / decoder | first 3 layers of mBART (following GFSLT-VLP); frozen or trained: not specified |
| Contrastive feature dim | 1024 (sign and text), L2-normalised |
| Similarity | fine-grained frame↔token soft alignment, below |
| Contrastive loss | bidirectional InfoNCE over in-batch pairs, with its own temperature \(\tau'\), distinct from the attention temperature \(\tau\); the formula below is the standard expansion (the paper gives none) |
| Caption loss | CoCa-style sign→text translation loss \(\mathcal{L}_{slt}\) |
| Total | \(\mathcal{L}_{pt}=\mathcal{L}_{con}+\mathcal{L}_{slt}\) (unweighted) |
| Optimiser | SGD, lr 0.01, weight decay 0.001, cosine, 20 warmup epochs |
| Epochs / batch | 200 / **8** |
| Corpus | CSL-News (vocabulary 11,540); Stage I data for ASL not specified |
| Per-loss ablation | not reported (Table 6 only compares whole pretraining strategies) |

Fine-grained sign→text similarity, with frame features \(f_1..f_M\) and
token features \(t_1..t_L\):

\[
E_{ij}=f_i^\top t_j,\qquad
P_{ij}=\operatorname{softmax}_j(E_{ij}/\tau),\qquad
s(S,T)=\frac{1}{M}\sum_{i}\sum_{j}P_{ij}E_{ij}
\]

Each sign frame spreads attention over the text tokens, and the
video–sentence score averages the matched similarity over frames.
\(s(T,S)\) is computed the same way, with tokens attending over frames and
averaging over tokens. \(\tau\) interpolates between ColBERT-style MaxSim
(\(\tau\to0\)) and the mean pairwise similarity (\(\tau\to\infty\)).

The scores then enter an in-batch bidirectional InfoNCE over \(B\) pairs.
The positive is \((S_a,T_a)\); the negatives are \((S_a,T_b)\), \(b\ne a\):

\[
\mathcal{L}_{S\to T}=-\frac{1}{B}\sum_{a}\log\frac{\exp\!\big(s(S_a,T_a)/\tau'\big)}{\sum_{b}\exp\!\big(s(S_a,T_b)/\tau'\big)},
\qquad
\mathcal{L}_{T\to S}=-\frac{1}{B}\sum_{a}\log\frac{\exp\!\big(s(T_a,S_a)/\tau'\big)}{\sum_{b}\exp\!\big(s(T_a,S_b)/\tau'\big)}
\]
\[
\mathcal{L}_{con}=\mathcal{L}_{S\to T}+\mathcal{L}_{T\to S},\qquad
\mathcal{L}_{pt}=\mathcal{L}_{con}+\mathcal{L}_{slt}
\]

With batch 8, every query sees only 7 negatives. The retrieval score of
§2.1 has the same functional form as \(s(S,T)\), but it is **sign→sign with
no temperature**, so the score used at retrieval time is never optimised
directly.

### 2.3 What SignRAG measures about retrieval

| Item | Reported? |
|---|---|
| Recall@K / hit rate / retrieval accuracy | no (App. E.5: "do not currently report direct retrieval results") |
| Token-overlap F1 between top-K texts and GT (Eq. 20) | yes, only as a curve over K (Fig. 3a): rises then falls; tracks BLEU-4 |
| Random / oracle retrieval baselines | no |
| Fraction of queries where retrieval helps or hurts | no |
| Identical-text filtering effect | yes: SFT BLEU-4 11.21 without vs 31.22 with (Table 5) |
| Qualitative examples | Table 15 (top-3 lists, some loosely related, one duplicate); D.7 one failure case |
| Retrieval latency | 3.49 ms/query (1K gallery) → 23.88 ms (18.4K); OpenASL 705.85 ms/query at 95,888 (Tables 12–13) |

### 2.4 Why we expect it to be weak

| Weakness | Evidence | Addressed by |
|---|---|---|
| Sign→sign retrieval; the gallery text is only carried along | motion similarity ≠ meaning; signer style and tempo leak into features | Step 3 sign→text score; later a text-anchored retriever (§11) |
| Retrieval score ≠ training score, no temperature | Eq. 5–8 (sign–sign, no \(\tau\)) vs Eq. 21–23 (sign–text, trainable \(\tau\)); with no \(\tau\) a query is not even its own top-1 in 6% of cases (§8.6, S4) | Step 2/3 retrieve with the trained score; `--frame_tau` ablation |
| Weak contrastive signal | batch 8 → 7 in-batch negatives; own pair as the only positive | full-gallery negatives; identical + semantic positives; batch-8 ablation (S2-e) |
| Noisy neighbours | F1 falls as K grows; RUG-RFT exists to suppress harmful reliance | measured directly here |
| Encoder fine-tuned on the gallery split | train queries get better neighbours than test queries | train–test gap metric; cross-domain encoder E2 |

### 2.5 Related: SCL-SLT (Lai et al., ACL 2026, arXiv 2604.22374)

*Selective Contrastive Learning for Gloss-Free SLT* attacks the same false-
negative problem from the batch side.

- **Setting.** RGB, ResNet-18 + mBART-large-50 with LoRA, GFSLT-VLP-style
  CLIP pretraining with CiCo-style similarity, batch 16.
- **Trajectory analysis.** Over reference checkpoints, only **35.9%** of
  in-batch negatives are consistently pushed apart (H→L). 31.9% stay highly
  similar (H→H), 17.8% become more similar (L→H), and 14.5% stay low.
- **Pair Selection.** Mini-batches are built from these similarity
  trajectories with an easy→hard curriculum (log schedule best).
- **Results.** CSL-Daily 23.25 BLEU-4, PHOENIX14T 26.00 (with VLP).
- **Duplicate statistics (their Table 3).** On CSL-Daily, ~99% of the
  ~6.6k unique target texts map to multiple videos, so identical-target
  collisions in a batch are common. On PHOENIX14T they are rare.
- **Similarity ablation (their Table 5).** Mean pooling 12.44, [CLS] 14.85,
  CiCo fine-grained 25.30 BLEU-4. This warns that pooled similarity may
  underperform fine-grained similarity in a VLP→SLT pipeline. Our matrix
  compares the two directly on retrieval metrics (S2-a/s vs S2-d/ds).

**Relation to this experiment.**

- SCL-SLT changes *which negatives enter a batch*. It does not redefine
  positives, and it does not describe an explicit identical-text step.
- We change *what counts as a positive* (identical text + paraphrase ≥
  \(\theta\)) and use the *whole* training-text gallery as negatives
  instead of selecting a batch.
- The two are complementary; trajectory-based selection could be added on
  top of S2-d later.

## 3. Confirmed Setup (checked 2026-10-21)

### 3.1 Checkpoints

Test scores were recomputed from the saved predictions with
`SLRT_metrics.translation_performance`, after stripping the shared
`sample: <name>, prediction:` line prefix (which inflates BLEU to ~47 if
left in).

| ID | Path (under `/data/hwu/Uni_sign_repo/out/`) | Test BLEU-4 / R-L | Role |
|---|---|---|---|
| E1 | `how2sign_pose_only/how2sign_pose_only_slt.pth` (released Uni-Sign) | H2S 14.50 / 34.51 | faithful to SignRAG: encoder trained on the target train split |
| E2 | `openasl_pose_pretrain/openasl_pose_only_slt.pth` (released Uni-Sign) | OpenASL 22.64 / 42.83 | encoder for H2S that has never seen H2S: no memorisation of gallery clips |
| E3 | same file as E2 | same | faithful on OpenASL |

Both files are `{'model': state_dict}` with `pose_proj.weight (768, 1024)`,
`pose_proj.bias (768,)`, `part_para (1024,)`, and load into `models.Uni_Sign`
(mT5-base) with `strict=True`. Not used: our `stage3_finetuning_openasl_h2s`
(H2S test BLEU-4 11.14, weaker than E1) and `stage1_pretraining_openasl`.

### 3.2 Data

| | How2Sign | OpenASL |
|---|---|---|
| train clips / videos | 31,086 / 2,192 | 96,476 / 2,042 |
| dev clips / videos | **none** (`labels.dev` absent) | 967 / 472 |
| test clips / videos | 2,349 / 149 | 976 / 456 |
| test videos that also occur in train | **0** | **456 (all)** |
| unique train texts | 29,981 | 84,399 |
| train clips whose text occurs ≥ 2× in train | 1,561 (5.0%) | 13,341 (13.8%) |
| most repeated train texts | "hi" ×99, "good" ×95, "okay" ×78 | "hello everyone" ×722, "sign1news" ×711, "and those are your top stories for today" ×674 |
| test clips whose text occurs verbatim in train | 77 (3.3%) | 104 (10.7%) |
| frames per clip, mean / median / p95 (300-clip sample) | 178 / 144 / 465 | 236 / 208 / 536 |
| clips longer than `--max_length 256` | ~21% | ~39% train, ~33% test |

Text normalisation (`signrag_common.norm_text`), used for every "identical
text" statistic and filter: lowercase, replace every non-word character
except `'` with a space, collapse whitespace.

Video ID (`signrag_common.video_id`) = first 11 characters of the clip name
(the YouTube ID). This parses correctly for every clip in both datasets
(`--7E2sU6zP4_10-5-rgb_front.mp4` → `--7E2sU6zP4`,
`Mci9oyb5V2E-00:00:06.000-…` → `Mci9oyb5V2E`).

### 3.3 Hardware and environment

- 4 × L40S (46 GB), 128 CPU cores, 566 GB RAM.
- Python: `miniconda3/envs/Uni-Sign/bin/python` (torch 2.1.1, transformers
  4.40.0, sklearn 1.0.2, sacrebleu 2.2.0, rouge 1.0.1). No faiss and no
  nltk; neither is needed at this gallery size.
- `/data` has 6.6 TB free; all outputs go under
  `/data/hwu/Uni_sign_repo/out/signrag_retrieval/` (`$OUT` below).

### 3.4 Semantic-positive audit (full label sets, already run)

`signrag_contrastive.py --audit_positives` uses
`sentence-transformers/paraphrase-MiniLM-L6-v2` (local HF cache; mean
pooling, L2-normalised) on the deduplicated train texts.

- "Extra positives" are paraphrases at cosine ≥ \(\theta\) other than the
  text itself, capped at 64.
- The last column is the share of unique test texts (identical matches
  excluded) whose nearest train text reaches \(\theta\).
- Full reports: `$OUT/audit_positives/{How2Sign,OpenASL}/positive_audit.json`.

| \(\theta\) | H2S texts w/ extra pos. | H2S clips | H2S mean extra / clip | H2S test w/ neighbour | OpenASL texts | OpenASL clips | OpenASL mean extra / clip | OpenASL test w/ neighbour |
|---|---|---|---|---|---|---|---|---|
| 0.95 | 0.8% | 1.2% | 0.02 | 1.0% | 1.2% | 6.8% | 0.22 | 2.7% |
| 0.90 | 2.5% | 3.8% | 0.07 | 2.6% | 3.0% | 12.9% | 0.52 | 6.2% |
| 0.85 | 5.3% | 7.3% | 0.17 | 4.4% | 5.6% | 15.6% | 0.90 | 8.5% |
| 0.80 | 12.0% | 14.0% | 0.36 | 8.3% | 10.6% | 21.0% | 1.43 | 14.9% |
| 0.75 | 24.9% | 26.7% | 0.81 | 15.4% | — | — | — | — |
| 0.70 | 44.3% | 45.7% | 1.85 | 30.5% | — | — | — | — |

What the example pairs (borderline = least similar positive) show:

- **≥ 0.90: mostly true paraphrases.**
  - "let's try that again" ~ "try that again"
  - "then the other knee" ~ "and then the other knee"
  - "i m candace jones" ~ "i'm candace jones"
  - "that s all for now" ~ "thats all for now"
- **0.85: still mostly the same meaning.**
  - "here you go" ~ "here we go"
  - "you can do this too" ~ "you can do it"
  - "this is amazing" ~ "this is wonderful"
- **≤ 0.80: topical relatedness, not the same meaning.**
  - "so we're going to start" ~ "we're going to start in the front"
  - "why is that" ~ "that s why"
- **Failure mode at every threshold: surface-identical sentences with
  different facts.** In sign language these are different signs.
  - Numbers: "it is monday may 6" ~ "… may 25" (0.950); "3 8 million" ~
    "2 1 million" (0.905); "12 teams" ~ "15 teams" (0.870).
  - Order: "turn the fourth card" ~ "turn the first three cards" (0.970);
    "step 1" ~ "the final step" (0.930).
  - Handedness: "starting with your right foot" ~ "… with your left foot"
    (0.904).

Consequences:

1. **Threshold.** Default \(\theta=0.85\) (`POS_TH` in the script; decided
   2026-10-22, after reviewing the 0.85 examples). 0.9 is the ablation
   (`S2s09_E1_h2s`). Do not use ≤ 0.8 as hard positives; at most as soft
   weights later.
2. **How2Sign gains few positives** (7.3% of clips at 0.85, 3.8% at 0.9),
   so the effect of semantic positives there will be modest. OpenASL (15.6%
   / 12.9% of clips; news boilerplate) is where they matter.
3. **Open item:** a lexical guard that rejects positive pairs whose digits /
   number words / left–right tokens differ. Not implemented yet.
4. **Retrieval filters.** 2.6% (H2S) and 6.2% (OpenASL) of test texts have
   a ≥ 0.9 paraphrase in train, compared with 3.3% / 10.7% verbatim
   duplicates. If paraphrases are later filtered from train-time
   retrieval, these rates are what the filter should be calibrated to (as
   for identical text).

## 4. Method

### 4.1 Pipeline

```text
 STEP 1  ZERO-TRAINING BASELINE   (frozen encoder, zero new params)
 ===========================================================================
 [A] ENCODER   E1 -> H2S (faithful) | E2 -> H2S (cross-domain) | E3 -> OpenASL

 [B] EXTRACT   (every clip once, deterministic uniform frames, T <= 256)
     pose pkl --> 4-part ST-GCN --> + part_para --> X (T x 1024)   [cached]
                                                     |
                                       pose_proj --> Z = X W^T + b (T x 768)
                                                     |   (derived, not stored)
                                       [F2, optional] mT5 encoder @ pose pos

 [C] GALLERY = train split
     QUERIES = test (all) + train (3k seeded sample, for the train-test gap)

 [D] SCORE  (exhaustive, chunked, sharded over 4 GPUs)
     frame : SignRAG Eq. 5-8 on Z, phi_q = phi_g = identity  (--frame_tau ablation)
     pool  : cos(mean Z_q, mean Z_g)                       control

 [E] FILTER -> top-3      (rules in 4.4)
 [F] BASELINES            random top-3 (5 seeds) | TF-IDF text oracle
 [G] METRICS              section 5


 STEP 2  CONTRASTIVE TRAINING OF phi   (default: ST-GCN frozen, cached X)
 ===========================================================================
     sign: X --> phi (init = pose_proj W, b) --> f_1..f_M     (T x 768, L2)
           [--train_encoder: X comes from the trained ST-GCN, poses per batch]
     text: GT --> official mT5-base encoder --> text head --> t_1..t_L (L2)
                  (frozen, cached per dataset)  (init = identity)

     positives: identical text  [+ --positives semantic: paraphrase cos >= theta]

     S2-a/s  ours
           z_s = mean f,  z_t = mean t
           InfoNCE: s->t over the FULL deduplicated train-text gallery,
                    t->s over ALL training clips (linear phi, frozen encoder:
                    mean phi(X) = phi(mean X)), multi-positive, learnable tau

     S2-d  SignRAG Stage I, on our pretrained encoder
           E_ij = f_i . t_j ;  P = softmax_j(E / tau) ;  s = mean_i sum_j P E
           bidirectional in-batch InfoNCE with tau', multi-positive
           + L_slt : phi(X) --> checkpoint translation mT5 (frozen) --> CE

     early stop on dev R@5 averaged over s->t and t->s (r5_mean;
     video-disjoint pseudo-dev on H2S)
     at step 0, phi(X) == Z exactly  ->  Step 2 starts from Step 1


 STEP 3  RE-RUN RETRIEVAL WITH TRAINED phi
 ===========================================================================
     X --> trained phi --> Z'  -->  frame (Eq. 5-8) | pool | s2t (sign->text)
                                    | t2s (text->sign) | --pair_recall (V2T/T2V R@K)
     same queries, filters, baselines, metrics  ->  delta over Step 1
     (--train_encoder runs: re-extract X with encoder.pth first)
```

### 4.2 Step 1: feature extraction (`signrag_retrieval.py extract`)

- **Model.** `models.Uni_Sign`, built from `utils.get_args_parser()`
  defaults (pose only, `--max_length 256`), checkpoint loaded with
  `strict=True`, `eval()`. ST-GCN runs in fp32.
- **Getting X without touching `models.py`.** Replace `model.pose_proj`
  with `nn.Identity()` and call `model.encode_pose(src_input)`. This
  returns \(X\) = concatenated part features + `part_para`; then
  \(Z=XW^\top+b\) is exactly the baseline's mT5 input. The first batch of
  every split checks this equality (assert, tolerance 1e-4).
- **Deterministic frames.** `S2T_Dataset.load_pose` subsamples clips longer
  than 256 frames with `random.sample`, even at eval.
  `S2T_Dataset_Uniform` (same file) overrides `load_pose` with uniform
  sampling, `np.linspace(0, duration-1, 256).round()`, and is otherwise
  identical. This affects ~21% of H2S clips and ~33–39% of OpenASL clips.
- **Files per split.**
  - `X.f16`: raw float16 `[sum T, 1024]`, read as `np.memmap`.
  - `offsets.npy`: `[N+1]`.
  - `meta.json`: dataset, split, checkpoint, sampling, feature dims, and
    per-clip name / text / video_id / length. It is written **last**, so
    its presence marks a finished split.
  - `--save_enc` adds `enc.f16` (F2: mT5 encoder states at the pose
    positions of `[prompt][pose]`, 768-d).
- **Per feature directory.** `pose_proj.pt` (W, b, source checkpoint) is
  the Step-1 \(\phi\).
- **Size.** ~2 KB per frame. The smoke sample averages 144–172 frames per
  clip after the cap, so H2S ≈ 10–11 GB and OpenASL ≈ 36 GB.

### 4.3 Step 1: scoring (`signrag_retrieval.py retrieve`)

- **frame**: `signrag_common.soft_match_scores`, Eq. 5–8 exactly:
  - per-frame \(\ell_2\) normalisation;
  - padding masked in the softmax and in the mean over query frames;
  - no temperature by default; `--frame_tau` adds one as an ablation.
- **pool** (control): cosine of the masked-mean projected features.
- **s2t** (Step 3 only): \(z_s\) against the gallery clips' text
  embeddings.
  - Pooled-objective \(\phi\): `head(mean mT5)`.
  - Fine-objective \(\phi\): the Stage-I soft alignment against text tokens
    with the learned \(\tau\).
- **t2s** (Step 3 only): the query clip's *sentence* against the gallery
  clips' sign embeddings, i.e. text→sign. It uses the same filters (a
  training query never retrieves its own clip, a clip with the identical
  sentence, or, on H2S, its own video).
  - Pooled-objective \(\phi\): `head(mean mT5)` vs pooled \(\phi(X)\).
  - Fine-objective \(\phi\): text tokens attend over the gallery frames.
- **`--pair_recall`** (Step 3 only): within-split V2T / T2V R@1/5/10 with
  the query set itself as gallery (the standard sign-retrieval protocol).
  Written to `pair_recall_<split>.json`.
- **Implementation.**
  - Gallery chunks are sorted by length, so padding stays small, and kept
    on the GPU, or streamed from pinned host memory if they exceed 60% of
    free memory.
  - Query batches are sorted by length and dealt round-robin to one
    process per GPU (`torch.multiprocessing.spawn`).
  - Each process keeps a full `[b, N]` score row per query batch, applies
    the filters, and takes an exact top-K. Shards are merged in the parent.
  - pool and s2t run as dense matmuls on one GPU.

### 4.4 Filters (applied before taking top-3)

| Query split | self | identical text | same video |
|---|---|---|---|
| H2S train | drop | drop | **drop**: test videos are disjoint from train, so same-video neighbours never exist at test |
| OpenASL train | drop | drop | **keep**: every test video also occurs in train; `--same_video on` is the ablation (`S1_E3_openasl_samevideo`) |
| test / dev (both) | — | keep | keep |

- Test queries are never filtered. The gallery is train only, so a test
  sentence that happens to occur verbatim in train is a legitimate find
  (3.3% of H2S test, 10.7% of OpenASL test).
- `save_results` re-checks every filtered top-K list and **asserts** that
  no rule was violated (smoke test S5).
- `--no_filter` disables all rules (smoke test S4).

### 4.5 Baselines (same queries, same gallery, same filters)

- **random**: uniform top-3 from the filtered gallery, 5 seeds; reported as
  mean (std in `metrics.json`).
- **tfidf_oracle**: cosine between the query's **GT text** and the gallery
  texts (sklearn `TfidfVectorizer`, word 1–2-grams, sublinear tf). It uses
  the reference, so it is an analysis-only upper bound. A dense
  sentence-embedding oracle comes later (R0 in §11).

### 4.6 Step 2: contrastive training (`signrag_contrastive.py`)

- **Sign side.**
  - \(\phi\) = `Phi`: `Linear(1024, 768)` initialised from the
    checkpoint's `pose_proj` (`--init pose_proj`, default) or randomly
    (`--init random`).
  - `--phi_mlp_hidden H` adds a residual 2-layer MLP with a zero-initialised
    output layer. It is still identical to Step 1 at step 0.
  - Pooled objective: \(z_s=\operatorname{Norm}(\operatorname{maskedmean}_t\,\phi(X_t))\).
- **Encoder switch (`--train_encoder`).**
  - **Off (default):** the ST-GCN is frozen and training reads the cached
    \(X\) (the whole split sits on one GPU).
  - **On:** the Uni-Sign model is loaded from the extract checkpoint, with
    `pose_proj` swapped for `Identity`, and the ST-GCN parameters join the
    optimiser at `--lr_encoder` (default 1e-5).
    - Training batches load poses through `S2T_Dataset` (random frame
      subsampling, the baseline's training behaviour).
    - Dev batches use `S2T_Dataset_Uniform` and re-encode with the current
      encoder.
    - Checkpoints also write `encoder.pth`, a full Uni-Sign
      `{'model': state_dict}` with the trained ST-GCN. Step 3 must
      re-extract \(X\) with it; `retrieve` refuses a mismatched feature
      directory.
    - The batch size is not limited by ST-GCN memory: `--encoder_chunk`
      (GradCache, §4.9) keeps the default batch 256.
- **Text side.**
  - The **official** mT5-base encoder (`config.mt5_path` =
    `pretrained_weight/mt5-base`, i.e. google/mt5-base), *not* the
    checkpoint's translation-finetuned mT5.
  - Input is the GT sentence alone (no prompt, `max_length 50`, as for
    labels); `--text_layer` picks the encoder layer (default final).
  - Masked mean-pool, then a text head `Linear(768, 768)` initialised to
    identity.
  - **Why the official weights.** The checkpoint's mT5 was finetuned on
    `[prompt][pose]` inputs, so its text-only behaviour may have drifted.
    The official encoder is a clean text model, independent of the sign
    side, as SignRAG's separate mBART text encoder is.
  - **One cache per dataset.** Token states (ragged `tokens.f16`) and their
    masked mean (`pooled.npy`) are computed once and shared by E1/E2/E3, so
    all encoders align to the same text space. The head is re-applied to
    the cache every step, so the gallery is never stale.
  - **Cost.** Official mT5 is span-corruption pretrained, not trained for
    sentence similarity, so the raw mean-pooled space is weak and the head
    adapts it only linearly. This is acceptable for the SignRAG-faithful
    step; the strong embedder comes in §11.
- **Gallery deduplication.** Normalised train texts are deduplicated (H2S
  29,981, OpenASL 84,399).
- **Positives (`--positives`).**
  - `exact`: the clip's own (normalised) text.
  - `semantic`: also every training text whose paraphrase-model cosine is
    ≥ `--pos_threshold` (default 0.85, at most `--pos_max` 64). Stored as
    CSR over the gallery columns and expanded per batch (`csr_rows_mask`).
  - In-batch positives are the same relation restricted to the batch.
  - The audit in §3.4 sets the threshold.
- **Loss (pooled objective).** With \(s_{ij}=z_{s,i}^\top z_{t,j}/\tau\):

\[
\mathcal{L}_{s\to t}=\frac{1}{B}\sum_i\Big[\log\sum_{j\in\mathcal{G}}e^{s_{ij}}-\log\sum_{j\in\mathcal{P}(i)}e^{s_{ij}}\Big],
\qquad
\mathcal{L}=\tfrac12\big(\mathcal{L}_{s\to t}+\mathcal{L}_{t\to s}^{\text{in-batch}}\big)
\]

  - \(\mathcal{G}\) is the full deduplicated training-text gallery and
    \(\mathcal{P}(i)\) the positive set (exact or semantic). This is the
    "any positive is a hit" (MIL-NCE) form, `--pos_loss any`;
    `--pos_loss all` is the SupCon form.
  - **t→s negatives (`--t2s_negatives auto|gallery|batch`).** With a
    linear \(\phi\) on a frozen encoder,
    \(\operatorname{maskedmean}(\phi(X))=\phi(\operatorname{maskedmean}(X))\)
    (checked numerically, S13). The per-clip mean \(\bar X\) is therefore
    precomputed once, and text→sign is scored against **all training
    clips** each step, at the cost of one `[N, 1024]×[1024, 768]` matmul.
    Positives: every clip whose text is in \(\mathcal{P}(i)\). `auto` picks
    this whenever it is exact (pooled objective, no MLP, frozen encoder);
    otherwise t→s is in-batch.
  - \(\tau\) is learnable, initialised to 0.07, clamped to \(\ge 0.01\).
- **Loss (fine objective, `--objective fine`).**
  - `soft_match_pair_scores` computes both Stage-I directions from one
    `[B, B, M, L]` similarity tensor (bf16 autocast), in checkpointed
    chunks of `--sim_chunk` texts (default 32, §4.9).
  - The attention temperature `--tau_attn_init` (\(\tau\)) and the InfoNCE
    temperature (\(\tau'\)) are both learnable.
  - Negatives are in-batch; positives are the same multi-positive relation.
  - `--slt_weight w` adds \(w\cdot\mathcal{L}_{slt}\) on the first
    `--slt_batch` clips of each batch: the checkpoint's frozen translation
    mT5 reads \(\phi(X)\) in place of `pose_proj(X)` (prompt and label
    smoothing 0.2 as in Uni-Sign). Two mT5 copies are therefore involved:
    the official encoder is the text tower; the checkpoint mT5 is the
    translation head.
- **Optimiser.** AdamW with lr 1e-4 for \(\phi\), 1e-3 for the head and
  temperatures (and `--lr_encoder` for the ST-GCN); weight decay 0.01 (0
  for temperatures). Linear warmup (≤ 100 steps), then cosine decay; grad
  clip 1.0. Batch 256, at most 30 epochs.
- **Model selection.**
  - H2S has no dev split, so 5% of train *videos* (`--dev_frac`, ~110
    videos, ~1.5k clips) are held out. OpenASL uses its real dev split.
  - Dev is evaluated in both directions by
    `signrag_common.bidirectional_recall`:
    - s→t R@K (gallery = dev texts);
    - t→s R@K (gallery = dev clips);
    - `sem_*` variants that count a paraphrase ≥ \(\theta\) as a hit.
  - The default selection metric is `r5_mean` = mean of s→t and t→s R@5
    (`--select`). Patience 5.
  - Caveat: for E1, the pseudo-dev clips were seen during translation
    training.
- **Outputs.**
  - `phi_init.pth` (step 0) and `phi.pth` (best).
  - With `--train_encoder`, also `encoder_init.pth` / `encoder.pth`.
  - `log.txt` (JSON lines per epoch), `args.json`, and `done.json` written
    at the very end; it is the finished-run marker for the script.

### 4.7 Step 3: retrieval with the trained \(\phi\)

Re-run §4.3–4.5 with `--phi phi.pth` and `--score frame pool s2t`. The
train-query sample is seeded identically to Step 1. Comparing **frame**
with **s2t** on the same trained head is the direct test of "sign→sign vs
text-anchored" retrieval.

### 4.8 Ours vs SignRAG Stage I

| | Ours (S2-a / S2-s) | SignRAG Stage I (reproduced as S2-d) |
|---|---|---|
| Sign encoder | pretrained Uni-Sign ST-GCN, frozen (`--train_encoder` trains it) + \(\phi\) initialised from `pose_proj` | CoSign + 3-layer Transformer, trained from scratch (200 epochs) |
| Text tower | frozen official mT5-base + linear head (identity init) | first 3 mBART layers |
| Similarity | pooled cosine (masked means) | frame↔token soft alignment with \(\tau\) |
| Temperatures | one \(\tau\) | \(\tau\) (attention) + \(\tau'\) (InfoNCE) |
| Negatives | s→t: the whole deduplicated train-text gallery (H2S 29,981; OpenASL 84,399); t→s: all training clips | in-batch only (batch 8 → 7 negatives) |
| Retrieval directions | both, trained symmetrically, selected on the mean | both in the loss; retrieval is sign→sign only |
| Positives | identical text; + paraphrases ≥ \(\theta\) (`--positives semantic`) | own pair only |
| Translation loss | none (S2-d adds \(\mathcal{L}_{slt}\)) | \(+\mathcal{L}_{slt}\) |
| Score used at retrieval | the trained score (pool / s2t) | sign→sign Eq. 5–8, no \(\tau\) (never trained) |
| Cost per step | one `[B, N]` matmul | one `[B, B, M, L]` tensor |

In our code, the fine objective also accepts semantic positives (S2-ds), so
"objective" and "positives" can be ablated independently.

### 4.9 Memory: the number of negatives is not what grows

Peak GPU memory, forward + backward, worst case (every clip 256 frames,
50 text tokens), measured on one L40S. Scripts are in the session
scratchpad (`mem_bench.py`, `chunk_tests.py`).

| Setting | Peak |
|---|---|
| pooled, B=256, full gallery: H2S 29,981 / OpenASL 84,399 texts | 0.88 / 1.97 GiB |
| pooled, B=1024, full gallery: H2S / OpenASL | 2.50 / 3.51 GiB |
| fine, in-batch, B = 32 / 64 / 128 / 256, unchunked | 0.46 / 1.58 / 5.89 / **22.73 GiB** |
| fine, B=256, `--sim_chunk` 32 / 16 | **3.55** / 2.17 GiB |
| `--train_encoder`, B = 16 / 32 / 64 / 128, unchunked | 8.17 / 16.30 / 32.58 / **OOM** |
| `--train_encoder`, B=256, `--encoder_chunk 16` (pooled, full H2S gallery) | **8.68 GiB** |

- **Full-gallery negatives are cheap.** The text side is cached, so the
  sign→text loss over the whole gallery is one `[B, N]` matrix plus the
  head over the cache. It needs no sign features of the negatives.
- **What grows with B:**
  - the fine objective's `[B, B, M, L]` tensor (∝ B²);
  - the ST-GCN activations when the encoder is trained (∝ B, ~8 GiB per
    16 clips).
- **`--sim_chunk c` (fine objective).** `soft_match_pair_scores` processes
  `c` texts at a time under `torch.utils.checkpoint`. Scores and gradients
  are unchanged (S12) and memory is ∝ B·c.
- **`--encoder_chunk c` (GradCache, Gao et al. 2021).** Two passes per step:
  1. Encode the whole batch in chunks of `c` without a graph, then compute
     the loss (all B clips as in-batch negatives, the full gallery for
     s→t) and backpropagate to \(\partial\mathcal{L}/\partial X\).
  2. Re-encode chunk by chunk with a graph and backpropagate the cached
     \(\partial\mathcal{L}/\partial X\) slices into the ST-GCN.

  Gradients equal direct backprop through the same chunks (S11). The cost
  is a second ST-GCN forward per step.
- **BatchNorm note.** In train mode, BN uses per-chunk batch statistics.
  The default `c = 16` matches Uni-Sign's own per-GPU training batch. BN
  running statistics are updated in both passes, with identical batch
  statistics.

## 5. Metrics (`signrag_retrieval.py evaluate`)

Computed for every `retrieval_<split>_<method>.json` in a run. All text
metrics use the normalisation of §3.2.

| Metric (column) | Definition | Answers |
|---|---|---|
| `bleu4`, `rouge_l` | `translation_performance(refs, top1_texts)`, i.e. sacreBLEU 13a + `rouge`, the same code as translation eval | how much translation signal the top-1 neighbour carries; directly comparable to E1's 14.50 |
| `f1@1`, `f1@3` | SignRAG Eq. 20: \(R=\lvert T_{GT}\cap T_{ret}(K)\rvert/\lvert T_{GT}\rvert\), \(P=\lvert T_{GT}\cap T_{ret}(K)\rvert/\lvert T_{ret}(K)\rvert\), per-query F1 then averaged; \(T_{ret}(K)\) = token set of the top-K texts | comparability with SignRAG |
| `cf1@1`, `cf1@3` | as above after removing sklearn `ENGLISH_STOP_WORDS` | removes function-word inflation, which matters for English |
| `em@1` | top-1 normalised text = normalised GT (%) | how often a verbatim answer is retrieved |
| `same_video@1` | top-1 shares the query's video ID (%) | exposes OpenASL's shared-video context |
| `train_test_gap_bleu4` | bleu4(train queries, filtered) − bleu4(this split) | memorisation / mismatch, the input to the cross-fitting decision |
| `gap_closed_bleu4` | \((m-m_{rand})/(m_{oracle}-m_{rand})\) on bleu4 | one number for "how strong" the retriever is |
| dev `r@K`, `t2s_r@K`, `sem_*`, `r5_mean` (Step 2 log) | within-dev recall in both directions: own text / any clip with the query text / paraphrase ≥ \(\theta\) | alignment quality, used for model selection |
| `pair_recall.tsv` (Step 3, `--pair_recall`) | the same bidirectional recall on test (and the train sample), gallery = that query set | V2T / T2V R@K comparable to sign-video retrieval work (CiCo, SignCLIP, SAN) |

For the **t2s** method, the same columns read text→sign. For example,
`bleu4` compares the query sentence with the sentence of the top-1
retrieved clip, i.e. whether the returned signing belongs to a similar
sentence. A direct motion-side metric (e.g. DTW between the retrieved and
the query clip's keypoints) is an open item.

The `rouge` package rejects texts without word tokens. Such texts are
replaced by a placeholder and counted in `n_placeholder` (`metrics.json`).

## 6. Experiment Matrix (`script/signrag_retrieval.sh`)

Run names are those used by the script. Retrieval runs (Step 1 and Step 3)
go to `$OUT/runs/<run>`; Step 2 runs go to `$OUT/step2/<run>`. Every
retrieval run also contains random and TF-IDF oracle on identical queries.

| Level | Run | What |
|---|---|---|
| p0 | `S1_E1_h2s` | Step 1, E1 → H2S, frame + pool |
| p0 | `S2a_E1_h2s` → `S3a_E1_h2s` | ours, exact positives |
| p0 | `S2s_E1_h2s` → `S3s_E1_h2s` | ours, semantic positives (\(\theta\) = `POS_TH`, default 0.85) |
| p0 | `S2b_E1_h2s` → `S3b_E1_h2s` | S2-s with random \(\phi\) init: is the translation-head init worth it? |
| p1 | `S1_E1_h2s_tau005` | frame score with \(\tau=0.05\): does a temperature fix the hub problem? |
| p1 | `S1_E2_h2s` | Step 1, cross-domain encoder E2 → H2S (memorisation control) |
| p1 | `S1_E3_openasl`, `S1_E3_openasl_samevideo` | Step 1 on OpenASL; same-video filter off (default) / on |
| p1 | `S2s09_E1_h2s` | S2-s at \(\theta=0.9\) (threshold ablation) |
| p1 | `S2c_E1_h2s` | S2-s + residual MLP \(\phi^+\) |
| p1 | `S2d_E1_h2s` | SignRAG Stage I objective + \(\mathcal{L}_{slt}\) |
| p1 | `S2dn_E1_h2s` | S2-d without \(\mathcal{L}_{slt}\) |
| p1 | `S2e_E1_h2s` | S2-d at batch **8** (SignRAG's batch) |
| p1 | `S2ds_E1_h2s` | fine objective + semantic positives |
| encoder | `S2fd_E1_h2s` → re-extract → `S3fd_E1_h2s` | S2-dn (fine objective) with the ST-GCN trained (`--train_encoder`, batch 256 via `--encoder_chunk 16`, ≤ 15 epochs, patience 4). It replaced the pooled `S2f` on 2026-10-22: S2f was stopped at epoch 28 (dev `r5_mean` 21.8, below the frozen S2s), once the fine objective proved far stronger. **Result (2026-10-22): training the encoder hurts.** Best dev `r5_mean` is 50.65 (epoch 13 of 16) vs 60.72 for the frozen S2dn. Test V2T/T2V R@1 27.3/29.6 vs 35.4/38.8; `r5_mean` 44.3 vs 55.0; s2t/t2s retrieval BLEU 1.44/2.27 vs 1.86/2.58. Keep the ST-GCN frozen |
| manual | `S2dn_E3_openasl` / `S2ds_E3_openasl` → `S3dn_E3_openasl` / `S3ds_E3_openasl` | OpenASL retrievers (E3 = E2 file), fine objective with exact vs semantic (θ=0.85) positives. This is the semantic-positive test on the dataset where paraphrases are common (15.6% of clips, §3.4); on H2S, S3ds ≈ S3dn. OpenASL's 37 GB feature cache stays in CPU memory (`--x_device auto`). **Result (2026-10-22): semantic positives do nothing here either.** Both ran all 30 epochs, dev `r5_mean` 70.19 vs 70.19. Test pair recall V2T/T2V R@1 51.7/51.1 (dn) vs 51.5/51.0 (ds); `r5_mean` 68.8 vs 68.9. Test s2t/t2s retrieval BLEU 4.35/4.96 (dn) vs 4.41/4.82 (ds); oracle 16.01, random 0.15. **Caveat:** the untrained Step-1 pool score has *higher* s2t BLEU (5.46) because 31.8% of its top-1 come from the same video. Every OpenASL test video also occurs in train, and same-video neighbours share topic. The trained retrievers almost never return same-video neighbours (same_video@1 1.5%). They are better at pair matching but not at retrieval BLEU on OpenASL |
| manual | `S2dnck_E1_h2s` → `S3dnck_E1_h2s` | S2-dn with the text tower switched to the translation-finetuned mT5 encoder of the E1 checkpoint (`--text_encoder checkpoint`). **Result (2026-10-22): worse than the official mT5.** Test V2T/T2V R@1 26.7/30.2 vs 35.4/38.8 (S3dn); `r5_mean` 44.5 vs 55.0; s2t/t2s retrieval BLEU 1.55/2.29 vs 1.86/2.58. The finetuned encoder only ever saw `[prompt][pose]` inputs, so its text-only states are weaker. Keep the official mT5 |

F2 features (`--save_enc`, `--feature enc`) are supported but not in the
script (P2).

## 7. Deviations from SignRAG

| SignRAG | Here | Why |
|---|---|---|
| Stage I trains the skeleton encoder from scratch (200 epochs, SGD) | start from **pretrained** Uni-Sign encoders (E1/E2) and train only the alignment head; the ST-GCN is trained only with `--train_encoder` (S2-f) | we already have pretrained encoders, and this isolates the alignment step |
| CoSign front end + 3-layer Transformer, 77 keypoints | Uni-Sign 4-part ST-GCN on RTMPose COCO-WholeBody | our encoder; same role (frame-level skeleton features) |
| Qwen3-8B + LoRA downstream | mT5-base (Uni-Sign) | our translator; irrelevant to retrieval in Steps 1–3 |
| \(\phi_q,\phi_g\) of unspecified origin | Step 1: `pose_proj`; Step 2: one shared \(\phi\) initialised from `pose_proj` | no Stage I in our pipeline; the translation head is the closest analogue |
| Stage I: first 3 mBART layers as text encoder, fine-grained similarity, batch 8, CSL-News | Step 2: frozen **official mT5-base** encoder + head; pooled InfoNCE with full-gallery negatives (fine-grained in S2-d/e/ds); identical + semantic positives | an independent pretrained text encoder, as in SignRAG; cheaper and stronger objective, theirs kept as an ablation |
| Same-video rule not discussed | dataset-specific (§4.4) | H2S test is video-disjoint; OpenASL test is not |

## 8. Implementation

No baseline file is modified.

### 8.1 Files

| File | Contents |
|---|---|
| `signrag_common.py` | `norm_text`, `video_id`; `FeatSplit` (ragged memmap features) and `ragged_padded` / `gather_padded`; `Phi`, `pose_proj_phi`, `load_phi_checkpoint`; `soft_match_scores` (Eq. 5–8 / 21–23) and `soft_match_pair_scores`; `Gallery` / `Exclusion` (filters); `TextCache` + `build_mt5_text_cache` / `build_sem_cache` (keyed by normalised text, fixed order train→dev→test); `semantic_neighbours` / `csr_rows_mask`; `multi_positive_nce` |
| `signrag_retrieval.py` | `extract` (with `S2T_Dataset_Uniform`, `build_unisign`), `retrieve` (frame / pool / s2t, multi-GPU worker, baselines, filter assertion), `evaluate` |
| `signrag_contrastive.py` | Step 2 trainer (pooled / fine objective, exact / semantic positives, `--train_encoder`, `--slt_weight`) and `--audit_positives` |
| `script/signrag_retrieval.sh` | the §6 matrix: `p0` / `p1` / `encoder` levels, skip-if-done, writes and prints `$OUT/summary.tsv` |

### 8.2 Commands

```bash
OUT=/data/hwu/Uni_sign_repo/out/signrag_retrieval
E1=/data/hwu/Uni_sign_repo/out/how2sign_pose_only/how2sign_pose_only_slt.pth

# whole matrix (skips finished stages, prints the summary)
bash script/signrag_retrieval.sh            # or: p1 | encoder ;  POS_TH=0.9 to change theta

# Step 1, by hand
python signrag_retrieval.py extract --dataset How2Sign --finetune $E1 --splits train test \
    --feat_dir $OUT/feats/E1_h2s
python signrag_retrieval.py retrieve --feat_dir $OUT/feats/E1_h2s --run_dir $OUT/runs/S1_E1_h2s \
    --score frame pool --queries test train:3000
python signrag_retrieval.py evaluate --run_dir $OUT/runs/S1_E1_h2s

# Step 2 + Step 3, by hand
python signrag_contrastive.py --feat_dir $OUT/feats/E1_h2s --run_dir $OUT/step2/S2s_E1_h2s \
    --positives semantic --pos_threshold 0.85         # [--objective fine --slt_weight 1] [--train_encoder]
python signrag_retrieval.py retrieve --feat_dir $OUT/feats/E1_h2s --run_dir $OUT/runs/S3s_E1_h2s \
    --phi $OUT/step2/S2s_E1_h2s/phi.pth --score frame pool s2t t2s --pair_recall --queries test train:3000
python signrag_retrieval.py evaluate --run_dir $OUT/runs/S3s_E1_h2s

# semantic-positive audit (label files only)
python signrag_contrastive.py --audit_positives --dataset How2Sign --run_dir $OUT/audit_positives/How2Sign
```

Useful `retrieve` options:

- `--queries` takes `split` or `split:N` (seeded sample).
- `--feature X|enc`, `--frame_tau`, `--same_video auto|on|off`,
  `--no_filter`, `--no_baselines`.
- `--k`, `--gpus`, `--query_batch` (8), `--gallery_chunk` (128).

### 8.3 Output layout and formats

```text
$OUT/
  feats/<tag>/pose_proj.pt
  feats/<tag>/<split>/X.f16, offsets.npy, meta.json[, enc.f16]
  text_cache/<dataset>/mt5_layer-1/pooled.npy, tokens.f16, tok_offsets.npy, meta.json   (built: How2Sign)
  text_cache/<dataset>/sem_sentence-transformers__paraphrase-MiniLM-L6-v2/pooled.npy, meta.json (built: both)
  audit_positives/<dataset>/positive_audit.json                                           (built: both)
  runs/<run>/retrieve_args.json, retrieval_<split>_<method>.json,
             per_query_<split>_<method>.csv, metrics.json, metrics.tsv
             [, pair_recall_<split>.json, pair_recall.tsv]
  step2/<run>/args.json, log.txt, phi_init.pth, phi.pth, done.json[, encoder_init.pth, encoder.pth]
  summary.tsv, summary_pair_recall.tsv
```

- `retrieval_<split>_<method>.json`:
  `{"split", "method", "results": [{"query": {name, text, video_id}, "topk": [{name, text, video_id, score} × K]}]}`.
  Methods: `frame`, `pool`, `s2t`, `random_s0..4` (aggregated as `random`
  in the metrics), `tfidf_oracle`.
- `per_query_*.csv`: query, gt, top-1 name / text / score, exact_match,
  same_video.
- `phi.pth`: `phi`, `phi_cfg`, `text_head`, `text_cache` (absolute path),
  `objective`, `tau`, `tau_attn`, `positives`, `pos_threshold`, `epoch`,
  `dev`, `args`, `encoder` (path to `encoder.pth`, or `None`).

### 8.4 Compute (measured on the smoke runs, extrapolated)

| Stage | Measured | Extrapolated |
|---|---|---|
| Extraction (16 workers) | 3,000 H2S clips in 19 s | H2S (33.4k clips) ≈ 4 min; OpenASL (98k, longer clips) ≈ 15 min |
| Step 1 frame retrieval (4 GPUs) | 200 queries × 3,000 gallery: 6 s (of which ~3 s is process spawn); pool 1 s | H2S, 5.3k queries × 31k: ≈ 15 min; full H2S train queries (needed for RA-SFT): ≈ 1.5 h; OpenASL, 4k × 96k with longer clips: ≈ 1 h |
| Step 2, cached X | 160 clips: 0.1–0.3 s/epoch (pooled), 1–2 s (fine + \(\mathcal{L}_{slt}\)) | H2S pooled: well under a minute per epoch |
| Step 2, `--train_encoder` | 160 clips: 7–10 s/epoch (8 workers), including the GradCache second pass | H2S: pose loading bound, ~5 min/epoch |
| GPU memory | §4.9 | every configuration in §6 fits one L40S at batch 256 |
| Text caches | built once per dataset | — |

### 8.5 Smoke-test definitions

| ID | Check |
|---|---|
| S1 | extracting the same clips twice gives bit-identical `X` |
| S2 | `pose_proj(X)` equals `encode_pose` of the unmodified model |
| S3 | vectorised Eq. 5–8 equals a naive per-pair loop; pair scores equal two single-direction calls |
| S4 | unfiltered train queries retrieve themselves at top-1 |
| S5 | filtered train top-K never contains self, identical text or (H2S) same video (asserted in code on every run) |
| S6 | retrieval with `phi_init.pth` reproduces Step 1 exactly |
| S7 | `--train_encoder` at init gives the same dev metrics as the cached path |
| S8 | `retrieve` refuses an encoder-trained \(\phi\) on features from another encoder; re-extraction with `encoder.pth` then works |
| S9 | `--save_enc` / `--feature enc` (F2) run end to end |
| S10 | `script/signrag_retrieval.sh` runs p0 end to end, and a second invocation skips every stage |
| S11 | GradCache (`--encoder_chunk`) gives the same ST-GCN and \(\phi\) gradients as direct backprop through the same chunks |
| S12 | `--sim_chunk` gives the same fine-objective loss and gradients as the unchunked computation |
| S13 | masked_mean(\(\phi\)(X)) = \(\phi\)(masked_mean(X)) for the linear \(\phi\) (basis of the full-gallery t→s loss) |
| S14 | bidirectional training, dev selection on `r5_mean`, and Step 3 with `t2s` + `--pair_recall` run end to end for pooled and fine \(\phi\) |

### 8.6 Smoke-test results (2026-10-21)

**Setup.** E1 checkpoint; How2Sign, first 200 train clips + first 200 test
clips (`--limit 200`); train queries = all 200; outputs in the session
scratchpad. The audit in §3.4 ran on the full label sets.

| ID | Result |
|---|---|
| S1 | **pass**: `cmp` of two extractions of 200 train clips is identical |
| S2 | **pass**: max \|diff\| = 0.00e+00 on train and test (also for `encoder.pth`) |
| S3 | **pass**: frame score max \|vec − naive\| = 4.9e-06 (fp16); pair scores vs single calls: s2t 0.0, t2s 1.2e-07 |
| S4 | pool: self@1 = **100%**; frame (no \(\tau\), faithful Eq. 5–8): self@1 = **94%**; frame with `--frame_tau 0.05`: **100%** |
| S5 | **pass**: no assertion fired in any filtered run |
| S6 | **pass**: top-3 lists with `phi_init.pth` identical to Step 1 for frame and pool, test and train |
| S7 | **pass**: epoch-0 dev r1 / r5 / r10 = 2.5 / 25.0 / 35.0 for both the cached and the `--train_encoder` path |
| S8 | **pass**: `ValueError … re-extract features first: signrag_retrieval.py extract --finetune …/encoder.pth`; after re-extraction, frame / pool / s2t run |
| S9 | **pass**: F2 extract (100 clips) + frame / pool retrieval |
| S10 | **pass**: p0 ran end to end; the second invocation printed `[skip]` for all 9 stages |
| S11 | **pass** (2026-10-22; B=16, chunks of 4, deterministic cuDNN): ST-GCN grad max diff 3.7e-09 (max \|grad\| 5.5e-02, cosine 0.99999994); \(\phi\) grad diff 0.0 |
| S12 | **pass** (2026-10-22; B=32, chunk 7, bf16 autocast): loss 6.935583 both; max grad diff 1.9e-07 (sign), 3.0e-07 (text), 5.9e-05 (\(\tau\)) |
| S13 | **pass** (2026-10-22; 200 clips): max \|diff\| 7.6e-06 at values up to 43.9 (fp32 rounding) |
| S14 | **pass** (2026-10-22). Pooled + semantic 0.85 + t→s over all training clips: dev `r5_mean` 21.25 → 30.0. Fine: 21.25 → 28.75. Step 3 `pool s2t t2s --pair_recall` (pooled \(\phi\)) and `s2t t2s --pair_recall` (fine \(\phi\), multi-GPU token workers) + evaluate wrote `metrics.tsv` and `pair_recall.tsv` |

Training-loop runs with the memory options (2026-10-22, smoke set):
`--train_encoder --batch_size 128` (GradCache, chunk 16) and
`--objective fine --slt_weight 1.0 --batch_size 128` (`--sim_chunk 32`)
both train end to end. Dev R@5 went 25.0 → 30.0 and 22.5 → 30.0
respectively.

Step 2 variants on the smoke set (160 train clips, 40 pseudo-dev clips with
`--dev_frac 0.2`, 2 epochs). These only show that every path trains; the
numbers are not meaningful at this size.

| Variant | Command flags | dev R@5, epoch 0 → best |
|---|---|---|
| pooled, exact | `--batch_size 32` | 25.0 → 27.5 |
| pooled, semantic (\(\theta\)=0.85; 3.8% of texts gain positives) | `--positives semantic --pos_threshold 0.85` | 25.0 → 27.5 |
| fine (SignRAG Stage I) + \(\mathcal{L}_{slt}\) | `--objective fine --slt_weight 1.0 --slt_batch 8 --batch_size 16` | 22.5 → 32.5 |
| pooled + `--train_encoder` | `--train_encoder --batch_size 16` | 25.0 → 35.0 |

Step 1 metrics on the smoke set (gallery = 200 clips, so far below the real
31k). Again plumbing only: the oracle is far above everything, and frame
and pool are near random at this gallery size.

| split | method | bleu4 | rouge_l | f1@1 | cf1@1 | cf1@3 | em@1 |
|---|---|---|---|---|---|---|---|
| test | frame | 0.28 | 6.80 | 9.43 | 3.90 | 4.62 | 0.0 |
| test | pool | 0.34 | 7.76 | 10.72 | 3.12 | 3.53 | 0.0 |
| test | random (5 seeds) | 0.28 | 7.07 | 9.68 | 1.71 | 2.50 | 0.0 |
| test | tfidf_oracle | 3.65 | 18.88 | 25.90 | 15.31 | 14.16 | 0.5 |

In the script dry run (S10), the default batch 256 exceeds the 160 smoke
training clips, so S2-a performed zero steps. Its Step 3 frame/pool rows
are therefore identical to Step 1, a second confirmation of the step-0
identity.

**Finding from S4.** Without a temperature, the softmax over gallery frames
is nearly uniform (cosines lie in [−1, 1]), so Eq. 5–8 approaches the
*mean* pairwise frame similarity. Generically similar "hub" clips can then
outscore the query itself. This is a concrete weakness of the faithful
SignRAG score; the p1 run `S1_E1_h2s_tau005` measures it at full scale.

## 9. Readouts and Decision Rules

The numeric thresholds below are heuristics to be revisited once the
full-scale oracle numbers exist.

- **Frame vs pool (Step 1).** If frame ≤ pool, SignRAG's frame-level score
  adds nothing on our encoder. Report it and keep pool as the cheap
  default. `S1_E1_h2s_tau005` shows whether a temperature changes this.
- **Retriever vs random.** If Step 1 is near random on H2S test, SignRAG's
  retriever carries no signal there. That is direct evidence for H-weak.
- **Gap closed.** Low values (e.g. < 20%) mean large headroom for a strong
  retriever. This sizes the paper's main claim.
- **Train–test gap, E1 vs E2.** If the gap is large for E1 and small for
  E2, memorisation is confirmed, which justifies cross-fitting before
  RA-SFT.
- **Oracle ceiling.** If even the TF-IDF oracle's retrieval-only BLEU-4 on
  H2S is very low, RAG has little room on open-domain H2S. OpenASL
  (boilerplate, shared videos, 10.7% verbatim test texts, 15.6% of clips
  with ≥ 0.85 paraphrases) then becomes the main showcase.
- **Step 2/3.**
  - S3-a vs S1: the gain from contrastive training.
  - S2-s vs S2-a: the gain from semantic positives (expected modest on H2S,
    §3.4).
  - S2-s vs S2-s09: threshold 0.85 vs 0.9.
  - S2-s vs S2-b: the value of translation-head init.
  - S2-d vs S2-a: their objective vs ours.
  - S2-dn vs S2-d: the effect of \(\mathcal{L}_{slt}\).
  - S2-e vs S2-d: the cost of batch 8.
  - S2-f vs S2-s: frozen vs trained encoder.
  - s2t vs frame on the same head: sign→text vs sign→sign.

## 10. Risks and Open Questions

- **No H2S dev.** Step 1 is evaluated on test. This is acceptable only
  because Step 1 has no trained parameters and its variants are fixed in
  advance. Step 2 selects epochs on the pseudo-dev, and test is read once.
- **Encoder memorisation (E1).** E1 has trained on every gallery clip, so
  train-query numbers are optimistic. E2 is the control for this.
- **OpenASL shared videos.** Retrieval may look strong because of shared
  video context rather than sign understanding. Same-video@1 makes this
  visible.
- **Semantic positives can be wrong.** The paraphrase model scores
  sentences that differ only in numbers, order or left/right as
  near-identical (§3.4). A lexical guard is an open item.
- **Weak text tower.** Official mT5 is span-corruption pretrained, and its
  mean-pooled states are a weak sentence space. Step 2 is therefore a
  lower bound on what a contrastive head can do; a strong frozen sentence
  embedder is part of §11.
- **t2s quality is measured through text.** "Returns similar signing" is
  currently scored by the similarity of the retrieved clip's sentence to
  the query sentence, plus T2V recall. A motion-side metric (keypoint DTW
  to the query's own clip) is an open item.
- **`--train_encoder` disk use.** Every new best epoch writes a full
  Uni-Sign checkpoint (`encoder.pth`, ~2.3 GB, overwritten).
- **`/data` is NFS.** It hung Bash calls in Aug 2026 (fine since late
  Sep). Long extractions should run with timeouts or in the background.

## 10b. Word-Level Hints: Zero-Shot Word Spotting (plan 1, 2026-10-22)

**Motivation.** On How2Sign test, 94.7% of reference content words occur in
the training vocabulary. The top-3 *sentences* cover only 37.3% (TF-IDF
oracle) and 13.1% (S3dn s2t). Word-level hints therefore have a much
higher ceiling than sentence-level retrieval.

**Method.** `signrag_wordspot.py` reuses S2dn (fine objective) without
training.

- Every training content word (9,475 types with clip count ≥ 2; ≤ 32
  contextual instances each, 90,292 in total) is embedded as the mean of
  its sub-word token states (mT5 text tower + trained head).
- A clip scores each word by its best-matching frame: `maxsim` (best
  instance) or `proto` (mean instance). The `*_z` variants are z-scored per
  word over 2,000 training clips.
- Outputs: `$OUT/word_spotting/S2dn_zero_shot/`
  (`metrics.tsv`, `per_query.csv`) and `translator_baseline.json`.

**Results (H2S test, content words of the reference).**

| Method | P@1 | P@5 | R@5 | R@10 | hit@10 |
|---|---|---|---|---|---|
| maxsim | 25.1 | 12.4 | 10.6 | 13.2 | 51.9 |
| proto | 26.3 | 12.3 | 10.6 | 12.8 | 50.0 |
| maxsim_z / proto_z | 0.0 / 12.2 | 0.0 / 7.2 | 0.0 / 6.4 | 0.0 / 9.0 | 0.1 / 38.7 |
| frequency prior (same words for every clip) | 20.4 | 11.3 | 8.2 | 12.5 | 52.1 |
| words of the top-3 retrieved sentences (S3dn) | 19.0 | 8.6 | 7.8 | 10.5 | 40.5 |
| random words | 0.0 | 0.1 | 0.1 | 0.1 | 0.6 |
| **Uni-Sign baseline translation** (≈ 6 content words) | — | precision **36.5** | recall **34.0** | — | hit 77.6 |

**Reading.**

- Zero-shot spotting is barely above the frequency prior. Predictions are
  dominated by generic words (use, make, want, we'll, today) plus unrelated
  ones. Only 619 distinct top-1 words appear over 2,349 clips.
- **Cause.** The fine objective normalises each frame's attention only
  over the tokens of *one* sentence. Absolute frame–word similarities are
  therefore not comparable across the vocabulary.
- z-scoring over-boosts rare words and makes things worse.
- **The translator itself already outputs content words at 36.5% precision
  / 34.0% recall**, about 3× the zero-shot spotter. Hints from this spotter
  would mostly add noise.
- **Bar for any word-hint source** (plan 2: a trained MIL word detector; or
  an external sign dictionary such as WLASL, which adds information the
  translator never saw): it must beat ~36% precision at ~6 words, or be
  complementary to the translator's own words.

## 10b2. Word-Level Dictionaries: Coverage (2026-10-23)

**Resources.**

- **WLASL** (2,000 glosses): Uni-Sign's RTMPose poses and ISLR
  checkpoint are downloaded to `/data/hwu/Uni_sign_repo/data/WLASL/`
  (`wlasl_pose_format.zip`, `wlasl_pose_only_islr.pth`). Uni-Sign pose-only
  WLASL2000 P-I/P-C is 63.13/60.90.
- **ASL-LEX 2.0** (2,723 entries, 2,309 unique English keys): OSF
  `signdata.csv` is downloaded to `/data/hwu/Uni_sign_repo/data/ASL-LEX/`.
  It has translations, lexical class, frequency, phonology and Signbank IDs,
  but no videos. Videos for all ASL-LEX signs exist in ASL Citizen.
- The two vocabularies overlap on 1,383 keys. The union is 2,926; 926
  keys are ASL-LEX only.
- **ASL Citizen** (2,731 ASL-LEX signs, about 84k RGB videos, 52 signers):
  not downloaded. It is one 45.9 GB zip (direct wget from
  download.microsoft.com, non-commercial research license) with no poses,
  so RTMPose would have to be run (`demo/pose_extraction.py`).
- **WLASL setup check.** The poses are unzipped to
  `/data/hwu/Uni_sign_repo/data/WLASL/pose_format/{train,dev,test}`
  (14,289/3,916/2,878 files, matching the labels), linked from
  `dataset/WLASL/pose_format`. The official ISLR checkpoint gives test
  P-I/P-C **62.96/60.48** (paper 63.13/60.90), so the WLASL pipeline is
  reproduced. Output: `$OUT/word_dict/wlasl_islr_eval/`.

**Coverage of the content vocabulary.** Content words are lemmatised
(lemminflect, sklearn stop list). A word counts as covered if one of its
lemmas is a dictionary entry. "+translations" also accepts ASL-LEX's
non-dominant and Signbank English translations, which is looser.
Script: session scratchpad `dict_coverage.py`.

| corpus | dictionary | content-token cov. | covered words / sent (of 7.4) | sent ≥ 1 | sent ≥ 3 |
|---|---|---|---|---|---|
| H2S train | WLASL | 56.0% | 4.23 | 93.7% | 64.9% |
| H2S train | ASL-LEX entries | 54.9% | 4.16 | 93.7% | 64.2% |
| H2S train | WLASL + ASL-LEX (+tr) | 76.8% | 6.38 | 97.6% | 79.9% |
| H2S test | WLASL | 54.7% | 4.13 | 91.3% | 63.4% |
| H2S test | WLASL + ASL-LEX (+tr) | 75.2% | 6.26 | 96.2% | 77.9% |
| OpenASL train | WLASL | 53.8% | 3.94 | 92.8% | 63.3% |
| OpenASL train | WLASL + ASL-LEX (+tr) | 69.8% | 5.37 | 95.9% | 74.9% |

About half of the content tokens have a WLASL sign, and almost every
sentence has candidate words. Coverage is an upper bound: a subtitle word
need not be signed in the clip.

**Examples per word.** This counts the train subtitles that contain each
single-word WLASL gloss (1,955 of 2,000), using lemma matching. Script:
scratchpad `word_counts.py`.

| corpus | glosses found | ≥ 10 | ≥ 100 | median per gloss | total occurrences |
|---|---|---|---|---|---|
| How2Sign train | 1,633 | 1,048 | 272 | 12 | 129k |
| OpenASL train | 1,786 | 1,514 | 735 | 52 | 379k |
| both | 1,797 | 1,632 | 892 | 83 | 509k |

**Subtitle-constrained spotting prototype (2026-10-23).**
- Setup: the official WLASL ISLR model with no training. The input is 300
  random H2S train sentences. Each clip is normalised once, then cut into
  windows of 16/24/32/48/64 frames with stride 4. Each window gets
  beam-10 gloss predictions.
- A subtitle gloss counts as spotted when it is the top-1 (top-k) of some
  window.
- Control: the WLASL glosses of another sentence, minus this sentence's
  glosses, are scored on the same windows.
- Script: scratchpad `spot_proto.py` (analysis `spot_analyze.py`). Output:
  `$OUT/word_dict/spot_proto/`. Runtime is 12 min on one L40S.

| candidates | n | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| this sentence's glosses | 1,406 | **33.6%** | 49.4% | 56.5% |
| other sentence's glosses (control) | 1,273 | 1.5% | 5.4% | 9.9% |
| this sentence's words (glosses merged per word) | 1,294 | 36.8% | 53.0% | — |

- **Localisation.** There are 74 sentences with ≥ 3 top-1 words. Across
  them, the spotted frame follows the subtitle word order with mean
  Spearman **0.86** (shuffled −0.02), and the correlation is positive in
  96% of them.
- **Precision.** The control rate suggests that about 1.5 / 33.6 ≈ 5% of
  top-1 hits are chance hits.
- 431 of the 476 top-1 hits come from the shortest window (16 frames,
  about 0.7 s), so shorter windows should be tried.
- **Conclusion.** The WLASL model transfers to continuous H2S signing well
  enough to mine word → segment pairs. About a third of the
  dictionary-covered subtitle words are found, with high precision and
  correct order.

## 10c. RA-SFT on How2Sign (started 2026-10-22)

**Code.**
- `datasets_rag.py`: `S2T_Dataset_RAG` renders the top-3 retrieved training
  sentences as `"Similar sentences: s1 | s2 | s3"` in `signsmd_text`, with
  retrieval dropout 0.5 in training.
- `fine_tuning_rag.py`: the SignSMD training loop with the unchanged
  `Uni_Sign_SignSMD` model, input `[retrieved][prompt][pose]`. It has zero
  new parameters, and the E1 checkpoint loads with `strict=True`.
- `script/train_rag_h2s.sh`: runs the three variants in parallel, one GPU
  each, and prints a BLEU summary.
- deepspeed needs `CUDA_HOME=/home/student/hwu/miniconda3/envs/Uni-Sign`;
  the script sets it.

**Retrieval inputs.** `$OUT/rag_inputs/S3dn_E1_h2s/`.

- `retrieval_train_s2t.json`: all 31,086 training clips. Filters: self,
  identical text, same video. None of the top-3 lists contains the clip's
  own sentence.
- `retrieval_test_s2t.json`: 2,349 test clips, unfiltered. 4 of them
  retrieve a verbatim training duplicate.

**Runs** (`$OUT/ra_sft/H2S_{real,empty,random}`). All warm-start from E1
with identical settings: 10 epochs, batch 8 × grad-accum 8, AdamW lr 1e-4,
fp32.

- `real`: S3dn retrieval.
- `empty`: no sentences. This controls for extra fine-tuning.
- `random`: 3 random training sentences per clip. This controls for "more
  tokens".

Checkpoints are selected on H2S test BLEU-4, as in the existing H2S
pipeline and for the baseline (14.50). The comparison that matters is
**real vs empty vs random** under the same protocol.

**Zero-shot check: E1 with the RA-SFT input, no RA-SFT training**
(`$OUT/ra_sft/eval0_E1_{mode}`, `fine_tuning_rag.py --eval`, H2S test):

| input | BLEU-4 | ROUGE-L | mean pred. length |
|---|---|---|---|
| empty | 14.48 | 34.45 | 17.1 |
| real (S3dn top-3) | 12.78 | 32.99 | 18.8 |
| random (3 random train sentences) | 11.44 | 32.16 | 20.6 |

- `empty` reproduces the baseline (14.50 / 34.51): the extra `</s>` is harmless.
- Untrained, any prefix hurts. Retrieved sentences hurt less than random
  ones (−1.7 vs −3.0 BLEU-4), so the model does read the prefix and the
  retrieved content carries some signal.
- After 1–4 epochs of RA-SFT, all three variants are back at ≈ 15.
- E2 (OpenASL checkpoint) zero-shot on H2S: empty 5.13 / 20.73, real
  5.06 / 20.37. This is the starting point of the E2 runs.

**E1 results (finished 2026-10-23, 10 epochs, H2S test BLEU-4 per epoch).**

| run | ep 0–9 | best (epoch) | mean of last 5 |
|---|---|---|---|
| real | 14.87 15.04 14.75 14.90 14.65 14.69 14.81 14.78 14.66 14.82 | 15.04 (1) | 14.75 |
| empty | 15.13 14.90 14.84 14.98 14.88 14.72 14.99 14.81 14.79 15.11 | 15.13 (0) | 14.88 |
| random | 14.73 15.02 14.67 14.79 14.67 14.54 14.66 14.73 14.71 14.94 | 15.02 (1) | 14.72 |

**No retrieval effect on top of E1.** real ≈ empty ≈ random, all within
the ±0.2 epoch-to-epoch noise. The ≈ +0.5 over 14.50 comes from extra
fine-tuning (plus best-epoch selection on test).

**E1 vs E2 start (decided 2026-10-22).** The E1 runs are **not** SignRAG's
setting. In SignRAG, the translator enters RA-SFT after Stage II
pretraining (YouTube-ASL / CSL-News) and has never been fine-tuned on the
target set. RA-SFT *is* the first target SFT, and its retrieval gain
(CSL-Daily 29.72 → 31.22) is SFT vs RA-SFT from the same pretrained
start. E1 has already fit H2S train without retrieval, so the E1 runs
measure "retrieval on top of an already fine-tuned model". They are kept as
an ablation.

The SignRAG-faithful runs are `INIT=E2 bash script/train_rag_h2s.sh`:

- Start from E2 (OpenASL, never fine-tuned on H2S) → `$OUT/ra_sft/H2S_E2_{real,empty,random}`.
- 20 epochs, batch 8 × accum 24 on one GPU, which gives effective batch 192.
  This matches the 3-GPU × 8 × 8 OpenASL → H2S protocol of
  `finetune_h2s_signsmd_from_openasl.sh`.
- `empty` is the baseline here (E2 → H2S SFT without retrieval), not 14.50.
- The retriever stays S3dn (E1 features). This is also faithful: SignRAG
  fine-tunes its sign encoder on each downstream training set before
  building the gallery.
- Replacing it with an E2-feature retriever, to remove the train/test
  neighbour-quality gap (s2t BLEU 2.29 train vs 1.86 test), would go beyond
  SignRAG's setting. It is left as an ablation.

The E2 runs are queued to start when the E1 script exits. Their log is
`rag_h2s_e2.out` in the session scratchpad.

## 11. Roadmap After This Experiment

1. **R0 dense oracle.** A strong frozen sentence embedder (e.g.
   Qwen3-Embedding or bge/e5; HuggingFace is reachable) gives the oracle
   text→text retrieval. This yields the true RAG ceiling per dataset.
2. **Strong retriever (R1).**
   - Sign tower aligned to the frozen strong text space (LiT-style).
   - Full-gallery softmax and multi-positive.
   - KL distillation from the R0 oracle distribution (soft positives,
     generalising the hard semantic threshold), i.e. "rank like the
     oracle", which is what RA-SFT actually needs.
3. **Reranking (R2).** Over the top-100:
   - late interaction (frame × token MaxSim, ColBERT-style);
   - generative reranking with the Uni-Sign translator itself,
     \(\log p(\text{candidate}\mid\text{pose})/\lvert\text{candidate}\rvert\).
4. **Cross-fitting.** Two half-train retrievers produce train-query
   neighbours, so train-time retrieval quality matches test time.
5. **RA-SFT.**
   - A zero-parameter subclass in the style of `models_signsmd.Uni_Sign_SignSMD`
     with mT5 input `[retrieved texts][translate prompt][pose]`.
   - Retrieval dropout 0.5.
   - Controls: retrieval always empty (= baseline) and random retrieval.
6. **Dose–response analysis.** Replace a fraction \(r\) of test-time
   retrievals with oracle neighbours and plot final BLEU against retrieval
   quality. A steep curve shows retrieval quality is the bottleneck, with
   SignRAG-style and our retrievers as points on it. This is analysis only,
   because the oracle uses the reference.

### 11.1 Update 2026-10-25: evidence and the agreed design

**Evidence** (details in `doc/signrag_progress.md`, line E).
- Hint training with TF-IDF oracle sentences (`*_oracle_d`) gives +2.45
  B4 on CSL and +1.9 on H2S. The same pipeline with S3dn hints gives about
  0, and random sentences give 0.
- RAG therefore works once retrieval is good enough. The gap is retrieval
  quality, not the injection.
- Retrieval BLEU (top-1 as translation, test):

  | | CSL | H2S |
  |---|---|---|
  | S3dn | 5.30 | 1.86 |
  | translate, then TF-IDF on the translation | 10.71 | 3.20 |
  | TF-IDF on the GT (ORACLE) | 18.02 | 7.8 |

- S3dn already matches pairs well (CSL test paired R@1 59). It fails
  because it was trained for instance discrimination: only the identical
  sentence is a positive. RAG instead needs the closest *other* sentence,
  since a test sentence is almost never in the gallery.
- An oracle-trained model given weak hints at test time drops below the
  baseline (CSL 24.60). Train-time hints must match test-time quality, so
  train hints of a strong retriever must be cross-fitted (cf. CTC: self-fit
  WER 3.6 vs cross-fit 32.8).

**Agreed design (user, 2026-10-25): a strong, utility-aligned retriever.**
1. **Objective (agreed first step): graded relevance.** Soft targets over
   the training sentence gallery come from the TF-IDF (or BLEU) similarity
   between the query's GT translation and each candidate. The pose→text
   distribution is fitted to them with KL (R1 above), keeping the
   exact-pair contrastive term. Negatives come from the whole gallery
   (cached text embeddings).
2. **Capacity.**
   - Sign tower: the Uni-Sign SLT encoder (ST-GCN + mT5 encoder over
     `[prompt][pose]`, from the SLT checkpoint), top layers or LoRA
     trainable. The translator's encoder already carries enough content
     (translate-then-retrieve = 10.71).
   - Text tower: a strong multilingual sentence encoder (e.g. bge-m3) or
     mT5 with trainable top layers. The translation-finetuned mT5 text tower
     was worse before (S2dnck).
3. **Rerank (R2).** The pool is the bi-encoder top-100 ∪ translate-then-
   TF-IDF top-50, reranked by the translator's
   `log p(candidate | pose)` (`rerank_retrieval.py`). The test-only ceiling
   check started on 10-25 for CSL.
4. **Diversity.** Top-K distinct texts (`--distinct_text`, cf. Cheng et
   al. EMNLP 2022).
5. **Cross-fitting** of train-time hints (K folds).
6. **Evaluation.** Retrieval BLEU and hint F1@3 against the three rows
   above. RA-SFT follows only once the retriever is close to the oracle.
7. **SLG direction.** Text→pose retrieval with the same towers and graded
   pose similarity. An oracle SLG experiment comes first: GT-nearest pose
   exemplars into the generator, measured by DTW / back-translation.

### 11.2 Contribution framing (agreed 2026-10-25, to be supported by experiments)

**Utility-aligned retrieval augmentation for SLT.** Existing sign RAG
(SignRAG) and our S3dn train the retriever for instance pairs. Yet a test
sentence is almost never in the gallery, so RAG needs the closest *other*
sentence.

| Claim | Content | Evidence | Status |
|---|---|---|---|
| C1 diagnosis | Instance-trained retrievers fail for SLT RAG. The bottleneck is retrieval quality, not hint injection | Paired R@1 59 vs retrieval BLEU 5.3; ORACLE sentence hints +2.45 (CSL) / +1.9 (H2S); random 0; S3dn ≈ 0 | done |
| C2 draft-space retrieval | Translate first, then retrieve with the draft in text space ("translate, retrieve, refine") | Retrieval BLEU 10.71 vs 5.30 (CSL), 3.20 vs 1.86 (H2S) | needs RA-SFT (`CSL_twopass_*`) |
| C3 matched hint quality | Train-time drafts must carry test-like errors: cross-fitted translators, or synthetic drafts calibrated to the test draft BLEU | Mismatch hurts: oracle-trained model with weak hints 27.63 → 24.60 | needs `twopass_xfit` vs `twopass_selffit` |
| C4 learned graded reranker | Rank the candidate pool by learned similarity to the answer | Pool ORACLE best 16.84 vs 10.71 now | to implement |
| Analysis | Dose–response: hint quality → BLEU gain | Mixes of oracle and real hints | to run |
| Extension | Bidirectional (SLG: text → pose retrieval) | ORACLE SLG experiment first | to run |

Prior art to cite for C2/C3 (the components are not new alone):
Iter-RetGen (Findings EMNLP 2023), HyDE / GAR, deliberation networks
(NeurIPS 2017), cross-fitting / jackknifing, synthetic APE drafts
(Junczys-Dowmunt & Grundkiewicz 2016). One search found no SLT work with
draft-guided retrieval. A systematic check is needed before any "first"
claim.

**Planned runs (not launched; the user asked to wait until the current
batch finishes).** All use the CSL recipe (`csl_stage1_weight`, 20 ep,
batch 32, lr 3e-4, dev selection) with hints = top-3 distinct TF-IDF
sentences retrieved with a draft (`draft_retrieval.py`, written 10-25, not
yet run). Dev/test drafts come from the released CSL model (test BLEU
25.61).
- `CSL_twopass_syn`: train drafts are references corrupted by deletion /
  substitution, with the noise set so the draft BLEU ≈ 25.6.
- `CSL_twopass_xfit`: train drafts come from 4 cross-fitted translators
  (folds grouped by sentence ID, each trained on 3/4 of train).
- `CSL_twopass_selffit` (C3 ablation): train drafts come from the full
  model, which memorises them.
- H2S versions after CSL.

**Related work** (positioning): Wray et al. CVPR 2021 (semantic similarity
instead of one-to-one relevance in video retrieval); Cai et al. ACL 2021
(cross-lingual retriever trained jointly with NMT); REPLUG NAACL 2024 /
Izacard & Grave ICLR 2021 (retriever supervised by the generator); Cheng et
al. EMNLP 2022 (diverse translation memories); HyDE / GAR (generate then
retrieve); ReMoDiffuse ICCV 2023 (retrieval for motion generation). SignRAG
trains its retriever for instance pairs and handles harmful retrieval
downstream (RUG-RFT), not in the retriever.

### 11.3 Stage 2 run plan: graded-relevance retriever (CSL first; not started, awaiting go-ahead)

**Why CSL first.** It is the supervisor's priority, and the stage-1
baselines are on CSL. The cached features (`feats/E_csl`) and the current
retriever (`step2/S2dn_E_csl`) already exist. H2S follows: its retrieval
is harder (pool recall 34.8% vs 73.0%, ORACLE 7.8).

**Training (train split only; the GT is used only to build targets).**
- Gallery: the unique training sentences (CSL about 6.6k).
- Targets: for clip i with reference yᵢ, rᵢⱼ = TF-IDF(yᵢ, sⱼ) (CJK char
  1–2-grams). Mask yᵢ itself and identical texts, because a test sentence
  is almost never in the gallery (CSL dev/test: 0.2% in train). Then
  qᵢ = softmax(rᵢ / T).
- Model: the existing dual encoder (`signrag_contrastive.py`). Scores
  over the whole gallery come from cached text embeddings, and
  pᵢ = softmax(f(poseᵢ)·g(s) / τ).
- Loss: KL(qᵢ ‖ pᵢ) + λ·InfoNCE(exact pair), λ ≈ 0.1–0.5.
- Dev (CSL): the real dev split queries the **train-sentence gallery**,
  as RAG does at test time (0.2% sentence overlap). S2dn instead selected
  on within-dev retrieval, where every query's own sentence is in the
  gallery (R@1 61%). How2Sign has no dev split and will need a pseudo-dev
  grouped by sentence.
- Selection: dev retrieval BLEU (top-1) / F1@3, not exact R@5.

**Runs (CSL).**

| Run | Init | Trainable | Question |
|---|---|---|---|
| G0 (done) | — | S2dn: phi head, exact InfoNCE | baseline: test retrieval BLEU 5.30 |
| G1 | `S2dn_E_csl/phi.pth` | phi head (frozen features, about 25 s/epoch) | do graded targets help, cheaply? |
| G2 | `phi_init` | phi head | does the warm start matter? |
| G3 | better of G1/G2 | + ST-GCN encoder (`--train_encoder`) | capacity (one full GPU) |

Later: text tower ablation (frozen mT5 vs trainable top layers vs bge-m3),
and BLEU or embedding-similarity targets instead of TF-IDF.

**Code (written 10-26, smoke-tested, not run).** A separate
`graded_retriever.py`, so that S2dn's code and runs stay as they are.
- It reuses the S2dn pieces (fine objective, frozen features, frozen mT5
  token cache plus text head).
- Candidates per batch: the batch sentences ∪ the top `--nbr_k` (8)
  TF-IDF neighbours of each, about 1,000 per batch of 128.
- Targets: softmax(TF-IDF / `--target_T` 0.1) with the own sentence
  masked. Loss: soft cross-entropy + `--exact_weight` (0.25) × S2dn's
  in-batch exact loss.
- `--init_phi` gives G1; without it phi starts from pose_proj (G2).
- Selection: dev → train-gallery retrieval BLEU. Test is scored once.
- The checkpoint is saved in the S2dn format, so
  `signrag_retrieval.py retrieve --phi` builds hints.
- G3 (trainable encoder) is not implemented yet.
- Smoke test (3 steps): epoch 0 = S2dn reproduces test retrieval BLEU
  5.30 / F1@1 20.74 / F1@3 20.10, matching S3dn exactly.

**Targets to beat (CSL test retrieval BLEU).** S3dn 5.30;
translate-then-TF-IDF 10.71; ORACLE 18.02.

**Then.**
- RA-SFT with the new retriever's hints, train hints cross-fitted
  (K folds), compared with `CSL_empty` 25.18 and `CSL_oracle_d` 27.63.
- The same on H2S.

## 12. References

- Rao et al. *SignRAG: Unified Retrieval-Augmented Gloss-Free Sign Language
  Translation.* arXiv 2610.11371, 2026. §3.1–3.2, Eq. 5–8, 20–23,
  Tables 2, 5, 6, 12, 13; App. B.2, C.2, C.3, C.7, E.5.
- Lai, Zhao, Zhong, Su, Chen. *Selective Contrastive Learning for Gloss Free
  Sign Language Translation (SCL-SLT).* ACL 2026; arXiv 2604.22374.
- Li et al. *Uni-Sign: Toward Unified Sign Language Understanding at Scale.*
  ICLR 2025.
- Jiao et al. *CoSign: Exploring Co-occurrence Signals in Skeleton-based
  Continuous Sign Language Recognition.* ICCV 2023.
- Zhou et al. *Gloss-free Sign Language Translation: Improving from Visual-
  Language Pretraining (GFSLT-VLP).* ICCV 2023.
- Yu et al. *CoCa: Contrastive Captioners are Image-Text Foundation Models.*
  2022.
- Zhai et al. *LiT: Zero-Shot Transfer with Locked-image text Tuning.*
  CVPR 2022.
- Khattab & Zaharia. *ColBERT.* SIGIR 2020.
- Qu et al. *RocketQA.* NAACL 2021.
- Cheng et al. *CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual
  Contrastive Learning.* CVPR 2023.
- Reimers & Gurevych. *Sentence-BERT.* EMNLP 2019 (paraphrase-MiniLM-L6-v2).
