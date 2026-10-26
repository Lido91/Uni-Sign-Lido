# SignRAG Reproduction — Progress Log

Last updated: 2026-10-26 04:20 UTC

This file holds status, results and a step-by-step log for review. Method details, the run matrix and
reasoning are in `doc/signrag_style_retriever_plan.md`. All outputs are under
`$OUT = /data/hwu/Uni_sign_repo/out/signrag_retrieval/`.

## Status Board

| Line | Goal | State | Latest result | Next |
|---|---|---|---|---|
| **D. CSL-Daily** (top priority, requested by supervisor) | Reproduce SignRAG RA-SFT on CSL-Daily (paper: SFT 29.72 → RA-SFT 31.22 BLEU-4) | All CSL runs done (10-25 01:54) | Real retrieval: no gain (`real_d` 25.00 vs `empty` 25.18 test B4). **Sentence TF-IDF ORACLE `oracle_d` 27.63 (+2.45)**. GT-gloss ORACLE 31.04 | See Results / E; next is closing the retrieval gap |
| B. RA-SFT on H2S | Does sentence retrieval help mT5 SLT? | E1 done. E2 real stopped at 14/20 epochs (NaN loss). E2 empty crashed at epoch 2 (inf loss); the user decided not to rerun it | E1: real 15.04 / empty 15.13 (no retrieval gain). E2 real best 14.33 (epoch 11) | Paused |
| **E. Hint training (sentence TF-IDF oracle, word-level retrieval), H2S + CSL** | Train the translator with hints from TF-IDF on the GT (ORACLE), word-level versions, and real word retrieval | All done (H2S word runs 10-26 ~02:00) | **H2S_oracle_d 17.04 vs H2S_empty 15.13 (+1.9)**; CSL_oracle_d +2.45. Word hints: H2S_words 15.01 and H2S_oracle_words 15.10 (≈ 0); H2S_gtwords 16.15 (ORACLE, +1.0); CSL predgloss / oracle words ≈ 0 | Sentence hints, not word hints, carry the gain; closing the retrieval gap is line F |
| **F. Strong utility-aligned retriever** (agreed 10-25) | Retrieve sentences/words close to the *answer* (graded relevance, oracle distillation) with a high-capacity sign tower; later text→pose for SLG | Design in plan doc §11.1. Translator-rerank check done (CSL + H2S). Stage 1 running (Results / G). Stage 2 run plan in plan doc §11.3, not started | Baselines to beat (retrieval BLEU, test): S3dn 5.30 / 1.86, translate-then-TF-IDF 10.71 / 3.20, ORACLE 18.02 / 7.8 (CSL / H2S) | Contribution framing agreed (plan §11.2: C1 diagnosis, C2 draft-space retrieval, C3 matched hint quality, C4 learned reranker). **Stage 1: `CSL_twopass_syn` done (test 25.50 at best dev, dev 24.79 below `empty` 25.57: mixed); `selffit` running; folds running; fold2 rerun after a NaN abort, then `xfit` (~10-26 14:30)**. Stage 2 (graded-relevance retriever) awaits the go-ahead |
| C. Word-level dictionary | Mine word → segment pairs from H2S/OpenASL with WLASL; use the words as hints | Prototype done | Spotting top-1 33.6% vs control 1.5%; word-order Spearman 0.86 | Oracle-hint ceiling; full-scale mining |
| A. Sentence retriever | Strong sign ↔ text retriever | Done (S3dn) | H2S R@1 V2T 35.4 / T2V 38.8; OpenASL 51.7 / 51.1 | CiCo-style upgrades (parked) |

## Running Jobs

| GPU | Job | Started (UTC) | ETA (UTC) | Output |
|---|---|---|---|---|
| 3 | `fold0` translator (3/4 of train, CSL recipe), 14/20 epochs at 03:40, ~12.5 min/epoch; then translates its held-out quarter | 10-26 00:45 | ~05:10 | `$OUT/rag_inputs/twopass_csl/fold0{,_heldout}/` |
| 0 | `fold1` translator, 12/20 at 03:36 | 10-26 01:07 | ~05:20 | `…/fold1{,_heldout}/` |
| 1 | `fold3` translator, 8/20 at 03:40 | 10-26 01:59 | ~06:30 | `…/fold3{,_heldout}/` |
| 2 | `CSL_twopass_selffit` (train hints from self-fit drafts, retrieval BLEU 19.01 vs test 10.66), 3/20 at 03:40 | 10-26 02:41 | ~07:30 | `$OUT/ra_sft/CSL_twopass_selffit/` |
| chain | `bash script/twopass_csl.sh jobs fold2 CSL_twopass_xfit` (pid 2099136): `fold2` rerun with `--skip_nonfinite_loss` on the first idle GPU (~05:10 → ~09:45), then `CSL_twopass_xfit` once all 4 held-out translations exist (~09:45 → ~14:30) | 10-26 03:50 | xfit ~14:30 | `$OUT/rag_inputs/twopass_csl/queue_jobs.out` |
| — | The first queue's main loop (pid 1924836) was stopped at 03:50 so it cannot race the chain for `xfit`; its four running jobs continue (ppid 1) and log to `queue.out` | | | |
## Planned, Not Started

| Item | Dataset | Status | Details |
|---|---|---|---|
| Stage 2: graded-relevance retriever G1 (warm start from S2dn) / G2 (from scratch) / G3 (+ encoder) | CSL first | Awaiting go-ahead; G1/G2 code written and smoke-tested (`graded_retriever.py`), G3 not yet implemented. G1/G2 fit in GPU 2's spare memory; G3 needs an idle GPU (one expected from ~10-26 06:00) | plan doc §11.3 |
| RA-SFT with the stage-2 retriever (cross-fitted train hints) | CSL, then H2S | After stage 2 | plan doc §11.3 |
| Stage 1 on H2S (`H2S_twopass_*`) | H2S | After the CSL stage-1 results | plan doc §11.2 |
| Dose–response (mix oracle and real hints; hint quality → BLEU gain) | CSL | Not started | plan doc §11 item 6 |
| SLG oracle (GT-nearest pose exemplars into the generator) | — | Not started | plan doc §11.1 item 7 |

## Results

### D. CSL-Daily

- **Reference (SignRAG Table 2a).** SFT 29.72 → RA-SFT 31.22 BLEU-4. This
  is the paper's only retrieval ablation.
- **Data facts.**
  - Train has 18,401 videos but only 6,578 unique sentences, each signed by
    several signers.
  - Only 0.2% of dev/test sentences appear verbatim in train (How2Sign:
    3.0%).
  - Gloss annotations use a 2,000-gloss vocabulary.
- **Baseline (released `csl_daily_pose_only_slt.pth`, our eval, char-level BLEU).**

| Split | B1 | B2 | B3 | B4 | R-L |
|---|---|---|---|---|---|
| dev | 53.18 | 40.47 | 31.55 | 25.14 | 54.28 |
| test | 53.79 | 40.91 | 32.00 | 25.61 | 54.94 |
| official prediction files: dev / test B4 | | | | 25.27 / 25.70 | 54.30 / 55.00 |

- **Code changes for Chinese (2026-10-24).**
  - `signrag_common.py` gains `text_tokens` (characters for Chinese),
    `bleu_text` (char-level, as `fine_tuning.evaluate` does) and a
    CSL-aware `video_id`.
  - `signrag_retrieval.py` uses them for the F1 tokens, retrieval BLEU and
    the TF-IDF oracle (char 1–2-grams). `CSL_Daily` was added to the
    `--dataset` choices of both scripts.
  - `fine_tuning_rag.py` fix: on datasets with a dev split it now logs
    `dev_*` and the real `test_*`. Before, the dev scores were written as
    `test_*`. H2S runs are unaffected.
  - English behaviour is unchanged.
- **Pipeline.** `script/signrag_csl_daily.sh`, with stages `retriever`,
  `rasft` and `summary`.
  - RA-SFT recipe: start from `csl_stage1_weight.pth`, 20 epochs,
    effective batch 32, lr 3e-4, fp32.
  - Selected on dev; test is reported at the best dev epoch.
- **Smoke tests passed.** Extraction, retrieval and evaluation work on
  Chinese, and train queries are filtered (em@1 = 0). The stage-1 weight
  loads strictly. The log has separate dev and test entries.
- **Retriever (S3dn_E_csl).**
  - Features from `csl_daily_pose_only_slt.pth`; fine objective, exact
    positives, batch 256.
  - Dev selection picked r5_mean 78.16 after 27 epochs.

| CSL-Daily | V2T R@1 / R@5 / R@10 | T2V R@1 / R@5 / R@10 |
|---|---|---|
| dev (paired, within split) | 60.45 / 78.64 / 83.94 | 59.60 / 77.67 / 82.18 |
| test (paired, within split) | 58.67 / 77.89 / 84.52 | 59.52 / 78.07 / 83.71 |

| Top-1 train sentence used as the translation (test) | BLEU-4 | ROUGE-L | F1@1 |
|---|---|---|---|
| random | 0.18 | 11.98 | 6.50 |
| S1 frame (pose_proj, untrained) | 2.13 | 15.84 | 11.90 |
| S3dn s2t (ours) | 5.30 | 23.19 | 20.74 |
| TF-IDF oracle (uses the GT text) | 18.02 | 42.03 | 43.09 |

  - The RA-SFT inputs are in `$OUT/rag_inputs/S3dn_E_csl/`: s2t top-3
    for train (18,401 queries, filtered), dev and test.
  - Train s2t top-1 BLEU-4 is 6.11, close to test (5.30), so the
    train-time hints are not easier than the test-time hints.
- **RA-SFT (finished 10-24 06:16, 20 epochs each, no non-finite losses).**
  Both start from `csl_stage1_weight.pth` with effective batch 32, lr 3e-4
  and fp32. The table shows the epoch with the best dev B4.

| Run | best epoch | dev B4 | test B1 | B2 | B3 | B4 | R-L |
|---|---|---|---|---|---|---|---|
| empty (SFT baseline) | 19 | 25.57 | 53.31 | 40.40 | 31.55 | **25.18** | 54.97 |
| real (top-3 S3dn sentences, dropout 0.5) | 18 | 25.30 | 52.74 | 39.84 | 30.99 | **24.62** | 54.54 |
| released `csl_daily_pose_only_slt.pth` (our eval) | — | 25.14 | 53.79 | 40.91 | 32.00 | 25.61 | 54.94 |
| SignRAG Table 2a: SFT → RA-SFT | — | — | — | — | — | 29.72 → 31.22 | — |

- **Diagnostic: oracle hints at test time (10-24 14:30).**

| `CSL_real` best checkpoint, test-time hints | dev B4 | test B4 |
|---|---|---|
| S3dn top-3 (as trained) | 25.30 | 24.62 |
| TF-IDF oracle top-3 (most similar train texts to the GT) | 25.24 | 24.77 |

  The model ignores the hints. Even near-answer sentences do not move
  BLEU, so retrieval quality is not the only bottleneck.
- **Bug found (10-24 14:40): the "top-3" was one sentence repeated three
  times.**
  - CSL-Daily repeats every sentence over several signers. The s2t score
    depends only on the text, so all clips of the best sentence tie and
    fill the top-3.
  - 1,176 of 1,176 test queries were affected; for the TF-IDF oracle,
    1,173 of 1,176.
  - The first pair's real run therefore saw one distinct sentence, not
    three. H2S is barely affected, since only 5% of its train texts repeat.
  - Fix: `signrag_retrieval.py --distinct_text` fetches 64 candidates and
    keeps the first K distinct normalised texts. The default is off, so
    H2S outputs are unchanged.
  - New inputs: `$OUT/rag_inputs/S3dn_E_csl_distinct/` (s2t, TF-IDF
    oracle, random; train filtered).
- **New `--rag_mode gloss`.** `datasets_rag.py` and `fine_tuning_rag.py`
  can now use the ground-truth gloss sequence as the hint (`Glosses: …`,
  dropout 0.5 in training), a ceiling for word-level hints.
- **Overnight queue (10-24 14:47, `script/csl_daily_overnight.sh`).** All
  runs use one GPU each with effective batch 32 (8 × accum 4), otherwise
  the same recipe.

| Run | Hint (train / test) | Question |
|---|---|---|
| `real_d` | distinct S3dn top-3 | the corrected main experiment |
| ~~`real_d_s43`~~ | cancelled 10-24 (no seed repeats) | — |
| `oracle_d` | distinct TF-IDF oracle top-3 (train filtered) | can mT5 learn to use good sentence hints? (ceiling) |
| `gloss` | GT gloss | ceiling for word-level hints (line C) |
| `random_d` | 3 random train sentences | control |
| ~~`empty_s43`~~ | stopped 10-24 after 3.5 min (no seed repeats) | — |
| diag evals | `CSL_real` with no hints / distinct oracle hints; `oracle_d` with S3dn hints; `real_d` with oracle / no hints | how much each model relies on its hints |
| H2S spotting | all H2S train sentences, 12/16-frame windows | full pseudo-dictionary (line C) |

- **Comparison with SignRAG (from the paper, arXiv 2610.11371).**

| | SignRAG | Ours |
|---|---|---|
| Input | 2D keypoints (MMPose) | 2D keypoints (RTMPose, Uni-Sign) |
| Sign encoder | CoSign + 3-layer Transformer | Uni-Sign 4-part ST-GCN |
| Translator | **Qwen3-8B + LoRA** (decoder-only LLM) | **mT5-base, full fine-tune** |
| Pretraining | Stage I: contrastive + caption, CSL-News, 200 epochs. Stage II: encoder + projector + LoRA-LLM, CSL-News + YouTube-ASL, 50 epochs | Uni-Sign stage 1 (pose-only, CSL-News) |
| SFT baseline B4 (test) | 29.72 (23.46 without pretraining) | 25.18 |
| Retrieval | sign→sign, their Stage-I encoder, Eq. 5–8, gallery = train clips | sign→text (S3dn: trained head, frozen mT5 text encoder) |
| Filters | self + identical text | self + identical text (same) |
| Top-K / dropout | 3 / 0.5 | 3 / 0.5 (same) |
| Hint position | [instruction][sign][retrieved] | [retrieved][prompt][sign] |
| SFT recipe | 10 epochs, lr 1e-4, batch 4 × 4 GPUs, bf16, LoRA only | 20 epochs, lr 3e-4, batch 32, fp32, full model (official Uni-Sign recipe) |
| Selection | validation B4 | dev B4 (same) |
| RL (RUG-RFT) | +1.7 (31.22 → 32.91) | not done |
| Code / weights | not found in the paper | — |

  So our runs transplant SignRAG's RA-SFT recipe (filters, K, dropout, dev
  selection, target-train gallery) onto Uni-Sign. They are not a
  re-implementation of SignRAG. The largest gaps are the 8B LLM translator
  and their two-stage pretraining.
- **Reading the RA-SFT result.**
  - The SFT baseline is reproduced: empty gives test 25.18, against 25.61
    for the released model.
  - Retrieval does not help. Real is −0.27 dev and −0.56 test B4. From
    epoch 4 on, real's dev curve stays 0.3–1.0 below empty's.
  - This matches the H2S E1 result (real 15.04 vs empty 15.13).
  - SignRAG's SFT baseline is 4.5 B4 stronger (LLM translator, its own
    Stage II pretraining), so their +1.5 comes from a different operating
    point.

### B. RA-SFT on How2Sign (test, B1/B2/B3/B4, ROUGE-L)

| Run | B1 | B2 | B3 | B4 | R-L |
|---|---|---|---|---|---|
| Official E1 checkpoint (our eval) | 39.44 | 26.50 | 19.22 | 14.48 | 34.62 |
| E1 zero-shot real / empty / random | 35.97 / 39.70 / 32.91 | 23.75 / 26.59 / 21.59 | 17.08 / 19.26 / 15.41 | 12.78 / 14.48 / 11.44 | 32.99 / 34.45 / 32.16 |
| E1 RA-SFT real | 40.29 | 27.21 | 19.89 | 15.04 | 35.54 |
| E1 RA-SFT empty | 40.34 | 27.23 | 19.94 | 15.13 | 35.62 |
| E1 RA-SFT random | 40.05 | 27.11 | 19.85 | 15.02 | 35.76 |
| E2 zero-shot real / empty | 26.33 / 26.45 | 13.38 / 13.59 | 8.01 / 8.09 | 5.06 / 5.13 | 20.37 / 20.73 |
| E2 RA-SFT real (best, epoch 11; stopped at 14 of 20) | 39.10 | 26.24 | 19.05 | 14.33 | 35.11 |
| E2 RA-SFT empty (crashed at epoch 2; not rerun) | — | — | — | 12.37 (ep 1) | — |

- **E2 instability.**
  - E2 real aborted at epoch 14, step 369 with a NaN loss.
  - E2 empty aborted at epoch 2, step 971 with an inf loss.
  - In both cases the inputs of the saved batch are finite
    (`nonfinite_batch_rank0.pt` in each run dir), so this is a numerical
    blow-up in the model, not bad data.
  - Earlier, the epoch-1 train loss of E2 real was logged as 2.9e26.
  - The runs had no `--skip_nonfinite_loss`, so the guard in
    `fine_tuning.handle_nonfinite_loss` stops at the first bad step.

### E. Hint Training: Sentence TF-IDF Oracle and Word-Level Retrieval (from 2026-10-24)

**Results so far (10-25 19:20).** CSL is selected on the best dev epoch;
H2S has no dev split and follows the existing pipeline. ORACLE means the
hint is built from the sample's ground truth.

| CSL-Daily run | hint (train = test source) | type | dev B4 | test B4 | test R-L |
|---|---|---|---|---|---|
| `empty` | none | baseline | 25.57 | 25.18 | 54.97 |
| `random_d` | 3 random train sentences | control | 25.34 | 25.19 | 55.21 |
| `real` | S3dn top-3 (one sentence repeated) | real retrieval | 25.30 | 24.62 | 54.54 |
| `real_d` | S3dn top-3 distinct | real retrieval | 24.79 | 25.00 | 54.60 |
| `predgloss` | CTC-spotted glosses (test WER 31) | real word retrieval | 24.81 (ep 1) | 24.75 (last epoch 25.49) | 54.73 |
| `oracle_words` | 8 words from the TF-IDF top-3 | ORACLE | 25.04 | 25.85 | 55.69 |
| **`oracle_d`** | TF-IDF top-3 distinct sentences | **ORACLE** | **27.57** | **27.63** | 57.67 |
| `gloss` | GT gloss | ORACLE | 31.16 (ep 1) | 31.04 (≈ 28.5 at later epochs) | 63.86 |

| How2Sign run (E1 recipe) | hint | type | test B4 (best) | test B4, epochs |
|---|---|---|---|---|
| `H2S_empty` | none | baseline | 15.13 | 14.7–15.1 |
| `H2S_real` / `H2S_random` | S3dn top-3 / random | real / control | 15.04 / 15.02 | — |
| **`H2S_oracle_d`** | TF-IDF top-3 distinct sentences | **ORACLE** | **17.04** | 16.8–17.0 from epoch 3 on |
| `H2S_oracle_words` | 8 words from the TF-IDF top-3 | ORACLE | 15.10 (ep 9) | 14.7–15.1 |
| `H2S_words` | WLASL-spotted words (precision ≈ 17%) | real word retrieval | 15.01 (ep 9) | 14.7–15.0 |
| `H2S_gtwords` | GT words with a WLASL gloss | ORACLE | **16.15** (ep 9) | 15.1 → 16.2, still rising |

- **Translator-as-retriever (10-25 19:30, CPU, no GT).** The baseline
  model's own test translation is used as the TF-IDF query over the train
  sentences, and the top-1 is scored as a translation.

| Retrieval BLEU-4 (top-1 as translation, test) | CSL | H2S |
|---|---|---|
| S3dn (pose → text) | 5.30 | 1.86 |
| **translate first, then TF-IDF on the translation** | **10.71** | **3.20** |
| TF-IDF on the GT (ORACLE) | 18.02 | 7.8 |

  No GT is needed, yet this already doubles S3dn's retrieval quality and
  covers about half the way to the oracle. It is the baseline any new
  retriever must beat. Train-time hints of this kind would need predictions
  on train clips from a model that did not train on them.
- **`CSL_real_d` diagnostics (rerun 10-25 19:19).**
  - With TF-IDF oracle hints: dev 24.75 / test 25.06.
  - With no hint: 24.70 / 24.77.
  - With its own hints: 24.79 / 25.00.
  - Like `CSL_real`, a model trained with weak hints ignores them.
- **Reading.**
  - With *good* sentence hints in training, mT5 does use them: +2.45 (CSL)
    and +1.9 (H2S).
    - Diagnostic: `CSL_oracle_d` evaluated with real S3dn hints drops to
      24.60 test (dev 24.88), below `empty`. The model relies on its hints,
      so hint quality decides the result.
  - Real retrieval (S3dn) gives ≈ 0 on both datasets. The gap to the oracle
    is retrieval quality: top-1 as translation is BLEU 5.3 (CSL) and 1.9
    (H2S), against 18.0 and 7.8 for the oracle.
  - Word hints: perfect glosses help a lot (+5.9 at the peak), but CTC
    glosses at WER 31% / precision 79% give ≈ 0. The 8 TF-IDF oracle words
    (precision 31%) also give ≈ 0. Word hints need far higher precision, or
    a model trained to tolerate noise.


**Training input.** Every run trains the translator (RA-SFT); none only swaps
hints at test time.
- The mT5 input is `[hint][Translate … prompt][pose]`.
- The hint is dropped with probability 0.5 in training and always given at
  test time, from the same source.

**Sentence-level TF-IDF oracle (priority, user-confirmed 10-24).**
- The hint is the top-3 *distinct* training sentences most similar to the
  sample's **ground-truth translation** (TF-IDF), written as
  `Similar sentences: s1 | s2 | s3`.
- The gallery is the training split only. Training queries exclude
  themselves and every clip with the identical sentence. Test queries are
  not filtered; verbatim train duplicates are 0.2% of CSL test and 3.3% of
  H2S test.
- **This is an ORACLE upper bound**: it needs the reference translation.

| Run | Dataset / recipe | Inputs | Status | Log |
|---|---|---|---|---|
| `CSL_oracle_d` | CSL-Daily, `csl_stage1_weight`, 20 ep, batch 32, lr 3e-4, dev selection | `$OUT/rag_inputs/S3dn_E_csl_distinct/retrieval_{train,dev,test}_tfidf_oracle.json` | **training on GPU 0 since 10-24 14:57** (lane D) | `$OUT/ra_sft/CSL_oracle_d/log.txt` |
| `H2S_oracle_d` | How2Sign, E1 checkpoint, 10 ep, batch 64, lr 1e-4 (as `H2S_empty` 15.13) | `$OUT/rag_inputs/tfidf_E1_h2s_distinct/retrieval_{train,test}_tfidf_oracle.json` | **first in the automatic queue**; starts on the first free GPU (~10-25 01:00) | `$OUT/ra_sft/H2S_oracle_d/log.txt`, `$OUT/word_retrieval/queue.out` |

- **H2S TF-IDF inputs (new, 10-24 15:05).** `signrag_retrieval.py retrieve
  --score pool --queries train test --distinct_text` covers all 31,086
  train and 2,349 test clips.
  - Every row has distinct texts. A few short queries have k < 3.
  - Top-1 as the translation, BLEU-4 / ROUGE-L: test 7.81 / 29.71, train
    7.97 / 28.44. The train hints are as good as the test hints.

**Word-level TF-IDF oracle (separate control; does not replace the
sentence runs).**
- Same TF-IDF top-3 neighbours as above. The words of the neighbours are
  ranked by Σ_k score_k · idf(w), and the top 8 form the hint
  (`tfidf_word_hints.py`).
- H2S units: content words of the neighbour sentences (`Words: …`).
- CSL units: the neighbours' *training* gloss labels (`Glosses: …`). These
  are labels of the retrieved training clips, never the query's.
- ORACLE, since the query is the GT translation.

| | avg words | precision vs GT | recall vs GT | words of all 3 neighbours (P / R) |
|---|---|---|---|---|
| CSL test (vs GT gloss set) | 7.96 | 30.9 | 37.1 | 24.5 / 52.3 |
| CSL train (cross-checked) | 7.97 | 30.4 | 38.3 | 24.4 / 53.7 |
| H2S test (vs GT content words) | 6.60 | 34.9 | 41.2 | 32.4 / 46.9 |
| H2S train | 6.69 | 33.8 | 39.8 | 31.1 / 45.7 |

- CSL examples (test, query → hint):
  - 他的言语里饱含着恳求、温柔、亲切。 → `Glosses: 温柔 温暖 亲切 声音 环境 拒绝 男 怎么`
  - 签证不是随便做。 → `Glosses: 证件 签证 领事 办理 评价 外国 签名 随便`
- H2S examples:
  - "The dish now we're going to be preparing is called paneer masala." →
    `Words: paneer masala dish preparing called we're going thicken`
  - "…look to see if I could find it even cheaper…" →
    `Words: recreational continue seventy players hitting club that's they're`
- Outputs: `$OUT/word_retrieval/{csl,h2s}/tfidf_oracle_words/`
  (`hints_*.json`, `metrics.json`).
- Runs: `CSL_oracle_words` and `H2S_oracle_words` (queued).

**Word-level retrieval without GT (the real method), one per sign
language.** In both, the test-time hint uses only the pose: no translation
or gloss of the sample.
- **How2Sign (ASL): WLASL lexicon.** `word_retrieval_h2s.py`.
  - `spot` runs the released WLASL ISLR model (2,000 ASL glosses) over
    windows of the normalised clip and keeps the top-10 beams per window.
  - `hints` keeps the glosses that are top-1 of some window with evidence
    ≥ θ, at most 8, in temporal order (`Words: …`).
  - θ is calibrated on training clips only (max F1 against their
    subtitles).
  - The same dump gives the train pseudo-dictionary (subtitle-constrained,
    train only), plus a separate ORACLE `H2S_gtwords`: GT content words
    that have a WLASL gloss.
  - **Validation (10-24 17:50)**: 1,000 random train clips for calibration
    and 500 random test clips. θ is calibrated on train only. Test results:

| windows | words / clip | P vs WLASL-matchable gold | R | F1 | P@1 | hit@5 | frequency prior P (same #words) |
|---|---|---|---|---|---|---|---|
| 16, stride 4 | 4.66 | **16.8** | 19.1 | 16.5 | 27.4 | 52.0 | 14.1 |
| 12+16, stride 4 | 5.37 | 14.8 | 19.6 | 15.6 | 26.1 | 51.4 | 14.1 |
| 12+16+24, stride 4 | 4.82 | 15.9 | 18.5 | 15.9 | 26.5 | 51.4 | 14.1 |
| 16, stride 8 (chosen for the full run) | 3.82 | 18.2 | 16.2 | 15.9 | 25.3 | 49.8 | 15.0 |

  - **Reading.** Unconstrained WLASL retrieval is weak on continuous H2S
    signing.
    - Precision is about 17%, against 14–15% for the frequency prior and
      36.5% for the content words of the translator's own output (§10b
      bar).
    - The subtitle-constrained prototype was precise (top-1 33.6% vs 1.5%
      control) only because it chose among about 4 subtitle words.
    - WLASL covers 55% of the reference content words.
    - So `H2S_words` is expected to add mostly noise, but it is run as
      planned.
  - **Cost.** About 10 s per clip for 3 window sizes on a shared GPU, so the
    full run uses only 16-frame windows at stride 8 (F1 15.9 vs 16.5).
  - **Full run**: rerun finished 10-25 19:55 (the 10-24 attempt ran out of
    GPU memory) → `$OUT/word_retrieval/h2s/spot_full/`, then
    `wlasl_hints/` (θ = −1.00 on 3,000 train clips).
    - Test: 4.05 words per clip, precision 15.9% (frequency prior 14.9%),
      recall 15.6%, P@1 22.7.
    - Train: 4.10 words, precision 15.9%. Train and test hints are equally
      (weakly) informative.
    - Train pseudo-dictionary (subtitle-constrained, train only): 1,383
      glosses with entries, 33,109 entries, median 7 per gloss
      (`dictionary_train.json`).
- **CSL-Daily (CSL): no isolated CSL lexicon or CSL ISLR model is
  available locally.** The WLASL ASL lexicon is not used for CSL.
  - The lexicon is the 2,000 glosses of the CSL-Daily *train* split.
  - A CTC gloss spotter (`word_retrieval_csl.py`: LayerNorm-Linear, conv,
    2-layer BiGRU) is trained on the cached frame features `feats/E_csl`
    with the train gloss sequences. CTC spikes localise each gloss to frames
    without timing labels.
  - Train hints are cross-fitted (5 folds grouped by sentence ID); dev/test
    hints come from the full-train model. Hint format: `Glosses: …`.
  - **Result (10-24 17:50).**

| CTC spotter (25 epochs, dev-selected) | WER | gloss-set P / R / F1 | pred / gold length |
|---|---|---|---|
| test (full-train model) | **31.36** | **79.2 / 74.7 / 76.5** | 7.1 / 7.7 |
| dev | 32.39 | 78.8 / 74.3 / 76.1 | 7.0 / 7.6 |
| train, cross-fitted (used for train hints) | 32.81 | 78.2 / 73.8 / 75.5 | 6.8 / 7.3 |
| train, self-fit (not used) | 3.57 | 99.6 / 98.0 / 98.7 | 7.1 / 7.3 |
| frequency prior (7 most frequent glosses) | 103.25 | 13.7 / 13.7 / 13.4 | 7.0 / 7.7 |

  - Cross-fitting matters: self-fit train hints would be near-perfect
    (WER 3.6), while test hints have WER 31. The cross-fitted train WER
    (32.8) matches test, so train and test hints have the same quality.
  - These non-oracle word hints are far more precise than the word-level
    TF-IDF oracle (79% vs 31%).
  - Examples (test; GT gloss → predicted, with CTC spike frames):
    - 老师处罚学生打扫: 老师 处罚 同学 们 打扫 → identical, frames
      0/14/25/34/48.
    - 他是一个聪明、活泼、可爱的农村孩子: 他 是 聪明 活泼 可爱 农村 孩子
      → 他 是 聪明 **脏** 可爱 **了解** 孩子.
    - 签证不是随便做: 签证 随便 做 不是 → 签证 随便 **工作** 不是.
  - Output: `$OUT/word_retrieval/csl/ctc_E_csl/`. `CSL_predgloss` is
    ready (second in the queue).
  - Caveat: the `E_csl` encoder was fine-tuned on CSL-Daily train
    translations, so cross-fitted train hints may be slightly optimistic.
    `metrics.json` compares cross-fit train WER with test WER to measure
    this.
- **Evaluation limits.** Neither dataset has word-level timing.
  - H2S retrieval is scored at clip level against the reference's content
    words (lemma match).
  - CSL is scored with WER and gloss-set P/R/F1 against the sentence gloss
    sequence. Localisation itself is not evaluable.
  - The GT-gloss / GT-word runs (`CSL_gloss`, `H2S_gtwords`) are ORACLE
    ceilings, not word retrieval.

**Diagnostics of the first CSL pair (eval only).**

| `CSL_real` best checkpoint, test-time hint | dev B4 | test B4 |
|---|---|---|
| S3dn top-3 (as trained) | 25.30 | 24.62 |
| none | 25.13 | 24.91 |
| TF-IDF oracle top-3 (duplicate rows) | 25.24 | 24.77 |
| TF-IDF oracle top-3 distinct | 25.26 | 24.56 |

The model trained with weak hints ignores them. Whether good hints in
*training* teach mT5 to use them is what `*_oracle_d` measures.

### F. Strong Retriever: Step 1, Translator Rerank (test only)

Candidate pool per test clip: TF-IDF top-50 on the baseline translation ∪
S3dn top-50 distinct (about 94 candidates). The pool is reranked by the SLT
model's mean token log p(candidate | pose). Top-1 is scored as a
translation. `rerank_retrieval.py`, output `$OUT/rerank/{csl,h2s}/`.

| CSL-Daily test | BLEU-4 | ROUGE-L | F1@1 | F1@3 |
|---|---|---|---|---|
| S3dn top-1 | 5.30 | 23.19 | 20.74 | 20.09 |
| translator rerank of the pool | 7.57 | 28.19 | 26.40 | 24.61 |
| TF-IDF on the translation, top-1 | **10.71** | 32.37 | 31.35 | 30.98 |
| ORACLE: best candidate in the pool (TF-IDF to the GT) | 16.84 | 41.22 | 42.28 | 42.28 |
| ORACLE: TF-IDF on the GT over the full gallery | 18.02 | — | — | — |

- **The pool is good, the ranking is not.**
  - The pool contains the oracle's top-1 for 73.0% of clips (ORACLE
    analysis), and its best member reaches 16.84 of the 18.02 ceiling.
  - Ranking by the translator's likelihood (7.57) is worse than plain
    TF-IDF on the translation (10.71): the likelihood of a *different*
    sentence is a poor similarity signal.
- **Implication for line F.** Recall@100 is solved by a cheap union. The
  work is a ranker trained for graded relevance against the GT, which is
  the agreed graded-relevance objective (plan §11.1), learned on train
  clips and evaluated on dev/test.
| How2Sign test (pool ≈ 98) | BLEU-4 | ROUGE-L | F1@1 | F1@3 |
|---|---|---|---|---|
| S3dn top-1 | 1.86 | 10.93 | 15.49 | 17.02 |
| TF-IDF on the translation, top-1 | 3.20 | 16.44 | 21.68 | 24.59 |
| translator rerank of the pool | 3.78 | 16.19 | 21.16 | 22.72 |
| ORACLE: best candidate in the pool | 7.03 | 25.79 | 33.98 | 33.98 |
| ORACLE: TF-IDF on the GT over the full gallery | 7.8 | — | — | — |

- H2S (finished 10-25 20:16).
  - Again the pool's ORACLE best (7.03) is near the ceiling (7.8).
  - The translator rerank is about on par with TF-IDF on the translation
    (BLEU 3.78 vs 3.20, F1@1 21.2 vs 21.7).
  - The pool holds the oracle top-1 for only 34.8% of clips (CSL: 73.0%),
    so on H2S recall is also harder.

- **Why S3dn ranks poorly (checked 10-25 21:00, corrected 10-26 04:20).**
  - Only 0.2% of CSL dev and test sentences (dev 1077, test 1176 clips)
    occur in train.
  - S2dn was trained on exact pairs and selected by *within-dev*
    retrieval. For CSL, that is the real dev split, with the dev sentences
    themselves as the gallery. Each query's own sentence is therefore a
    candidate, which gives dev R@1 61%.
  - RAG retrieves from the train sentences, where the own sentence is
    almost never present. The old selection metric measured a different
    task.
  - (Correction: on 10-25 this was attributed to a random 5% pseudo-dev
    taken from train. That split is used only without a dev split, i.e.
    for How2Sign.)
  - Stage 2 uses graded targets and selects on dev → train-gallery
    retrieval BLEU.

### G. Stage 1: Translate, Retrieve, Refine (CSL, from 10-25 21:00)

Hints are the top-3 distinct training sentences retrieved by TF-IDF with a
*draft translation* as the query (`draft_retrieval.py`). Dev/test drafts
come from the released CSL model (`csl_daily/baseline_eval`). Train
drafts must have test-like quality (C3). Output:
`$OUT/rag_inputs/twopass_csl/`.

| Hints | Draft B4 | Retrieval B4 (top-1) | F1@1 | F1@3 |
|---|---|---|---|---|
| dev, released-model drafts | 25.14 | 10.88 | 31.38 | 30.45 |
| **test, released-model drafts** | 25.61 | **10.66** | 31.28 | 30.89 |
| train synthetic, noise 0.3 | 33.62 | 14.84 | 38.36 | 36.22 |
| train synthetic, noise 0.4 | 22.55 | 12.76 | 35.24 | 33.53 |
| **train synthetic, noise 0.5 (used)** | 14.24 | **10.82** | 31.93 | 30.65 |
| train synthetic, noise 0.6 | 7.98 | 8.52 | 27.47 | 26.87 |

- **Calibration.** The translator sees only the hints, not the draft, so
  the noise is chosen to match *hint* quality: 0.5 gives retrieval BLEU
  10.82 vs 10.66 on test. At equal draft BLEU (noise ≈ 0.38),
  random-noise drafts retrieve better hints than real translator drafts,
  whose errors are systematic.
- **Runs** (CSL recipe; compare with `CSL_empty` 25.18 and `CSL_oracle_d`
  27.63):
  - `CSL_twopass_syn`: synthetic train drafts, noise 0.5. Running, GPU 2.
  - `CSL_twopass_xfit`: train drafts from 4 fold translators (grouped by
    sentence ID, about 13.8k train / 4.6k held-out clips each, trained
    from `csl_stage1` with the CSL recipe). These are a bit weaker than the
    released model, so train hints may be slightly worse than test hints.
    Queued.
  - `CSL_twopass_selffit` (C3 ablation): the released model translates
    its own training clips, giving near-perfect train hints. Queued.
- **Self-fit drafts (`selffit_preds`, 10-25 22:00).** The released model
  translates its own training clips at **BLEU-4 99.75**: it has memorised
  train. Self-fit train hints are therefore oracle-like, while the test
  hints come from 25.6-BLEU drafts. This is the mismatch C3 tests. The same
  call re-scored dev at 25.18 with `fine_tuning_rag.py --rag_mode empty`
  (`fine_tuning.py` gave 25.14), so the two code paths agree.
- **`CSL_twopass_syn` result (finished 10-26 ~02:40).** It is selected at
  the best dev epoch, 18.

  | Run | dev B4 (best) | test B4 at that epoch | test, last 10 epochs |
  |---|---|---|---|
  | `CSL_empty` | 25.57 | 25.18 | 24.7–25.4 |
  | `CSL_real_d` (S3dn hints) | 24.79 | 25.00 | 23.7–25.2 |
  | **`CSL_twopass_syn`** | 24.79 | **25.50** | 24.6–25.7 |
  | `CSL_oracle_d` (ORACLE) | 27.57 | 27.63 | 26.6–27.6 |

  - The outcome is mixed. Test is +0.32 over `empty` and higher in most of
    the last epochs, but dev is lower (-0.78).
  - Both differences are within the epoch-to-epoch spread, so there is no
    clear gain. Hints with retrieval BLEU about 10.7 are far below the
    oracle's 18.
- **Self-fit train hints.** Retrieval BLEU 19.01 / F1@3 41.61
  (`metrics_train_selffit.json`), i.e. oracle-like, vs 10.66 at test. This
  is the C3 mismatch that `CSL_twopass_selffit` tests.
- **fold2 aborted (10-26 01:59).** NaN loss at epoch 3, step 1491, with
  finite inputs. The batch is saved in
  `fold2_failed_nan_ep3/nonfinite_batch_rank0.pt`, and the directory was
  kept under that name. fold2 is rerun with `--skip_nonfinite_loss`.

### C. Word-Level Dictionary

- **WLASL ISLR reproduction.** Test P-I/P-C 62.96 / 60.48 (paper
  63.13 / 60.90).
- **Coverage.** 56% of H2S content tokens have a WLASL sign (about 4 per
  sentence). WLASL + ASL-LEX (with translations) cover 77%.
- **Examples per WLASL word.** The median number of train subtitles
  containing a word is 12 for H2S and 83 for H2S + OpenASL (509k
  occurrences in total).
- **Spotting prototype.** 300 H2S train sentences, no training, sliding
  windows of 16–64 frames.

| Candidates | Top-1 | Top-5 | Top-10 |
|---|---|---|---|
| This sentence's glosses (n = 1,406) | 33.6% | 49.4% | 56.5% |
| Another sentence's glosses, control (n = 1,273) | 1.5% | 5.4% | 9.9% |

- **Spotting findings.**
  - The spotted frames follow the subtitle word order: Spearman 0.86,
    against −0.02 when shuffled.
  - About 5% of top-1 hits are chance hits.
  - 91% of hits come from 16-frame windows.

### A. Sentence Retriever (H2S test)

| Run | V2T R@1 / R@5 | T2V R@1 / R@5 | Retrieval BLEU-4 (s2t top-1) |
|---|---|---|---|
| S3dn (best) | 35.4 / 52.6 | 38.8 / 57.3 | 1.86 |
| CiCo (reference) | 51.6 | 56.6 | — |

## Data and Checkpoints

| Item | Path |
|---|---|
| CSL-Daily poses | `/data/hwu/Uni_sign_repo/data/CSL_Daily/pose_format/pose-rtmpose-192-clean/`, linked from `dataset/CSL_Daily/pose_format` |
| CSL-Daily SLT checkpoint (pose-only) | `/data/hwu/Uni_sign_repo/out/csl_daily_pose_only/csl_daily_pose_only_slt.pth` |
| CSL-News stage-1 pretrain (pose-only) | `/data/hwu/Uni_sign_repo/out/csl_pose_pretrain/csl_stage1_weight.pth` |
| Official CSL-Daily predictions | `/data/hwu/Uni_sign_repo/out/eval_csl_daily_pose_only/` |
| WLASL poses and ISLR checkpoint | `/data/hwu/Uni_sign_repo/data/WLASL/`, linked from `dataset/WLASL/pose_format` |
| ASL-LEX 2.0 metadata | `/data/hwu/Uni_sign_repo/data/ASL-LEX/` |
| ASL Citizen | not downloaded (45.9 GB RGB, no poses) |

## Step Log (newest first)

Every step is listed with its time, what was done, the command or script,
the files touched, the output location and the outcome. "Scratchpad" means
`/tmp/claude-1693400495/-home-student-hwu-Workplace-Uni-Sign/d92967e9-9115-4eb4-a65c-98be30f6227a/scratchpad/`.
That folder is temporary, so any script there worth keeping should be
moved into the repo.

### 2026-10-26

- **04:00–04:20: stage-2 code written (not run).**
  - New `graded_retriever.py`; design in plan doc §11.3.
  - Smoke test: 3 steps on GPU 2 (shared with `twopass_selffit`), output
    in the scratchpad only. Epoch 0 (= S2dn) reproduces S3dn's test
    retrieval BLEU 5.30, so the evaluation matches.
  - Corrected the 10-25 claim about S2dn's dev split. CSL used the real
    dev split with within-dev retrieval, not a 5% pseudo-dev (Results /
    F).
  - G1/G2 await the go-ahead.
- **03:40–03:55: overnight results recorded; fold2 rerun chained.**
  - H2S word runs finished (~02:00): `H2S_words` 15.01, `H2S_oracle_words`
    15.10, `H2S_gtwords` 16.15 (`H2S_empty` 15.13; Results / E).
  - `CSL_twopass_syn` finished: test 25.50 at the best dev epoch, but dev
    is 24.79 vs 25.57. Mixed (Results / G).
  - The queue started fold0 (GPU 3, 00:45), fold1 (GPU 0, 01:07), fold2
    (GPU 1, 01:10), fold3 (GPU 1, 01:59, after fold2 failed) and
    `CSL_twopass_selffit` (GPU 2, 02:41).
  - fold2 aborted on a NaN loss at epoch 3. Its directory was renamed to
    `fold2_failed_nan_ep3`.
  - `script/twopass_csl.sh`:
    - training now passes `--skip_nonfinite_loss`;
    - new `jobs NAME...` mode runs named jobs in order, each once its
      inputs exist and a GPU is free.
  - Stopped the first queue's main loop (pid 1924836) so it cannot also
    start `xfit`. Its four running jobs continue.
  - Started `jobs fold2 CSL_twopass_xfit` (pid 2099136,
    `queue_jobs.out`).
  - Git: fixed `.gitignore` (`__pycache__/` anywhere, `*.pyc`,
    `dataset/`). The 43 committed `.pyc` files were removed from the
    index (staged, not committed; local files kept).

### 2026-10-25

- **22:50: planning written down.** Added the stage-2 run plan (CSL
  first; G1/G2/G3, sentence-grouped dev, retrieval-BLEU selection) as
  plan doc §11.3, and the section "Planned, Not Started" here. Nothing
  launched.
- **22:30: stage-1 queue bug fixed and relaunched.**
  - `selffit_preds` finished at 22:00 (train BLEU 99.75, dev 25.18;
    Results / G).
  - The queue then exited at 22:01 (`twopass_csl/queue_1st_stopped.out`).
    Its stop check fired whenever *any* pending job had missing inputs,
    even though those inputs come from the folds that were still waiting
    for a GPU.
  - Fixed: it now stops only when nothing is running and *no* pending job
    is ready. Relaunched 22:32 (pid 1924836); it waits for idle GPUs.
  - About 30 min of queue time was lost. No GPU sat idle, because the
    folds need GPUs that are still busy.
  - Partial results: H2S word runs at epochs 4–5; `twopass_syn` at
    epoch 4 (see B/E/G).
- **20:58–21:10: stage 1 started on the idle GPU (user: "空闲的GPU可以先跑阶段1试试").**
  - Dev/test hints: `draft_retrieval.py --source preds` on
    `csl_daily/baseline_eval/{dev,test}_tmp_pres.txt` (tag `base`).
  - Train noise sweep 0.3–0.6 (`--source synthetic`, CPU); 0.5 is
    chosen to match test hint quality (Results / G).
  - `CSL_twopass_syn` launched by hand on GPU 2 at 21:00 (detached,
    port 29611).
  - `fine_tuning_rag.py`: new `--train_label / --dev_label / --test_label`
    (CSL only). These are needed to train fold translators and to
    translate train clips with `--eval`. Defaults are unchanged, so
    running jobs are unaffected.
  - New `script/twopass_csl.sh` (queue; skip + summary): fold label files
    in `twopass_csl/folds/`, then `selffit_preds` → `fold0..3` →
    `CSL_twopass_selffit` → `CSL_twopass_xfit`. A training job needs
    ≥ 40 GB free, i.e. an idle GPU, so the H2S runs are not touched.
    `selffit_preds` (≥ 16 GB) started next to `twopass_syn` on GPU 2.
  - Found that CSL dev/test share only 0.2% of sentences with train, so
    S2dn's dev R@1 of 61% does not reflect RAG retrieval (Results / F;
    the cause was corrected on 10-26).
  - No seed repeats and no reruns of existing experiments.
- **20:55: H2S rerank result recorded** (Results / F).
  - Pool best ORACLE 7.03 vs 7.8; rerank 3.78 vs TF-IDF on the translation
    3.20; oracle top-1 in the pool for 34.8% of clips.
  - GPU 2 is idle; new runs wait for the current batch (user).
- **20:30–20:45: contribution framing, runs deferred.**
  - The user wants the translate-retrieve-refine idea, with matched
    train/test hint quality, packaged as a contribution. The framing and
    the planned runs are in plan doc §11.2.
  - Prior art for the components (Iter-RetGen, HyDE/GAR, deliberation
    networks, cross-fitting) is listed there. No SLT work was found in one
    search.
  - New `draft_retrieval.py` (draft → TF-IDF top-3 distinct; synthetic or
    prediction drafts) is written but **not run**.
  - The user asked to start the new runs only after the current batch
    (`H2S_words`, `H2S_gtwords`, `H2S_oracle_words`, H2S rerank)
    finishes.

- **20:00–20:20: progress.**
  - The CSL translator rerank finished. Rerank gives 7.57, below TF-IDF on
    the translation (10.71). The pool's ORACLE best is 16.84, and it holds
    the oracle top-1 for 73% of clips; see Results / F.
  - First attempt: a dtype error (the dataset yields float64), fixed.
  - Full H2S WLASL hints: precision 15.9%, against a 14.9% frequency
    prior.
  - `H2S_words` (GPU 0) and `H2S_gtwords` (GPU 1) started automatically.
  - The H2S rerank is chained after its candidate job. The first chain
    would have matched itself in `pgrep`; it was replaced by a wait on
    the PID.

- **19:45–20:00: strong retriever (line F) started.**
  - The user agreed on graded-relevance training and a high-capacity
    retriever. The design was added to plan doc §11.1, which extends the
    §11 roadmap that already listed oracle distillation, reranking and
    cross-fitting.
  - Step 1 is eval only: `rerank_retrieval.py` (new). The candidate pool is
    S3dn top-50 (`signrag_retrieval.py --k 50 --distinct_text`, written to
    `$OUT/rerank/cand_s3dn_{csl,h2s}/`) plus TF-IDF top-50 on the baseline
    translation. The pool is reranked by the SLT model's
    log p(candidate | pose).
  - CSL runs on GPU 2, shared with the spotting job → `$OUT/rerank/csl/`.
  - ORACLE pool-coverage numbers are reported for analysis only.
- **19:30: translator-as-retriever measured** (see Results / E):
  retrieval BLEU 10.71 (CSL) and 3.20 (H2S), without GT.

- **19:00–19:20: status check, failures diagnosed, reruns started.**
  - All GPUs were idle from 10-25 01:54 to 19:09: the queue had finished
    and the failed items could not rerun on their own.
  - **Failure 1: H2S spotting (10-24 17:51) ran out of GPU memory.** I had
    raised bs to 64, which needs ~15 GB, while the shared training jobs
    used ~29 GB of 44.5 GB. Only 1,000 train clips were written.
  - **Failure 2: partial hints.** The chained `hints` step then ran on that
    partial dump, wrote `hints_*` / `oracle_gtwords_*` before crashing,
    and the queue (readiness = files exist) launched `H2S_words` and
    `H2S_gtwords`. Both failed within 15 s (missing clips).
  - **Fixes.**
    - `hints` now refuses an incomplete dump unless `--allow_partial` is
      given, and writes `DONE` at the end. The queue waits for `DONE`.
    - The partial output was kept as `wlasl_hints_partial_1000clips_20261024/`.
    - Spotting was rerun with bs 32 (validated at ~8.6 GB), one shard per
      GPU.
  - **Failure 3: `H2S_oracle_words` aborted after 27 min on a NaN loss**
    (`nonfinite_batch_rank0.pt`). It is rerun with `--skip_nonfinite_loss`
    (skip isolated steps, abort after 5 in a row). The flag was added to
    the remaining H2S runs only; the finished runs never hit a non-finite
    step.
  - **Failure 4: the two `CSL_real_d` diagnostics ran out of memory** on
    GPU 0, where the queue had placed `H2S_oracle_d`. They are rerun on
    GPU 3.
  - **Relaunch slip.** The relaunched queue placed `H2S_oracle_words` on
    GPU 0 before the spot shard there had allocated memory. I stopped it
    after ~10 s (no epoch) and relaunched the queue with FREE_MB=2000 after
    the shards had allocated.
  - **Results collected:** see the tables at the top of Results / E.

### 2026-10-24

- **17:50–18:00: word retrieval validated, full H2S spotting started.**
  - CSL CTC spotter finished: test WER 31.36, gloss-set F1 76.5; the
    cross-fitted train WER 32.81 matches test.
  - H2S WLASL validation: unconstrained precision about 17% (frequency
    prior 14%). The window choice is 16 frames at stride 8.
  - Full H2S spotting started on GPUs 0–3, shared with the training jobs.
  - The idle queue was restarted with `CSL_predgloss` second (no job had
    been launched).
  - Code: `word_retrieval_h2s.py` gained `--use_lens` / `--keep_stride` for
    the hints step.

- **15:03–15:10: hint training scheduled (user request: sentence-level
  TF-IDF oracle first).**
  - `CSL_oracle_d` has been training on GPU 0 since 14:57.
  - `H2S_oracle_d` is first in the new `script/word_hint_queue.sh`
    (pid 992495, detached).
  - The queue also holds, in order: `CSL_oracle_words`, `H2S_oracle_words`,
    `CSL_predgloss`, `H2S_words`, `H2S_gtwords`. A job starts on the first
    GPU with no queue job and < 12 GB used, once its hint files exist.
  - New code, uncommitted, for review:
    - `tfidf_word_hints.py`;
    - `word_retrieval_h2s.py` (spot / hints);
    - `word_retrieval_csl.py` (CTC, cross-fit);
    - `datasets_rag.py` / `fine_tuning_rag.py` gain `--rag_mode hint`, which
      reads a precomputed hint string per clip;
    - `script/word_hint_queue.sh`.
  - lemminflect was copied to `/data/hwu/Uni_sign_repo/pylib` for
    `word_retrieval_h2s.py`; the conda env is unchanged.
- **15:00: lane B shell stopped.** Its next step, the old H2S spotting, is
  replaced by `word_retrieval_h2s.py spot`. `gloss` (pid 975844) keeps
  running. The step was also removed from `csl_daily_overnight.sh`.
- **14:51–14:57: lane D diagnostics done.** `CSL_real` with no hint gives
  24.91 test B4, with distinct oracle hints 24.56; see Results / E.
  `CSL_oracle_d` then started.

- **14:50–15:00: cancelled the seed-repeat runs (user request).**
  - The user asked for no seed repeats. The other runs (`real_d`,
    `oracle_d`, `gloss`, `random_d`, diagnostics and H2S spotting) continue.
  - **`empty_s43` (GPU 3) was stopped.** Its 11-process tree (deepspeed
    runner pid 975843 and descendants) got SIGTERM at 14:50:54 and all
    processes exited, so no SIGKILL was needed.
    - Lane C logged `[FAIL] empty_s43`; that line comes from the stop, not
      from a crash.
    - Lane C then started `random_d` on GPU 3 at 14:50:54, as intended.
    - `ra_sft/CSL_empty_s43/` is kept as is. It is empty, because no epoch
      had finished.
  - **`real_d_s43` (queued) was cancelled.**
    - The running lane shells had already loaded the queue functions, so
      editing the script could not cancel it.
    - Instead, the lane A shell (pid 975832), the only process that would
      launch `real_d_s43`, was stopped with SIGTERM.
    - Its running child `real_d` (deepspeed pid 978652) was not touched and
      keeps running, reparented to init.
    - A detached replacement notifier (pid 980945) appends
      `[done] real_d <date>` to `overnight_A.out` once pid 978652 exits.
      Lane D waits for that line before its `real_d` diagnostics.
  - **Script.** `real_d_s43` and `empty_s43` were removed from
    `script/csl_daily_overnight.sh`, so a rerun cannot start them.
  - **Also fixed: a stale wait was blocking lane D.**
    - My earlier background wait shell (pid 966974) had the oracle-eval
      output path in its own command line, so its `pgrep` matched itself
      and never ended.
    - Lane D's `pgrep` matched the same shell, so lane D sat waiting from
      14:47 to 14:51.
    - Killing that shell (no experiment involved) released lane D, which
      started `diag_real_emptyhint` on GPU 0 at 14:51:15.
  - **Process check at 14:55.**
    - GPU 0: lane D diagnostic eval.
    - GPU 1: `real_d`.
    - GPU 2: `gloss`.
    - GPU 3: `random_d`.
    - No process runs with seed 43; the only command line containing
      "s43" is the notifier's message text.

- **14:47: started the overnight queue.**
  - Command: `bash script/csl_daily_overnight.sh all`, detached
    (pid 975827).
  - Four lanes on GPU 0–3; see Running Jobs.
  - Smoke tests passed first: `--distinct_text` gives 0 duplicate rows and
    k = 3 for every method; `--rag_mode gloss` trains and evaluates.
- **14:45: code changes, uncommitted, for review.**
  - `signrag_retrieval.py`: `--distinct_text` / `--distinct_fetch` and
    `distinct_topk()`, applied in `save_results` when on.
  - `datasets_rag.py`: `render(name, gloss)` with the new `gloss` mode
    (`GLOSS_PREFIX = "Glosses: "`).
  - `fine_tuning_rag.py`: `--rag_mode` choices gain `gloss`.
  - New `script/csl_daily_overnight.sh`.
  - Scratchpad `spot_proto.py`: the control falls back to the next sentence
    when every sentence is selected.
- **14:40: found the duplicate top-3 bug** (see Results / D).
- **14:30: oracle-hint diagnostic.** The `CSL_real` model with TF-IDF
  oracle hints gives dev 25.24 / test 24.77, i.e. no change. Output:
  `$OUT/csl_daily/oracle_eval_real/`.
- **14:25: compared our setup with the SignRAG paper.** The table is in
  Results / D.

- **14:20: checked the CSL-Daily RA-SFT results.**
  - Both runs finished at 06:16 without crashing, so the
    `--skip_nonfinite_loss` restart question is moot.
  - Best dev epoch, test B4: real 24.62, empty 25.18. No retrieval gain.
  - Per-epoch logs: `$OUT/ra_sft/CSL_{real,empty}/log.txt`.
  - All GPUs have been idle since 06:16.

- **01:45: status check and the user's decision.**
  - The user said not to run E2 empty. It had already stopped on its own
    (crash at epoch 2, see below) and will not be rerun.
  - E2 real stopped at epoch 14 with a NaN loss at 00:14. The chain then
    started E2 empty at 00:14, which crashed with an inf loss at 01:12.
  - Because the E2 chain and the retriever had both exited, the queued
    waiter started CSL-Daily RA-SFT at 01:13: real on GPU 0,1, empty on
    GPU 2,3.
- **00:18: CSL-Daily retriever pipeline finished.**
  - S2dn: 27 epochs, dev r5_mean 78.16.
  - S3dn test R@1: V2T 58.67, T2V 59.52.
  - s2t top-1 retrieval BLEU-4: 5.30 on test (oracle 18.02, random 0.18).
  - The RA-SFT inputs were written.
  - Tables are in Results / D.

- **00:05: queued CSL-Daily RA-SFT.**
  - The user decided to let E2 (real, then empty) finish first.
  - A detached waiter polls once a minute. When both the E2 chain
    (pid 373296) and the CSL retriever (pid 547568) have exited, it runs
    `GPU_REAL=0,1 GPU_EMPTY=2,3 bash script/signrag_csl_daily.sh rasft`.
  - Log: `$OUT/csl_daily/rasft.out`.
- **00:00: started the CSL-Daily retriever pipeline on GPU 3.**
  - Command: `GPU=3 bash script/signrag_csl_daily.sh retriever`, detached
    (ppid 1).
  - Stages:
    1. `extract` train/dev/test with `csl_daily_pose_only_slt.pth` →
       `$OUT/feats/E_csl/`;
    2. `S1_E_csl`: frame/pool retrieval plus random and TF-IDF baselines;
    3. `S2dn_E_csl`: fine objective, exact positives, batch 256, the same
       arguments as H2S S2dn;
    4. `S3dn_E_csl`: s2t/t2s retrieval with pair recall;
    5. RA-SFT inputs `$OUT/rag_inputs/S3dn_E_csl/`: train queries
       filtered, plus dev and test.
  - Log: `$OUT/csl_daily/retriever.out`.
- **23:58 (10-23): smoke-tested RA-SFT on CSL-Daily.**
  - Run: `fine_tuning_rag.py --dataset CSL_Daily --rag_mode real
    --limit 16 --epochs 1` from `csl_stage1_weight.pth` on GPU 3, with
    dummy full-coverage retrieval files.
  - Outcome: strict load OK, and the log row has both `dev_*` and `test_*`.
  - The first attempt failed with `KeyError`: the smoke retrieval covered
    only 64 clips, and the dataset checks coverage before `--limit` is
    applied. This was a test artefact, not a code bug.
  - Output: scratchpad `csl_smoke/`.
- **23:56 (10-23): smoke-tested retrieval on CSL-Daily.**
  - `extract --limit 64` on train/dev/test, then `retrieve` (frame, pool,
    baselines) and `evaluate`.
  - Outcome: char-level metrics are computed, train queries are filtered
    (em@1 = 0), and `video_id` is the clip name.
  - Output: scratchpad `csl_smoke/`.
- **23:50 (10-23): code changes for Chinese and the dev split.** The code
  is uncommitted and awaits review.
  - `signrag_common.py`:
    - new `is_cjk`;
    - new `text_tokens`, which gives characters for Chinese and the old
      `norm_text().split()` otherwise;
    - new `bleu_text`, the char-level text exactly as
      `fine_tuning.evaluate` builds its CSL references;
    - `video_id` returns the clip name for CSL-Daily names
      (`S######_P####_T##`) and the old 11-character YouTube ID otherwise.
  - `signrag_retrieval.py`:
    - `_token_set` uses `text_tokens`;
    - `retrieval_metrics` passes refs and hyps through `bleu_text`;
    - `tfidf_score_fn` uses char 1–2-grams for Chinese;
    - `--dataset` choices gain `CSL_Daily`.
  - `signrag_contrastive.py`: `--dataset` choices gain `CSL_Daily`.
  - `fine_tuning_rag.py`:
    - on datasets with a dev split, the log now has `dev_*` (the selection
      split) and the real `test_*`. Before, the dev stats were written
      under `test_*` and the test stats were dropped;
    - `--rag_mode real` on a dev-split dataset requires `--rag_dev_json`;
    - How2Sign behaviour is unchanged.
  - New `script/signrag_csl_daily.sh`:
    - stages `retriever`, `rasft` and `summary`;
    - every stage skips when its output exists;
    - the RA-SFT recipe follows the official CSL-Daily recipe:
      `csl_stage1_weight.pth`, 20 epochs, effective batch 32, lr 3e-4,
      fp32, dev selection.
  - Unit check: English tokens and BLEU text are unchanged, and the Chinese
    BLEU text equals the `fine_tuning.evaluate` reference string.
- **23:45 (10-23): scored the official CSL-Daily prediction files.**
  - Script: scratchpad `csl_bleu_files.py`. It strips the `sample: …`
    prefix and scores per character, as `fine_tuning.evaluate` does.
  - `out/eval_csl_daily_pose_only`: dev 25.27, test 25.70 B4.

### 2026-10-23

- **23:50: CSL-Daily baseline eval.**
  - Command: `deepspeed --include localhost:3 fine_tuning.py --finetune
    …/csl_daily_pose_only_slt.pth --dataset CSL_Daily --task SLT --eval`
    (bf16 default).
  - Result: dev 25.14 / test 25.61 B4.
  - Output: `$OUT/csl_daily/baseline_eval/`.
- **23:45: data check.** Only 0.2% of CSL-Daily test sentences appear
  verbatim in train (H2S: 3.0%). Train has 6,578 unique texts for 18,401
  clips.
- **23:40: the supervisor's request.** Reproduce SignRAG on CSL-Daily.
  Line D became the top priority.
- **23:38: spotting prototype finished (300 H2S train sentences).**
  - Top-1 33.6% vs control 1.5%; word-order Spearman 0.86.
  - Scripts: scratchpad `spot_proto.py` and `spot_analyze.py`.
  - Output: `$OUT/word_dict/spot_proto/`.
  - Details are in plan doc §10b2.
- **23:30: examples per WLASL word in the H2S/OpenASL subtitles.**
  Scratchpad `word_counts.py`. The table is in plan doc §10b2.
- **23:22: CSL-Daily poses ready.**
  - 20,654 pkl unzipped; the symlink `dataset/CSL_Daily/pose_format` was
    re-pointed to the inner `pose-rtmpose-192-clean/`.
  - Check: 0 missing files for train/dev/test labels.
- **23:20: WLASL ISLR reproduction.**
  - Command: `fine_tuning.py --dataset WLASL --task ISLR --eval` on GPU 3.
  - Result: P-I/P-C 62.96 / 60.48.
  - Output: `$OUT/word_dict/wlasl_islr_eval/`.
  - `dataset/WLASL/pose_format` was linked to the unzipped poses
    (14,289/3,916/2,878 files).
- **23:18: CSL-Daily downloads from HF `ZechengLi19/Uni-Sign`.**
  - Files: `csl_daily_pose_format.zip`, `csl_daily_pose_only_slt.pth` and
    `csl_stage1_weight.pth`. The paths are in the Data table above.
  - Not downloaded: the RGB variants, which need RGB video, and CSL-News
    poses (255 GB).
- **19:50: dictionary coverage.**
  - WLASL covers 56% of H2S content tokens; WLASL + ASL-LEX (with
    translations) covers 77%.
  - Script: scratchpad `dict_coverage.py`. Plan doc §10b2.
- **18:53: E2 real RA-SFT started.**
  - Command: `INIT=E2 MODES=real GPU_REAL=0,1,2 bash
    script/train_rag_h2s.sh`, then E2 empty on the same GPUs, as a detached
    chain (pid 373296).

### 2026-10-22

- E1 RA-SFT real/empty/random finished. Retrieval brought no gain (real
  15.04 vs empty 15.13 B4).
- E2 was set up to start from the OpenASL checkpoint, matching the SignRAG
  setting.
