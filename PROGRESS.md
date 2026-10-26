# Hinglish Hate-Speech Detection Pipeline: Progress & Status

**Repository:** `c:\nlp-moderation`  
**Current Milestone:** Stages 1–6 Complete  
**Last Updated:** October 2026  

---

## 1. Pipeline Stages 1–4 Summary & Key Metrics

### Stage 1: Load and Audit
- **Raw Corpus:** `data/raw/hinghate/hinglish_corpus_final.tsv` (248,129 rows $\times$ 4 columns: `text`, `label`, `source`, `script`).
- **Class Distribution:**
  - Non-hate (`0`): 229,462 (92.48%)
  - Hate (`1`): 18,667 (7.52%)
- **Source Breakdown:** `youtube`: 233,425 | `reddit`: 10,389 | Blank (`""`): 4,315.
- **Script Breakdown:** `roman`: 185,082 | `deva`: 51,374 | `mixed`: 7,358 | Blank (`""`): 4,315.
- **Duplicate Audit:** 111,449 exact duplicate rows (44.92%); 136,680 unique texts (125,802 non-hate, 10,878 hate). Exactly **0 conflicting labels** across duplicate texts.
- **Confound Discovery:** Script is a major empirical confound. Post-dedupe hate rate in Devanagari is 14.07% and Mixed is 14.24%, compared to only 5.84% in Roman. Reddit contains only 41 unique hate rows (<1%).
- **Sequence Length:** Median 13 words; max 5,250 words; only 2.18% of unique texts exceed 64 words.

### Stage 2: Clean, Dedupe, and Split
- **Cleaning Strategy:**
  - Removed only `\u200b` and `\ufeff`. Indic ligatures `\u200c` (ZWNJ) and `\u200d` (ZWJ) were preserved intact.
  - Collapsed repeated characters to maximum 2; collapsed whitespace; preserved `<USER>` placeholders, emojis, casing, and both scripts.
  - Imputed blank `source` and `script` to `"unknown"` (0 rows dropped).
- **Normalized-Key Deduplication:** Deduplicated on lowercase, punctuation-stripped, space-normalized keys $\to$ **136,632 unique examples** (0 label conflicts, 111,497 total rows removed).
- **Stratified 80/10/10 Split on (label, script), Seed 42:**
  - **Train (80%):** 109,305 rows (8,701 hate, 7.96%) $\to$ [train.parquet](file:///c:/nlp-moderation/data/processed/train.parquet)
  - **Train Subsampled (1:3 ratio):** 34,804 rows (8,701 hate, 26,103 non-hate, 25.00%) $\to$ [train_subsampled.parquet](file:///c:/nlp-moderation/data/processed/train_subsampled.parquet)
  - **Validation (10%):** 13,663 rows (1,088 hate, 7.96%) $\to$ [val.parquet](file:///c:/nlp-moderation/data/processed/val.parquet)
  - **Test (10%):** 13,664 rows (1,088 hate, 7.96%) $\to$ [test.parquet](file:///c:/nlp-moderation/data/processed/test.parquet)
  - **Zero Leakage:** Exactly 0 normalized key overlap across train, val, and test splits.
- **MuRIL Token-Length Audit (`google/muril-base-cased`):**
  - Train token percentiles: p50 = 20, p90 = 50, p95 = 67, p99 = 105.
  - Share $> 64$ tokens: 5.48% overall (5.36% non-hate, 6.90% hate). At `max_len = 64`, 94.52% of texts fit without truncation.

### Stage 3: Baseline (TF-IDF + Logistic Regression)
- **Features:** Word n-grams (1–2) + Char n-grams (2–5) with sublinear TF scaling (230,898 features).
- **Training:** Fit on `train_subsampled.parquet` ($N = 34,804$) with `class_weight='balanced'`.
- **Validation Tuning:** Optimal $C = 10.0$, optimal decision threshold $\tau = 0.73$.
- **Test Metrics ($N = 13,664$):**
  - **Overall:** Macro-F1 = **0.7385**, Hate Precision = **0.4701**, Hate Recall = **0.5919**, Hate-Class F1 = **0.5240**, PR-AUC = **0.5427**.
  - **Per-Script Hate F1:** Roman = **0.5321**, Devanagari = **0.5010**, Mixed = **0.6667**, Unknown = **0.3448**.

### Stage 4: MuRIL Fine-Tuning (3 Seeds: 42, 43, 44)
- **Model:** `google/muril-base-cased` (Sequence Classification, 2 labels, FP32).
- **Settings:** lr 2e-5, AdamW, weight decay 0.01, linear warmup 10%, 3 epochs, per-device batch 8, gradient accumulation 4 (effective batch 32), class-weighted cross-entropy ($w_0 = 0.6667, w_1 = 2.0000$).
- **Per-Seed Test Performance:**
  - **Seed 42 ($\tau = 0.89$):** Macro-F1 = **0.8823** | Hate-F1 = **0.7832** | PR-AUC = **0.8759**
  - **Seed 43 ($\tau = 0.90$):** Macro-F1 = **0.8842** | Hate-F1 = **0.7873** | PR-AUC = **0.8711**
  - **Seed 44 ($\tau = 0.84$):** Macro-F1 = **0.8865** | Hate-F1 = **0.7917** | PR-AUC = **0.8757**
- **Multi-Seed Test Aggregate (Mean $\pm$ Std):**

| Script Stratum | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **0.8825 ± 0.0016** | **0.7882 ± 0.0117** | **0.7794 ± 0.0179** | **0.7836 ± 0.0034** | **0.8742 ± 0.0027** |
| **Roman** | 0.8744 ± 0.0008 | 0.7772 ± 0.0205 | 0.7507 ± 0.0224 | **0.7633 ± 0.0019** | 0.8586 ± 0.0028 |
| **Devanagari (`deva`)** | 0.8864 ± 0.0029 | 0.7952 ± 0.0020 | 0.8153 ± 0.0127 | **0.8051 ± 0.0052** | 0.8884 ± 0.0034 |
| **Mixed** | 0.8889 ± 0.0078 | 0.8327 ± 0.0231 | 0.7853 ± 0.0098 | **0.8082 ± 0.0132** | 0.9212 ± 0.0117 |
| **Unknown** | 0.9196 ± 0.0279 | 0.8485 ± 0.0525 | 0.8485 ± 0.0525 | **0.8485 ± 0.0525** | 0.9043 ± 0.0240 |

- **Threshold Tuning Sensitivity:**
  - Tuned $\tau \approx 0.88$: Balanced precision/recall (**0.7847** P / **0.7840** R, Hate-F1 **0.7844**).
  - Default $\tau = 0.50$: Prioritizes recall (**0.8382** R / **0.7421** P, Hate-F1 **0.7872**).
- **Nearest-Train-Neighbour Similarity Analysis:**
  - Evaluated max character 2–5 cosine similarity to training set across quartiles Q1–Q4.
  - In Q1 (test texts most distant from training n-grams), Hate-F1 remains **0.7558–0.7819** and PR-AUC is **0.8425–0.8676**, demonstrating genuine contextual generalization rather than surface memorization.

---

## 2. Engineering Decisions & Constraints

1. **Precision:** FP32 exclusively. Empirical benchmarks on the GTX 1660 Ti showed FP16 was slower (0.54 s/step vs 0.19 s/step in FP32) with no VRAM reduction.
2. **Memory Budget & Batching:** Per-device batch size 8 with gradient accumulation 4 (effective batch 32). Peak memory during training measured **5.02 GB / 6.00 GB** VRAM; no CUDA OOM occurred.
3. **Sequence Length:** `max_len = 64` with dynamic batch padding via HuggingFace tokenizer collator. Captures $>94.5\%$ of token sequences without truncation while fitting comfortably within VRAM.
4. **Python Environment:** Official Python 3.13 in `C:\nlp-moderation\.venv`. Windows specifics applied: `DataLoader(num_workers=0)`, UTF-8 encodings on all file I/O, no raw Devanagari or emoji printed to console.
5. **Evaluation Hygiene:** Hyperparameters and decision thresholds were strictly tuned on validation; the test set was evaluated once under fixed thresholds.
6. **Data & Checkpoint Licensing:** `data/` and `results/checkpoints/` are strictly gitignored to comply with the research-only corpus license and avoid large binary blobs.

---

## 3. Known Caveats & Risks

1. **Label Noise / Broad Hate Definition:** Qualitative inspection of [label_sample.csv](file:///c:/nlp-moderation/results/label_sample.csv) shows substantial label noise: aggressive political debate, partisan slogans ("Congress Mukt Bharat"), and counter-speech were frequently annotated as hate speech (`label=1`).
2. **Platform & Script Confound:** YouTube dominates the corpus (95.3% of unique examples) and contains virtually all the hate signal (10,724 hate rows). Reddit contains only 41 hate rows. Script is confounded with platform and topic.
3. **External Generalization Unverified:** While nearest-neighbour quartile analysis confirms in-distribution generalization, performance on standard external benchmarks (e.g. HASOC, HEOT) remains pending.

---

## 4. Repository File Map

```
c:\nlp-moderation\
│
├── configs\
│   ├── stage1_audit.yaml               # Audit parameters and reference paths
│   ├── stage2_preprocess.yaml          # Cleaning, dedupe, and stratified split configs
│   ├── stage3_baseline.yaml            # TF-IDF + Logistic Regression configs
│   └── stage4_muril.yaml               # MuRIL training hyperparameter configuration
│
├── src\
│   ├── data\
│   │   ├── audit.py                    # Stage 1 audit and inspection sampling
│   │   └── preprocess.py               # Stage 2 text cleaner, deduplicator, splitter
│   ├── models\
│   │   └── baseline.py                 # Stage 3 TF-IDF baseline and threshold tuner
│   ├── train\
│   │   ├── train_muril.py              # Stage 4 MuRIL fine-tuning and inference script
│   │   └── run_seeds.py                # Multi-seed runner and aggregator
│   └── utils\
│       ├── config.py                   # Config loaders and random seed setup
│       └── logging.py                  # CSV experiment tracking utility
│
├── results\
│   ├── audit.md                        # Full Stage 1 corpus audit report
│   ├── label_sample.csv                # 100 balanced rows (50 hate / 50 non-hate) for audit
│   ├── baseline_metrics.md             # Stage 3 baseline evaluation report
│   ├── muril_seed42_metrics.md         # Stage 4 Seed 42 evaluation report
│   ├── muril_seed43_metrics.md         # Stage 4 Seed 43 evaluation report
│   ├── muril_seed44_metrics.md         # Stage 4 Seed 44 evaluation report
│   ├── muril_multiseed_summary.md      # Multi-seed aggregated statistical summary
│   ├── preds_muril_seed42.parquet      # Seed 42 validation & test predicted probabilities
│   ├── preds_muril_seed43.parquet      # Seed 43 validation & test predicted probabilities
│   ├── preds_muril_seed44.parquet      # Seed 44 validation & test predicted probabilities
│   ├── checkpoints\                    # (Gitignored) Saved model checkpoints (*.pt)
│   └── runs.csv                        # Central experiment log with timestamped metrics
│
├── data\                               # (Gitignored) Raw & Processed data files
│   ├── raw\hinghate\                   # Original TSV corpus
│   └── processed\                      # train, train_subsampled, val, test Parquets
│
├── .gitignore                          # Ignores /data/, /.venv/, *.pt, results/checkpoints/
└── PROGRESS.md                         # Pipeline status document (this file)
```

### Stage 5: Error Analysis & 3-Seed Probability Ensemble
- **Model:** MuRIL 3-Seed Probability Ensemble (mean probability across seeds 42, 43, 44).
- **Validation Tuning:** Optimal decision threshold $\tau = 0.746$ (Val Macro-F1 = 0.8979, Val Hate-F1 = 0.8121).
- **Ensemble Test Performance ($N = 13,664$):** Macro-F1 = **0.8929**, Hate Precision = **0.8218**, Hate Recall = **0.7840**, Hate-Class F1 = **0.8024**, PR-AUC = **0.8904**.
- **Bootstrap 95% Confidence Intervals (1,000 resamples):**
  - Overall Hate-F1: **[0.7835, 0.8208]**
  - Roman Hate-F1: **[0.7608, 0.8135]**
  - Devanagari Hate-F1: **[0.7815, 0.8395]**
  - Paired Difference (Roman - Devanagari): **[-0.0636, 0.0157]** $\to$ **no detectable difference at this sample size (paired CI [-0.064, +0.016])**.
- **Near-Neighbour Label Agreement ($>0.80$ cosine to full `train.parquet`):**
  - 158 test rows (1.16% of test set). Overall agreement = **98.10%** (155/158, Wilson 95% CI: **[94.57%, 99.35%]**).
  - Non-Hate (`label=0`): **100.00%** (143/143, Wilson 95% CI: **[97.38%, 100.00%]**).
  - Hate (`label=1`): **80.00%** (12/15, Wilson 95% CI: **[54.81%, 92.95%]**). Disagreement: 3/15 = 20.00% (Wilson 95% CI: **[7.05%, 45.19%]**).
- **Political Keyword False-Positive Rate (FPR):**
  - Non-hate test rows with Roman political terms: **1.44%** (11/764, Wilson 95% CI: **[0.81%, 2.56%]**).
  - Non-hate test rows without Roman political terms: **1.47%** (174/11,812, Wilson 95% CI: **[1.27%, 1.71%]**).
  - Non-hate test rows with Any political terms (Roman+Deva): **1.52%** (20/1,315, Wilson 95% CI: **[0.99%, 2.34%]**).
  - Non-hate test rows without Any political terms: **1.47%** (165/11,261, Wilson 95% CI: **[1.26%, 1.70%]**).
  - Relative FPR Ratio = **1.04x** $\to$ **no detectable difference** (overlapping Wilson 95% CIs).
- **Artifacts Generated:** [results/error_analysis.md](file:///c:/nlp-moderation/results/error_analysis.md), [results/errors.csv](file:///c:/nlp-moderation/results/errors.csv) (top 50 FP / top 50 FN), and [results/audit_100.csv](file:///c:/nlp-moderation/results/audit_100.csv) (100 random hate test rows with empty `my_tag` column).

---

## 5. Stage 6: Verdict & Established Limitations

### The Verdict (Evaluated Under the Four Decision Rules)

1. **Rule 1 (Roman vs. Devanagari Script Gap):**
   - *Rule:* If Roman-script hate F1 is very low or far below Devanagari $\to$ need more Roman Hinglish data.
   - *Evidence:* 3-Seed Model Mean (evaluated individually): Roman Hate-F1 is **0.7633 ± 0.0019** vs. Devanagari **0.8051 ± 0.0052**. For the 3-seed probability ensemble, Roman Hate-F1 is **0.7894** (PR-AUC 0.8759) vs. Devanagari **0.8131** (PR-AUC 0.9028). The paired Roman-minus-Devanagari difference has a 95% bootstrap CI of **[-0.064, +0.016]**.
   - *Verdict:* Rule 1 is **NOT triggered**. There is **no detectable difference at this sample size (paired CI [-0.064, +0.016])**. We cannot conclude Roman data is deficient relative to Devanagari.

2. **Rule 2 (Multi-Seed Instability):**
   - *Rule:* If seed standard deviation of hate-class F1 is above ~2.0 points $\to$ results are unstable, need more data or cross-validation.
   - *Evidence:* Multi-seed SD across seeds 42, 43, 44 (single models) is **0.0034** (0.34 F1 points; Macro-F1 SD = 0.0016; PR-AUC SD = 0.0027). In contrast, the test Hate-F1 bootstrap SE is **~0.0095** (with a 95% CI of **[0.7835, 0.8208]** for the 3-seed probability ensemble, width = 0.0373), roughly 3x the seed SD on a like-for-like standard error vs standard deviation basis. Note that the bootstrap CI is evaluated on the ensemble, whereas the seed SD measures variability across individual models.
   - *Verdict:* Rule 2 is **NOT triggered**. Seed SD (0.0034) is well below the ~2.0 point threshold, and bootstrap sampling uncertainty (~0.0095 SE) is roughly 3x larger than seed SD (0.0034).

3. **Rule 3 (Label Quality Audit vs. Model Weakness):**
   - *Rule:* If errors are dominated by label noise rather than model weakness $\to$ need a cleaner benchmark.
   - *Evidence:* A manual hand-audit of 100 randomly sampled hate-labelled test rows (`label = 1`) in [results/audit_100.csv](file:///c:/nlp-moderation/results/audit_100.csv) reveals:
     - `neutral`: no visible hostility under the written guide (21–42% depending on whether ?/~ rows are counted; 18 of 42 carry ?, 6 carry ~): **42** / 100 (Wilson 95% CI: [0.33, 0.52])
     - `personal_abuse` (interpersonal toxicity): **22** / 100 (Wilson 95% CI: [0.15, 0.31])
     - `political_criticism` (partisan debate): **19** / 100 (Wilson 95% CI: [0.13, 0.28])
     - `group_hate` (strict identity-based hate): **17** / 100 (Wilson 95% CI: [0.11, 0.26])
     - Audit marks across 100 rows: **? 25**, **~ 7**, **! 4** (definitions: `?` contestable, `~` context could flip it, `!` flagged extreme).
     - Only 17% of audited positive examples represent strict group-based hate; 83% comprise comments with no visible hostility, personal insults, or political debate.
     - Note on False Positives: Inspection of the top-50 false positives in `results/errors.csv` is selected by extreme model confidence ($\bar{p} > 0.95$) and is **not representative** of all false-positive errors.
     - Note on False Negatives: Top false negatives (e.g., characters separated by spaces `l a u d e`, phonetic camouflage, implicit abusive sarcasm) represent **genuine model weaknesses**.
   - *Verdict:* Rule 3 is **TRIGGERED**. Ground-truth positive labels in HINGHATE are heavily dominated by non-hate and personal/political abuse rather than strict group hate, setting a noisy optimization target.

4. **Rule 4 (Final Dataset Verdict):**
   - *Rule:* Otherwise HINGHATE is enough, and an external test only adds a generalization claim.
   - *Verdict:* **HINGHATE is enough to train and benchmark within HINGHATE; not enough for general hate-speech claims (no external test; positive label broader than a strict definition).**

---

### Explicitly What is NOT Established

1. **External Generalisation:** The pipeline has only been tested in-distribution on the HINGHATE test split. Zero cross-dataset or cross-domain generalization has been verified.
2. **Label Quality:** A formal audit of 100 positive rows in [results/audit_100.csv](file:///c:/nlp-moderation/results/audit_100.csv) shows broad label definitions, but ground-truth quality across the entire dataset remains unverified.
3. **Single Annotator:** The audit of 100 rows was performed by a single annotator without second-pass adjudication or inter-annotator agreement metrics ($\kappa$).
4. **Positives-Only Audit:** Only rows with ground truth `label = 1` were audited; no systematic audit of ground truth `0` rows was conducted.
5. **No Parent Context:** Comments were annotated and classified in total isolation without the parent comment, video title, or conversational context.
6. **No Thread IDs:** The dataset does not preserve discussion thread IDs or video IDs, preventing hierarchical modeling or clustering of clustered conversations.
7. **Unverified Label Provenance:** Annotation guidelines/demographics not documented in the files we have; dataset page/paper not yet checked.
8. **Reddit Performance:** Reddit contains only 41 hate rows in the entire deduplicated corpus and only 6 hate rows in the test split. Test Hate-F1 on Reddit is **0.4444** (precision 0.6667, recall 0.3333). Generalization to Reddit or forum-style discourse is entirely unestablished.
9. **No Other Models:** Only MuRIL (`google/muril-base-cased`) has been evaluated; comparison against `xlm-roberta-base`, `HingRoBERTa`, or decoder baselines is unperformed.
10. **Any Claim from the 15-Row Near-Neighbour Analysis:** The finding that 3/15 (20.0%) near-neighbour hate instances disagreed with train labels has a Wilson 95% CI of **[7.05%, 45.19%]** (agreement Wilson 95% CI: **[54.81%, 92.95%]**). Because $N=15$ is tiny, no general quantitative claim about dataset-wide label error rate can be inferred from this statistic.
