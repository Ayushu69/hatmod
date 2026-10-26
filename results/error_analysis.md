# Stage 5: Comprehensive Error & Slice Analysis

- **Model Evaluated:** MuRIL 3-Seed Ensemble (arithmetic mean probability across seeds 42, 43, 44)
- **Tuned Decision Threshold:** `\tau = 0.746` (Tuned on Validation Macro-F1 = `0.8979`)
- **Evaluation Set:** Test Set ($N = 13,664$, natural 7.96% hate ratio)
- **Headline Performance:** Macro-F1 = **0.8929** | Hate Precision = **0.8218** | Hate Recall = **0.7840** | Hate-Class F1 = **0.8024** | PR-AUC = **0.8904**

---

## 1. Bootstrap 95% Confidence Intervals (1,000 Resamples over Test Set)

| Metric | Point Estimate | Bootstrap Mean | 95% Confidence Interval |
| :--- | :--- | :--- | :--- |
| **Overall Hate-Class F1** | **0.8024** | 0.8023 | **[0.7835, 0.8208]** |
| **Roman Hate-Class F1** | **0.7894** | 0.7890 | **[0.7608, 0.8135]** |
| **Devanagari Hate-Class F1** | **0.8131** | 0.8129 | **[0.7815, 0.8395]** |
| **Paired Difference (Roman - Devanagari)** | **-0.0237** | -0.0239 | **[-0.0636, 0.0157]** |

> **Key Takeaway:** There is **no detectable difference at this sample size (paired CI [-0.064, +0.016])** between Roman and Devanagari scripts.

---

## 2. Near-Neighbour Label Agreement Analysis

Evaluated using character 2–5 n-gram TF-IDF cosine similarity against the complete training set (`train.parquet`, $N = 109,305$):
- **Test rows with max cosine similarity $> 0.80$ to any train row:** **158** (1.16% of test set)
- **Overall Near-Neighbour Label Agreement:** **98.10%** (155/158, Wilson 95% CI: **[94.57%, 99.35%]**)
- **Test Label = 0 (Non-Hate, $N = 143$):** Nearest train neighbour agreed **100.00%** (143/143, Wilson 95% CI: **[97.38%, 100.00%]**).
- **Test Label = 1 (Hate, $N = 15$):** Nearest train neighbour agreed **80.00%** (12/15, Wilson 95% CI: **[54.81%, 92.95%]**). Disagreement: 3/15 = **20.00%** (Wilson 95% CI: **[7.05%, 45.19%]**).

> **Important Caveat:** Because $N=15$ is a very small sample size, the Wilson 95% CI is wide ([54.81%, 92.95%]). No statistically robust claim about dataset-wide label error rate can be drawn from this small slice alone.

---

## 3. Political Term Sensitivity & False-Positive Rate (FPR)

Non-hate test examples ($N = 12,576$) were audited for lexical bias against political and religious keywords:
- **Terms Audited:** `modi, bjp, congress, sarkar, rahul, yogi, hindu, muslim, jati, aarakshan` + Devanagari equivalents (`मोदी, बीजेपी, भाजपा, कांग्रेस, सरकार, राहुल, योगी, हिन्दू, हिंदू, मुस्लिम, मुसलमान, जाति, आरक्षण`).

| Slice | Sample Count ($N$) | False Positives | False-Positive Rate (FPR) | Wilson 95% CI | Relative FPR Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-hate with Roman political terms** | 764 | 11 | **1.44%** | **[0.81%, 2.56%]** | **0.98x** |
| **Non-hate without Roman political terms** | 11,812 | 174 | **1.47%** | **[1.27%, 1.71%]** | 1.00x (Baseline) |
| **Non-hate with Any political terms (Roman+Deva)** | 1,315 | 20 | **1.52%** | **[0.99%, 2.34%]** | **1.04x** |
| **Non-hate without Any political terms** | 11,261 | 165 | **1.47%** | **[1.26%, 1.70%]** | 1.00x (Baseline) |

> **Key Takeaway:** The FPR on texts containing political/religious keywords (1.52%, Wilson CI [0.99%, 2.34%]) vs neutral texts (1.47%, Wilson CI [1.26%, 1.70%]) shows **no detectable difference** (overlapping 95% confidence intervals).

---

## 4. Fine-Grained Slice Performance Breakdown

### A. Script Slices
| Script Stratum | Total $N$ | Hate Count | Macro-F1 | Hate-P | Hate-R | Hate-F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Roman** | 9,965 | 583 | 0.8884 | 0.8116 | 0.7684 | **0.7894** | 0.8759 |
| **Devanagari (`deva`)** | 3,091 | 435 | 0.8915 | 0.8266 | 0.8000 | **0.8131** | 0.9028 |
| **Mixed** | 418 | 59 | 0.9130 | 0.8889 | 0.8136 | **0.8496** | 0.9257 |
| **Unknown** | 190 | 11 | 0.9035 | 0.8182 | 0.8182 | **0.8182** | 0.9092 |

### B. Source Platform Slices
| Platform | Total $N$ | Hate Count | Macro-F1 | Hate-P | Hate-R | Hate-F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YouTube** | 13,040 | 1,071 | 0.8933 | 0.8223 | 0.7862 | **0.8038** | 0.8921 |
| **Reddit** | 434 | 6 | 0.7193 | 0.6667 | 0.3333 | **0.4444** | 0.6968 |
| **Unknown** | 190 | 11 | 0.9035 | 0.8182 | 0.8182 | **0.8182** | 0.9092 |

### C. Text Length Buckets
| Word Count Bucket | Total $N$ | Hate Count | Macro-F1 | Hate-P | Hate-R | Hate-F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1–10 words** | 5,023 | 294 | 0.8834 | 0.7690 | 0.7925 | **0.7806** | 0.8733 |
| **11–25 words** | 5,943 | 538 | 0.9009 | 0.8545 | 0.7862 | **0.8190** | 0.9077 |
| **26–50 words** | 2,074 | 199 | 0.8916 | 0.8407 | 0.7688 | **0.8031** | 0.8878 |
| **>50 words** | 624 | 57 | 0.8707 | 0.7586 | 0.7719 | **0.7652** | 0.8417 |

### D. Presence of Emoji
| Emoji Presence | Total $N$ | Hate Count | Macro-F1 | Hate-P | Hate-R | Hate-F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **With Emoji** | 4,663 | 400 | 0.8923 | 0.8492 | 0.7600 | **0.8021** | 0.8924 |
| **No Emoji** | 9,001 | 688 | 0.8932 | 0.8074 | 0.7980 | **0.8026** | 0.8902 |

---

## 5. Label Quality Audit (100 Random Hate-Labelled Test Rows) & Failure Analysis

A hand-audit of 100 randomly sampled positive test rows (`label = 1`) in [results/audit_100.csv](file:///c:/nlp-moderation/results/audit_100.csv) categorized ground-truth annotations under four mutually exclusive categories:

| Category Tag | Count | Proportion | Wilson 95% CI | Model Flagged at $\tau=0.746$ | Mean Ensemble Prob | Median Ensemble Prob |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **neutral** (no visible hostility under the written guide; 21–42% depending on whether ?/~ rows are counted; 18 of 42 carry ?, 6 carry ~) | 42 | 42.0% | **[0.33, 0.52]** | 29 / 42 (69.0%) | 0.736 | 0.991 |
| **personal_abuse** (interpersonal toxicity) | 22 | 22.0% | **[0.15, 0.31]** | 21 / 22 (95.5%) | 0.936 | 0.991 |
| **political_criticism** (partisan debate) | 19 | 19.0% | **[0.13, 0.28]** | 14 / 19 (73.7%) | 0.832 | 0.989 |
| **group_hate** (strict identity-based hate) | 17 | 17.0% | **[0.11, 0.26]** | 14 / 17 (82.4%) | 0.880 | 0.991 |

- **Audit Marks / Flags:** **? 25**, **~ 7**, **! 4** (definitions: `?` contestable, `~` context could flip it, `!` flagged extreme).
- **Core Findings on Dataset Labels:**
  1. Only **17%** (Wilson 95% CI: [11%, 26%]) of audited positive instances represent strict group-based hate targeting protected characteristics.
  2. The remaining **83%** of the positive class comprises comments with no visible hostility under the written guide (42%), personal abuse / profanity (22%), and partisan political criticism (19%).
  3. The model flagged 95.5% of personal abuse, 82.4% of group hate, 73.7% of political criticism, and 69.0% of neutral rows, reflecting that the positive training label in HINGHATE is significantly broader than strict hate speech.

- **Error Analysis & Failure Modes:**
  1. **Top-50 FP Read Caveat:** The top-50 false positives in [results/errors.csv](file:///c:/nlp-moderation/results/errors.csv) are sorted by model confidence ($\bar{p} > 0.95$ on ground truth `0`). This sample is selected specifically by extreme model certainty and is **not representative** of the overall distribution of false-positive errors.
  2. **False Negatives are Model Weaknesses:** The top false negatives ($\bar{p} < 0.15$ on ground truth `1`) represent genuine model weaknesses:
     - Characters separated by spaces to bypass subword tokenizers (e.g., `l a u d e`, `m u l l a`).
     - Heavily misspelled or phonetically camouflaged slur variants.
     - Sarcastic or implicit abuse lacking overt abusive vocabulary.
  3. **Length Sensitivity:** Short comments ($\le 10$ words) exhibit lower Hate-F1 (0.7806) due to ambiguous insults without sufficient lexical context.

---

## 6. Generated Inspection Datasets
- Top 50 False Positives & Top 50 False Negatives: [results/errors.csv](file:///c:/nlp-moderation/results/errors.csv)
- Manual Audit Sample (100 Random Hate-Labelled Test Rows): [results/audit_100.csv](file:///c:/nlp-moderation/results/audit_100.csv)
