# Stage 3 Baseline: TF-IDF + Logistic Regression

- **Model:** Logistic Regression (`class_weight='balanced'`, `C=10.0`)
- **Features:** TF-IDF (char n-grams 2-5 + word n-grams 1-2, sublinear tf)
- **Tuned Decision Threshold (Validation):** `0.73`
- **Evaluation Split:** Test set ($N = 13,664$, natural hate ratio: 7.96%)

## Test Metrics Summary (Headline Metrics: Macro-F1 & Hate F1)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **13,664** | **0.7385** | **0.4701** | **0.5919** | **0.5240** | **0.5427** |
| **Roman** | 9,965 | 0.7500 | 0.4892 | 0.5832 | 0.5321 | 0.5338 |
| **Devanagari (`deva`)** | 3,091 | 0.7011 | 0.4361 | 0.5885 | 0.5010 | 0.5534 |
| **Mixed** | 418 | 0.8029 | 0.6143 | 0.7288 | 0.6667 | 0.6855 |
| **Unknown** | 190 | 0.6453 | 0.2778 | 0.4545 | 0.3448 | 0.4956 |

### Observations:
1. **Script Disparity:** Hate-class F1 on Roman Hinglish vs Devanagari demonstrates baseline performance across scripts prior to transformer fine-tuning.
2. **Precision-Recall Balance:** At threshold `0.73`, the balanced model maintains high recall while moderating false alarms on the imbalanced test set.
