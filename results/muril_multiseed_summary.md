# MuRIL Multi-Seed Performance Summary (Seeds 42, 43, 44)

- **Model:** `google/muril-base-cased` (Sequence Classification, 2 labels)
- **Seeds Evaluated:** 42, 43, 44
- **Decision Threshold:** Tuned per-seed on validation set (maximizing val Macro-F1)
- **Test Set Size:** $N = 13,664$ (Natural 7.96% hate ratio)

## Test Metrics Across Seeds (Mean ± Std)

| Script Stratum | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | 0.8825 ± 0.0016 | 0.7882 ± 0.0117 | 0.7794 ± 0.0179 | **0.7836 ± 0.0034** | 0.8742 ± 0.0027 |
| **roman** | 0.8744 ± 0.0008 | 0.7772 ± 0.0205 | 0.7507 ± 0.0224 | **0.7633 ± 0.0019** | 0.8586 ± 0.0028 |
| **deva** | 0.8864 ± 0.0029 | 0.7952 ± 0.0020 | 0.8153 ± 0.0127 | **0.8051 ± 0.0052** | 0.8884 ± 0.0034 |
| **mixed** | 0.8889 ± 0.0078 | 0.8327 ± 0.0231 | 0.7853 ± 0.0098 | **0.8082 ± 0.0132** | 0.9212 ± 0.0117 |
| **unknown** | 0.9196 ± 0.0279 | 0.8485 ± 0.0525 | 0.8485 ± 0.0525 | **0.8485 ± 0.0525** | 0.9043 ± 0.0240 |

### Per-Seed Hate-Class F1 Breakdown:
- **Seed 42 ($\tau = 0.89$):** Macro-F1 = 0.8823, Hate-F1 = 0.7832, PR-AUC = 0.8759
- **Seed 43 ($\tau = 0.93$):** Macro-F1 = 0.8842, Hate-F1 = 0.7871, PR-AUC = 0.8711
- **Seed 44 ($\tau = 0.96$):** Macro-F1 = 0.8810, Hate-F1 = 0.7804, PR-AUC = 0.8757
