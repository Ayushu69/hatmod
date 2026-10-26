# Stage 4: MuRIL Fine-Tuning Evaluation (Seed 42)

- **Model:** `google/muril-base-cased` (Sequence Classification, 2 labels)
- **Precision:** FP32 exclusively
- **Sequence Length:** `max_len = 64` (Dynamic batch padding)
- **Batching:** Per-device batch 8 $\times$ Gradient Accumulation 4 (Effective batch size = 32)
- **Optimizer:** AdamW (`lr=2e-5`, `weight_decay=0.01`, linear warmup 10%)
- **Loss:** Class-weighted cross entropy based on subsampled train distribution
- **Best Validation Macro-F1 (checkpoint):** `0.8851`

---

## 1. Decision Threshold Selection on Validation Set

The decision threshold was tuned on the validation set ($N = 13,663$) using two distinct criteria:
1. **Max Val Macro-F1:** Threshold $\tau = 0.84$ (Achieved Val Macro-F1 = `0.8886`) — *directly comparable with Stage 3 baseline methodology*.
2. **Max Val Hate-Class F1:** Threshold $\tau = 0.84$ (Achieved Val Hate F1 = `0.7960`) — *prioritizing positive class detection*.

---

## 2. Test Set Evaluation: Threshold Tuned for Max Val Macro-F1 ($\tau = 0.84$)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **13,664** | **0.8865** | **0.7646** | **0.8208** | **0.7917** | **0.8757** |
| **Roman** | 9,965 | 0.8835 | 0.7618 | 0.8010 | 0.7809 | 0.8613 |
| **Devanagari (`deva`)** | 3,091 | 0.8821 | 0.7609 | 0.8414 | 0.7991 | 0.8879 |
| **Mixed** | 418 | 0.9082 | 0.8226 | 0.8644 | 0.8430 | 0.9346 |
| **Unknown** | 190 | 0.8843 | 0.7500 | 0.8182 | 0.7826 | 0.8808 |

---

## 3. Test Set Evaluation: Threshold Tuned for Max Val Hate-Class F1 ($\tau = 0.84$)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **13,664** | **0.8865** | **0.7646** | **0.8208** | **0.7917** | **0.8757** |
| **Roman** | 9,965 | 0.8835 | 0.7618 | 0.8010 | 0.7809 | 0.8613 |
| **Devanagari (`deva`)** | 3,091 | 0.8821 | 0.7609 | 0.8414 | 0.7991 | 0.8879 |
| **Mixed** | 418 | 0.9082 | 0.8226 | 0.8644 | 0.8430 | 0.9346 |
| **Unknown** | 190 | 0.8843 | 0.7500 | 0.8182 | 0.7826 | 0.8808 |
