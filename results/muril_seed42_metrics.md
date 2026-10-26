# Stage 4: MuRIL Fine-Tuning Evaluation (Seed 42)

- **Model:** `google/muril-base-cased` (Sequence Classification, 2 labels)
- **Precision:** FP32 exclusively
- **Sequence Length:** `max_len = 64` (Dynamic batch padding)
- **Batching:** Per-device batch 8 $\times$ Gradient Accumulation 4 (Effective batch size = 32)
- **Optimizer:** AdamW (`lr=2e-5`, `weight_decay=0.01`, linear warmup 10%)
- **Loss:** Class-weighted cross entropy based on subsampled train distribution
- **Best Validation Macro-F1 (checkpoint):** `0.8840`

---

## 1. Decision Threshold Selection on Validation Set

The decision threshold was tuned on the validation set ($N = 13,663$) using two distinct criteria:
1. **Max Val Macro-F1:** Threshold $\tau = 0.89$ (Achieved Val Macro-F1 = `0.8902`) — *directly comparable with Stage 3 baseline methodology*.
2. **Max Val Hate-Class F1:** Threshold $\tau = 0.89$ (Achieved Val Hate F1 = `0.7982`) — *prioritizing positive class detection*.

---

## 2. Test Set Evaluation: Threshold Tuned for Max Val Macro-F1 ($\tau = 0.89$)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **13,664** | **0.8823** | **0.7861** | **0.7803** | **0.7832** | **0.8759** |
| **Roman** | 9,965 | 0.8749 | 0.7756 | 0.7530 | 0.7641 | 0.8589 |
| **Devanagari (`deva`)** | 3,091 | 0.8868 | 0.7960 | 0.8161 | 0.8059 | 0.8920 |
| **Mixed** | 418 | 0.8799 | 0.8070 | 0.7797 | 0.7931 | 0.9141 |
| **Unknown** | 190 | 0.9035 | 0.8182 | 0.8182 | 0.8182 | 0.9034 |

---

## 3. Test Set Evaluation: Threshold Tuned for Max Val Hate-Class F1 ($\tau = 0.89$)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **13,664** | **0.8823** | **0.7861** | **0.7803** | **0.7832** | **0.8759** |
| **Roman** | 9,965 | 0.8749 | 0.7756 | 0.7530 | 0.7641 | 0.8589 |
| **Devanagari (`deva`)** | 3,091 | 0.8868 | 0.7960 | 0.8161 | 0.8059 | 0.8920 |
| **Mixed** | 418 | 0.8799 | 0.8070 | 0.7797 | 0.7931 | 0.9141 |
| **Unknown** | 190 | 0.9035 | 0.8182 | 0.8182 | 0.8182 | 0.9034 |
