# Stage 4: MuRIL Fine-Tuning Evaluation (Seed 42)

- **Model:** `google/muril-base-cased` (Sequence Classification, 2 labels)
- **Precision:** FP32 exclusively
- **Sequence Length:** `max_len = 64` (Dynamic batch padding)
- **Batching:** Per-device batch 8 $\times$ Gradient Accumulation 4 (Effective batch size = 32)
- **Optimizer:** AdamW (`lr=2e-5`, `weight_decay=0.01`, linear warmup 10%)
- **Loss:** Class-weighted cross entropy based on subsampled train distribution
- **Best Validation Macro-F1 (checkpoint):** `0.8804`

---

## 1. Decision Threshold Selection on Validation Set

The decision threshold was tuned on the validation set ($N = 13,663$) using two distinct criteria:
1. **Max Val Macro-F1:** Threshold $\tau = 0.90$ (Achieved Val Macro-F1 = `0.8846`) — *directly comparable with Stage 3 baseline methodology*.
2. **Max Val Hate-Class F1:** Threshold $\tau = 0.89$ (Achieved Val Hate F1 = `0.7887`) — *prioritizing positive class detection*.

---

## 2. Test Set Evaluation: Threshold Tuned for Max Val Macro-F1 ($\tau = 0.90$)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **13,664** | **0.8842** | **0.7661** | **0.8097** | **0.7873** | **0.8711** |
| **Roman** | 9,965 | 0.8753 | 0.7451 | 0.7873 | 0.7656 | 0.8556 |
| **Devanagari (`deva`)** | 3,091 | 0.8889 | 0.7833 | 0.8391 | 0.8102 | 0.8854 |
| **Mixed** | 418 | 0.8899 | 0.8246 | 0.7966 | 0.8103 | 0.9148 |
| **Unknown** | 190 | 0.9518 | 0.9091 | 0.9091 | 0.9091 | 0.9288 |

---

## 3. Test Set Evaluation: Threshold Tuned for Max Val Hate-Class F1 ($\tau = 0.89$)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **13,664** | **0.8839** | **0.7612** | **0.8143** | **0.7869** | **0.8711** |
| **Roman** | 9,965 | 0.8761 | 0.7420 | 0.7942 | 0.7672 | 0.8556 |
| **Devanagari (`deva`)** | 3,091 | 0.8875 | 0.7771 | 0.8414 | 0.8079 | 0.8854 |
| **Mixed** | 418 | 0.8899 | 0.8246 | 0.7966 | 0.8103 | 0.9148 |
| **Unknown** | 190 | 0.9306 | 0.8333 | 0.9091 | 0.8696 | 0.9288 |
