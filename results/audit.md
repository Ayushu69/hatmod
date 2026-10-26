# Stage 1: Hinglish Hate-Speech Corpus Audit

## 1. Corpus Dimensions & Integrity
- **Raw Shape:** 248,129 rows x 4 columns
- **Columns:** `text, label, source, script`
- **Null Texts:** 0
- **Empty String Texts (`""`):** 0
- **Unique Texts:** 136,680
- **Exact Duplicate Text Rows:** 111,449 (44.92%)
- **Conflicting Labels on Duplicate Texts:** 0 (Zero label ambiguity on exact text matches)

## 2. Label Counts & Ratios

| Split / Stage | Non-Hate (`0`) | Hate (`1`) | Total | Hate Ratio (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Corpus** | 229,462 | 18,667 | 248,129 | 7.52% |
| **Post-Dedupe** | 125,802 | 10,878 | 136,680 | 7.96% |

- **Exact Duplicate Breakdown:**
  - Non-hate duplicates removed: 103,660 rows
  - Hate duplicates removed: 7,789 rows

## 3. Script Confound Analysis

| Script Category | Raw Total | Raw Hate % | Post-Dedupe Total | Post-Dedupe Hate % |
| :--- | :--- | :--- | :--- | :--- |
| **Roman** | 185,082 | 5.44% | 99,680 | 5.84% |
| **Devanagari (`deva`)** | 51,374 | 14.28% | 30,918 | 14.07% |
| **Mixed** | 7,358 | 13.86% | 4,186 | 14.24% |
| **Blank (`""`)** | 4,315 | 5.77% | 1,896 | 5.75% |

> **Key Observation:** The hate rate in Devanagari (~14.07%) and Mixed (~14.24%) is ~2.4x higher than in Roman (~5.84%). Script is an empirical confound that requires stratified splitting.

## 4. Source Platform Breakdown

| Platform Source | Raw Total | Raw Hate % | Post-Dedupe Total | Post-Dedupe Hate Count | Post-Dedupe Hate % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YouTube** | 233,425 | 7.85% | 130,303 | 10,724 | 8.23% |
| **Reddit** | 10,389 | 0.92% | 4,481 | 41 | 0.91% |
| **Blank (`""`)** | 4,315 | 5.77% | 1,896 | 110 | 5.75% |

> **Key Observation:** Reddit has only 41 unique hate examples out of 4,481 total deduped posts (<1% hate rate). YouTube comprises the vast majority of the signal.

## 5. Word Length Percentiles & Sequence Length Coverage

| Metric | Raw Corpus | Post-Dedupe Corpus |
| :--- | :--- | :--- |
| **Min** | 1 | 1 |
| **25th Percentile** | 9 | 8 |
| **Median (50th)** | 14 | 13 |
| **75th Percentile** | 23 | 22 |
| **90th Percentile** | 38 | 36 |
| **95th Percentile** | 50 | 49 |
| **99th Percentile** | 78 | 78 |
| **Max** | 5250 | 5250 |
| **Mean ± Std** | 18.76 ± 23.50 | 18.18 ± 28.44 |
| **Share > 64 Words** | 2.35% | 2.18% |

> **Implication for Model Configuration:** At `max_len = 64`, ~97.8% of post-deduped texts are captured without word truncation, ensuring high contextual coverage within the 6 GB VRAM budget.

## 6. Text Artifacts & Special Character Audit
- **Zero-Width Space (`\u200b`):** 9,893 occurrences
- **Zero-Width Non-Joiner (`\u200c`):** 165 occurrences
- **Zero-Width Joiner (`\u200d`):** 421 occurrences
- **`<USER>` Placeholders:** 17,015 occurrences
- **HTTP / HTTPS URLs:** 0 (clean from web links)

## 7. Next Steps for Stage 2
1. Clean zero-width characters while preserving Devanagari matras and script integrity.
2. Collapse whitespace and repeated characters (max 2).
3. Treat blank `source` and `script` as `"unknown"`.
4. Perform normalized-key deduplication and verify cross-split leakage.
5. Stratified 80/10/10 split on `(label, script)` with seed 42.
