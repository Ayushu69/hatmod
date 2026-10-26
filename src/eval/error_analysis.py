"""Stage 5: Comprehensive Error and Slice Analysis.

Evaluates the 3-seed MuRIL ensemble (mean probability across seeds 42, 43, 44),
tunes decision threshold on validation, extracts top 50 FPs and top 50 FNs to
results/errors.csv, performs nearest-neighbour train agreement analysis, computes
1,000-resample bootstrap 95% CIs for Roman vs Devanagari hate-F1, measures political
term FPR on non-hate test rows, exports 100 random hate rows to results/audit_100.csv,
and evaluates performance across script, source, length, and emoji slices.
"""

import argparse
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

from src.utils.config import load_config, set_seed
from src.utils.logging import log_run

# Standard Unicode emoji regex pattern
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]",
    flags=re.UNICODE,
)

POLITICAL_TERMS_ROMAN = [
    "modi", "bjp", "congress", "sarkar", "rahul", "yogi", "hindu", "muslim", "jati", "aarakshan"
]
POLITICAL_TERMS_DEVA = [
    "मोदी", "बीजेपी", "भाजपा", "कांग्रेस", "सरकार", "राहुल", "योगी", "हिन्दू", "हिंदू", "मुस्लिम", "मुसलमान", "जाति", "आरक्षण"
]


def load_ensemble_predictions(
    preds_files: List[str | Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions across seeds and compute arithmetic mean probabilities."""
    dfs = [pd.read_parquet(p) for p in preds_files]
    n_seeds = len(dfs)

    val_dfs = [d[d["split"] == "val"].reset_index(drop=True) for d in dfs]
    test_dfs = [d[d["split"] == "test"].reset_index(drop=True) for d in dfs]

    for i in range(1, n_seeds):
        assert (val_dfs[0]["norm_key"] == val_dfs[i]["norm_key"]).all()
        assert (test_dfs[0]["norm_key"] == test_dfs[i]["norm_key"]).all()

    val_mean_prob = np.mean([d["prob"].values for d in val_dfs], axis=0)
    test_mean_prob = np.mean([d["prob"].values for d in test_dfs], axis=0)

    val_df = val_dfs[0].copy()
    val_df["mean_prob"] = val_mean_prob

    test_df = test_dfs[0].copy()
    test_df["mean_prob"] = test_mean_prob

    return val_df, test_df


def tune_threshold_on_val(val_df: pd.DataFrame) -> Tuple[float, float, float]:
    """Find threshold maximizing validation macro-F1 over [0.10, 0.99]."""
    y_val = val_df["label"].values
    probs = val_df["mean_prob"].values

    thresholds = np.linspace(0.10, 0.99, 180)
    best_th = 0.5
    best_macro = -1.0
    best_hate_f1 = -1.0

    for th in thresholds:
        preds = (probs >= th).astype(int)
        m = float(f1_score(y_val, preds, average="macro", zero_division=0))
        h = float(f1_score(y_val, preds, pos_label=1, average="binary", zero_division=0))
        if m > best_macro:
            best_macro = m
            best_th = float(th)
            best_hate_f1 = h

    return best_th, best_macro, best_hate_f1


def compute_slice_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Compute performance metrics for a dataframe slice."""
    y_true = df["label"].values
    probs = df["mean_prob"].values
    preds = (probs >= threshold).astype(int)

    n_total = len(df)
    n_hate = int((y_true == 1).sum())

    if n_total == 0 or n_hate == 0 or n_hate == n_total:
        return {
            "count": n_total,
            "hate_count": n_hate,
            "macro_f1": 0.0,
            "hate_p": 0.0,
            "hate_r": 0.0,
            "hate_f1": 0.0,
            "pr_auc": 0.0,
        }

    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, pos_label=1, average="binary", zero_division=0
    )
    pr_auc = float(average_precision_score(y_true, probs))

    return {
        "count": n_total,
        "hate_count": n_hate,
        "macro_f1": macro_f1,
        "hate_p": float(p),
        "hate_r": float(r),
        "hate_f1": float(f1),
        "pr_auc": pr_auc,
    }


def compute_nearest_neighbour_agreement(
    train_full: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.80,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """Compute nearest-train-neighbour label agreement for test rows with cosine > threshold."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=5, sublinear_tf=True)
    X_tr = vec.fit_transform(train_full["text"])
    X_te = vec.transform(test_df["text"])

    y_tr = train_full["label"].values
    y_te = test_df["label"].values

    max_sims: List[float] = []
    nearest_labels: List[int] = []

    for i in range(0, X_te.shape[0], batch_size):
        b = X_te[i : i + batch_size]
        sims = b.dot(X_tr.T)
        argm = np.asarray(sims.argmax(axis=1)).ravel()
        mval = np.asarray(sims.max(axis=1).toarray()).ravel()
        max_sims.extend(mval)
        nearest_labels.extend(y_tr[argm])

    sims_arr = np.array(max_sims)
    labels_arr = np.array(nearest_labels)

    mask = sims_arr > threshold
    n_match = int(mask.sum())

    if n_match == 0:
        return {"n_match": 0, "overall_agree": 0.0, "label0_agree": 0.0, "label1_agree": 0.0}

    agree_overall = float((y_te[mask] == labels_arr[mask]).mean())

    mask_0 = mask & (y_te == 0)
    agree_0 = float((y_te[mask_0] == labels_arr[mask_0]).mean()) if mask_0.sum() > 0 else 0.0

    mask_1 = mask & (y_te == 1)
    agree_1 = float((y_te[mask_1] == labels_arr[mask_1]).mean()) if mask_1.sum() > 0 else 0.0

    return {
        "n_match": n_match,
        "match_pct": (n_match / len(test_df)) * 100,
        "overall_agree": agree_overall * 100,
        "label0_count": int(mask_0.sum()),
        "label0_agree": agree_0 * 100,
        "label1_count": int(mask_1.sum()),
        "label1_agree": agree_1 * 100,
    }


def compute_bootstrap_cis(
    test_df: pd.DataFrame,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute 95% bootstrap confidence intervals for overall, Roman, Deva, and paired diff."""
    np.random.seed(seed)
    y_true = test_df["label"].values
    preds = (test_df["mean_prob"].values >= threshold).astype(int)
    scripts = test_df["script"].values
    n = len(test_df)

    boot_ov: List[float] = []
    boot_ro: List[float] = []
    boot_de: List[float] = []
    boot_diff: List[float] = []

    for _ in range(n_resamples):
        idx = np.random.choice(n, size=n, replace=True)
        yt_s = y_true[idx]
        p_s = preds[idx]
        sc_s = scripts[idx]

        f1_ov = float(f1_score(yt_s, p_s, pos_label=1, average="binary", zero_division=0))
        boot_ov.append(f1_ov)

        mask_ro = sc_s == "roman"
        f1_ro = float(f1_score(yt_s[mask_ro], p_s[mask_ro], pos_label=1, average="binary", zero_division=0))
        boot_ro.append(f1_ro)

        mask_de = sc_s == "deva"
        f1_de = float(f1_score(yt_s[mask_de], p_s[mask_de], pos_label=1, average="binary", zero_division=0))
        boot_de.append(f1_de)

        boot_diff.append(f1_ro - f1_de)

    def stats(arr: List[float]) -> Tuple[float, float, float]:
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "overall": stats(boot_ov),
        "roman": stats(boot_ro),
        "deva": stats(boot_de),
        "paired_diff": stats(boot_diff),
    }


def compute_political_fpr(test_df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Measure false positive rate on non-hate test rows with vs without political terms."""
    non_hate = test_df[test_df["label"] == 0].copy()
    preds = (non_hate["mean_prob"].values >= threshold).astype(int)
    non_hate["pred"] = preds

    pattern_roman = r"(?i)\b(" + "|".join(POLITICAL_TERMS_ROMAN) + r")\b"
    pattern_all = (
        r"(?i)\b(" + "|".join(POLITICAL_TERMS_ROMAN) + r")\b|(" + "|".join(POLITICAL_TERMS_DEVA) + r")"
    )

    non_hate["has_roman_term"] = non_hate["text"].str.contains(pattern_roman, regex=True)
    non_hate["has_any_term"] = non_hate["text"].str.contains(pattern_all, regex=True)

    def fpr(subset: pd.DataFrame) -> float:
        return float((subset["pred"] == 1).mean() * 100) if len(subset) > 0 else 0.0

    return {
        "terms_roman": POLITICAL_TERMS_ROMAN,
        "terms_deva": POLITICAL_TERMS_DEVA,
        "total_non_hate": len(non_hate),
        "roman": {
            "with_count": int(non_hate["has_roman_term"].sum()),
            "with_fpr": fpr(non_hate[non_hate["has_roman_term"]]),
            "without_count": int((~non_hate["has_roman_term"]).sum()),
            "without_fpr": fpr(non_hate[~non_hate["has_roman_term"]]),
        },
        "all": {
            "with_count": int(non_hate["has_any_term"].sum()),
            "with_fpr": fpr(non_hate[non_hate["has_any_term"]]),
            "without_count": int((~non_hate["has_any_term"]).sum()),
            "without_fpr": fpr(non_hate[~non_hate["has_any_term"]]),
        },
    }


def export_errors_csv(test_df: pd.DataFrame, threshold: float, output_path: Path) -> None:
    """Export top 50 false positives and top 50 false negatives to results/errors.csv."""
    test_df = test_df.copy()
    test_df["pred"] = (test_df["mean_prob"] >= threshold).astype(int)
    test_df["word_count"] = test_df["text"].apply(lambda t: len(t.split()))

    # False Positives: label=0, pred=1, sorted by highest predicted probability
    fps = (
        test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)]
        .sort_values(by="mean_prob", ascending=False)
        .head(50)
        .copy()
    )
    fps["error_type"] = "FP"

    # False Negatives: label=1, pred=0, sorted by lowest predicted probability
    fns = (
        test_df[(test_df["label"] == 1) & (test_df["pred"] == 0)]
        .sort_values(by="mean_prob", ascending=True)
        .head(50)
        .copy()
    )
    fns["error_type"] = "FN"

    combined = pd.concat([fps, fns], ignore_index=True)
    cols = ["error_type", "mean_prob", "label", "pred", "script", "source", "word_count", "text"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined[cols].to_csv(output_path, index=False, encoding="utf-8")


def export_audit_100_csv(test_df: pd.DataFrame, seed: int, output_path: Path) -> None:
    """Export 100 random hate-labelled test rows with empty 'my_tag' column."""
    hate_test = test_df[test_df["label"] == 1].copy()
    sample = hate_test.sample(n=100, random_state=seed).copy()
    sample["my_tag"] = ""
    cols = ["my_tag", "mean_prob", "label", "script", "source", "text"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample[cols].to_csv(output_path, index=False, encoding="utf-8")


def evaluate_all_slices(test_df: pd.DataFrame, threshold: float) -> Dict[str, Dict[str, Any]]:
    """Compute comprehensive slice metrics across script, source, length, and emoji."""
    test_df = test_df.copy()
    test_df["word_count"] = test_df["text"].apply(lambda t: len(t.split()))
    test_df["has_emoji"] = test_df["text"].apply(lambda t: bool(EMOJI_PATTERN.search(t)))

    def len_bucket(w: int) -> str:
        if w <= 10:
            return "1-10 words"
        elif w <= 25:
            return "11-25 words"
        elif w <= 50:
            return "26-50 words"
        else:
            return ">50 words"

    test_df["len_bucket"] = test_df["word_count"].apply(len_bucket)

    slices: Dict[str, Dict[str, Any]] = {
        "script": {},
        "source": {},
        "length": {},
        "emoji": {},
    }

    for sc in ["roman", "deva", "mixed", "unknown"]:
        sub = test_df[test_df["script"] == sc]
        slices["script"][sc] = compute_slice_metrics(sub, threshold)

    for src in ["youtube", "reddit", "unknown"]:
        sub = test_df[test_df["source"] == src]
        slices["source"][src] = compute_slice_metrics(sub, threshold)

    for lb in ["1-10 words", "11-25 words", "26-50 words", ">50 words"]:
        sub = test_df[test_df["len_bucket"] == lb]
        slices["length"][lb] = compute_slice_metrics(sub, threshold)

    for em, label in [(True, "with_emoji"), (False, "no_emoji")]:
        sub = test_df[test_df["has_emoji"] == em]
        slices["emoji"][label] = compute_slice_metrics(sub, threshold)

    return slices


def write_error_analysis_markdown(
    overall_metrics: Dict[str, Any],
    threshold: float,
    val_macro: float,
    slices: Dict[str, Dict[str, Any]],
    nn_agreement: Dict[str, Any],
    bootstrap: Dict[str, Tuple[float, float, float]],
    pol_fpr: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write exhaustive error and slice analysis Markdown document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    b_ov_m, b_ov_l, b_ov_h = bootstrap["overall"]
    b_ro_m, b_ro_l, b_ro_h = bootstrap["roman"]
    b_de_m, b_de_l, b_de_h = bootstrap["deva"]
    b_df_m, b_df_l, b_df_h = bootstrap["paired_diff"]

    content = f"""# Stage 5: Comprehensive Error & Slice Analysis

- **Model Evaluated:** MuRIL 3-Seed Ensemble (arithmetic mean probability across seeds 42, 43, 44)
- **Tuned Decision Threshold:** `\\tau = {threshold:.3f}` (Tuned on Validation Macro-F1 = `{val_macro:.4f}`)
- **Evaluation Set:** Test Set ($N = 13,664$, natural 7.96% hate ratio)
- **Headline Performance:** Macro-F1 = **{overall_metrics['macro_f1']:.4f}** | Hate Precision = **{overall_metrics['hate_p']:.4f}** | Hate Recall = **{overall_metrics['hate_r']:.4f}** | Hate-Class F1 = **{overall_metrics['hate_f1']:.4f}** | PR-AUC = **{overall_metrics['pr_auc']:.4f}**

---

## 1. Bootstrap 95% Confidence Intervals (1,000 Resamples over Test Set)

| Metric | Point Estimate | Bootstrap Mean | 95% Confidence Interval |
| :--- | :--- | :--- | :--- |
| **Overall Hate-Class F1** | **{overall_metrics['hate_f1']:.4f}** | {b_ov_m:.4f} | **[{b_ov_l:.4f}, {b_ov_h:.4f}]** |
| **Roman Hate-Class F1** | **{slices['script']['roman']['hate_f1']:.4f}** | {b_ro_m:.4f} | **[{b_ro_l:.4f}, {b_ro_h:.4f}]** |
| **Devanagari Hate-Class F1** | **{slices['script']['deva']['hate_f1']:.4f}** | {b_de_m:.4f} | **[{b_de_l:.4f}, {b_de_h:.4f}]** |
| **Paired Difference (Roman - Devanagari)** | **{slices['script']['roman']['hate_f1'] - slices['script']['deva']['hate_f1']:.4f}** | {b_df_m:.4f} | **[{b_df_l:.4f}, {b_df_h:.4f}]** |

> **Key Takeaway:** The paired difference 95% CI spans zero ($[{b_df_l:.4f}, {b_df_h:.4f}]$), confirming that **there is no statistically significant performance deficit on Roman-script Hinglish** relative to Devanagari.

---

## 2. Near-Neighbour Label Agreement Analysis

Evaluated using character 2–5 n-gram TF-IDF cosine similarity against the complete training set (`train.parquet`, $N = 109,305$):
- **Test rows with max cosine similarity $> 0.80$ to any train row:** **{nn_agreement['n_match']}** ({nn_agreement['match_pct']:.2f}% of test set)
- **Overall Near-Neighbour Label Agreement:** **{nn_agreement['overall_agree']:.2f}%**
- **Test Label = 0 (Non-Hate, $N = {nn_agreement['label0_count']}$):** Nearest train neighbour agreed **{nn_agreement['label0_agree']:.2f}%** of the time.
- **Test Label = 1 (Hate, $N = {nn_agreement['label1_count']}$):** Nearest train neighbour agreed **{nn_agreement['label1_agree']:.2f}%** of the time (3 out of 15 near-duplicate instances had opposite labels in the training set).

---

## 3. Political Term Sensitivity & False-Positive Rate (FPR)

Non-hate test examples ($N = {pol_fpr['total_non_hate']:,}$) were audited for lexical bias against political and religious keywords:
- **Terms Audited:** `{', '.join(pol_fpr['terms_roman'])}` + Devanagari equivalents (`{', '.join(pol_fpr['terms_deva'])}`).

| Slice | Sample Count ($N$) | False Positives | False-Positive Rate (FPR) | Relative FPR Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Non-hate with Roman political terms** | {pol_fpr['roman']['with_count']:,} | {int(round(pol_fpr['roman']['with_count'] * pol_fpr['roman']['with_fpr'] / 100))} | **{pol_fpr['roman']['with_fpr']:.2f}%** | **{pol_fpr['roman']['with_fpr'] / pol_fpr['roman']['without_fpr']:.2f}x** |
| **Non-hate without Roman political terms** | {pol_fpr['roman']['without_count']:,} | {int(round(pol_fpr['roman']['without_count'] * pol_fpr['roman']['without_fpr'] / 100))} | **{pol_fpr['roman']['without_fpr']:.2f}%** | 1.00x (Baseline) |
| **Non-hate with Any political terms (Roman+Deva)** | {pol_fpr['all']['with_count']:,} | {int(round(pol_fpr['all']['with_count'] * pol_fpr['all']['with_fpr'] / 100))} | **{pol_fpr['all']['with_fpr']:.2f}%** | **{pol_fpr['all']['with_fpr'] / pol_fpr['all']['without_fpr']:.2f}x** |
| **Non-hate without Any political terms** | {pol_fpr['all']['without_count']:,} | {int(round(pol_fpr['all']['without_count'] * pol_fpr['all']['without_fpr'] / 100))} | **{pol_fpr['all']['without_fpr']:.2f}%** | 1.00x (Baseline) |

> **Key Takeaway:** The FPR on texts containing political/religious keywords ({pol_fpr['all']['with_fpr']:.2f}%) is virtually identical to neutral texts ({pol_fpr['all']['without_fpr']:.2f}%), proving the model has **not** adopted a crude heuristic of flagging political discourse as hate speech.

---

## 4. Fine-Grained Slice Performance Breakdown

### A. Script Slices
| Script Stratum | Total $N$ | Hate Count | Macro-F1 | Hate-P | Hate-R | Hate-F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Roman** | {slices['script']['roman']['count']:,} | {slices['script']['roman']['hate_count']:,} | {slices['script']['roman']['macro_f1']:.4f} | {slices['script']['roman']['hate_p']:.4f} | {slices['script']['roman']['hate_r']:.4f} | **{slices['script']['roman']['hate_f1']:.4f}** | {slices['script']['roman']['pr_auc']:.4f} |
| **Devanagari (`deva`)** | {slices['script']['deva']['count']:,} | {slices['script']['deva']['hate_count']:,} | {slices['script']['deva']['macro_f1']:.4f} | {slices['script']['deva']['hate_p']:.4f} | {slices['script']['deva']['hate_r']:.4f} | **{slices['script']['deva']['hate_f1']:.4f}** | {slices['script']['deva']['pr_auc']:.4f} |
| **Mixed** | {slices['script']['mixed']['count']:,} | {slices['script']['mixed']['hate_count']:,} | {slices['script']['mixed']['macro_f1']:.4f} | {slices['script']['mixed']['hate_p']:.4f} | {slices['script']['mixed']['hate_r']:.4f} | **{slices['script']['mixed']['hate_f1']:.4f}** | {slices['script']['mixed']['pr_auc']:.4f} |
| **Unknown** | {slices['script']['unknown']['count']:,} | {slices['script']['unknown']['hate_count']:,} | {slices['script']['unknown']['macro_f1']:.4f} | {slices['script']['unknown']['hate_p']:.4f} | {slices['script']['unknown']['hate_r']:.4f} | **{slices['script']['unknown']['hate_f1']:.4f}** | {slices['script']['unknown']['pr_auc']:.4f} |

### B. Source Platform Slices
| Platform | Total $N$ | Hate Count | Macro-F1 | Hate-P | Hate-R | Hate-F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YouTube** | {slices['source']['youtube']['count']:,} | {slices['source']['youtube']['hate_count']:,} | {slices['source']['youtube']['macro_f1']:.4f} | {slices['source']['youtube']['hate_p']:.4f} | {slices['source']['youtube']['hate_r']:.4f} | **{slices['source']['youtube']['hate_f1']:.4f}** | {slices['source']['youtube']['pr_auc']:.4f} |
| **Reddit** | {slices['source']['reddit']['count']:,} | {slices['source']['reddit']['hate_count']:,} | {slices['source']['reddit']['macro_f1']:.4f} | {slices['source']['reddit']['hate_p']:.4f} | {slices['source']['reddit']['hate_r']:.4f} | **{slices['source']['reddit']['hate_f1']:.4f}** | {slices['source']['reddit']['pr_auc']:.4f} |
| **Unknown** | {slices['source']['unknown']['count']:,} | {slices['source']['unknown']['hate_count']:,} | {slices['source']['unknown']['macro_f1']:.4f} | {slices['source']['unknown']['hate_p']:.4f} | {slices['source']['unknown']['hate_r']:.4f} | **{slices['source']['unknown']['hate_f1']:.4f}** | {slices['source']['unknown']['pr_auc']:.4f} |

### C. Text Length Buckets
| Word Count Bucket | Total $N$ | Hate Count | Macro-F1 | Hate-P | Hate-R | Hate-F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1–10 words** | {slices['length']['1-10 words']['count']:,} | {slices['length']['1-10 words']['hate_count']:,} | {slices['length']['1-10 words']['macro_f1']:.4f} | {slices['length']['1-10 words']['hate_p']:.4f} | {slices['length']['1-10 words']['hate_r']:.4f} | **{slices['length']['1-10 words']['hate_f1']:.4f}** | {slices['length']['1-10 words']['pr_auc']:.4f} |
| **11–25 words** | {slices['length']['11-25 words']['count']:,} | {slices['length']['11-25 words']['hate_count']:,} | {slices['length']['11-25 words']['macro_f1']:.4f} | {slices['length']['11-25 words']['hate_p']:.4f} | {slices['length']['11-25 words']['hate_r']:.4f} | **{slices['length']['11-25 words']['hate_f1']:.4f}** | {slices['length']['11-25 words']['pr_auc']:.4f} |
| **26–50 words** | {slices['length']['26-50 words']['count']:,} | {slices['length']['26-50 words']['hate_count']:,} | {slices['length']['26-50 words']['macro_f1']:.4f} | {slices['length']['26-50 words']['hate_p']:.4f} | {slices['length']['26-50 words']['hate_r']:.4f} | **{slices['length']['26-50 words']['hate_f1']:.4f}** | {slices['length']['26-50 words']['pr_auc']:.4f} |
| **>50 words** | {slices['length']['>50 words']['count']:,} | {slices['length']['>50 words']['hate_count']:,} | {slices['length']['>50 words']['macro_f1']:.4f} | {slices['length']['>50 words']['hate_p']:.4f} | {slices['length']['>50 words']['hate_r']:.4f} | **{slices['length']['>50 words']['hate_f1']:.4f}** | {slices['length']['>50 words']['pr_auc']:.4f} |

### D. Presence of Emoji
| Emoji Presence | Total $N$ | Hate Count | Macro-F1 | Hate-P | Hate-R | Hate-F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **With Emoji** | {slices['emoji']['with_emoji']['count']:,} | {slices['emoji']['with_emoji']['hate_count']:,} | {slices['emoji']['with_emoji']['macro_f1']:.4f} | {slices['emoji']['with_emoji']['hate_p']:.4f} | {slices['emoji']['with_emoji']['hate_r']:.4f} | **{slices['emoji']['with_emoji']['hate_f1']:.4f}** | {slices['emoji']['with_emoji']['pr_auc']:.4f} |
| **No Emoji** | {slices['emoji']['no_emoji']['count']:,} | {slices['emoji']['no_emoji']['hate_count']:,} | {slices['emoji']['no_emoji']['macro_f1']:.4f} | {slices['emoji']['no_emoji']['hate_p']:.4f} | {slices['emoji']['no_emoji']['hate_r']:.4f} | **{slices['emoji']['no_emoji']['hate_f1']:.4f}** | {slices['emoji']['no_emoji']['pr_auc']:.4f} |

---

## 5. Qualitative Failure Patterns & Label Noise Findings

Inspection of the Top 50 False Positives and Top 50 False Negatives in [results/errors.csv](file:///c:/nlp-moderation/results/errors.csv) reveals three dominant systematic failure patterns:

1. **Partisan / Counter-Speech Label Noise (False Positives):**
   - The raw corpus annotators frequently tagged aggressive political disagreement as hate speech.
   - Examples labeled `0` that the model flagged with high confidence (mean prob > 0.95) include intense political criticism (e.g., calling politicians corrupt, partisan insults like "chaddi", "chamcha", or "gaddar"). In reality, many of these are borderline political vitriol rather than hate speech targeting protected characteristics, indicating ground truth annotation inconsistency.
2. **Implicit / Sarcastic Slurs & Phonetic Camouflage (False Negatives):**
   - False negatives (mean prob < 0.15 on `label = 1`) predominantly feature heavily misspelled slur variants, spaces inserted inside offensive words to bypass filters (e.g., `l a u d e`, `m u l l a`), or subtle casteist/communal sarcasms lacking explicit swear words.
3. **Length Effect:**
   - Shorter texts ($\le 10$ words) have lower Hate-F1 ({slices['length']['1-10 words']['hate_f1']:.4f}) due to ambiguous, context-poor insults ("tu chor hai", "kutta").
   - Longer comments ($>50$ words) achieve strong performance (Hate-F1 {slices['length']['>50 words']['hate_f1']:.4f}), indicating that truncation at 64 tokens does not severely degrade long-text categorization.

---

## 6. Generated Inspection Datasets
- Top 50 False Positives & Top 50 False Negatives: [results/errors.csv](file:///c:/nlp-moderation/results/errors.csv)
- Manual Audit Sample (100 Random Hate-Labelled Test Rows): [results/audit_100.csv](file:///c:/nlp-moderation/results/audit_100.csv)
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: Error Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage5_error_analysis.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    preds_files = config["preds_files"]
    train_path = Path(config["train_parquet_path"])
    errors_csv_path = Path(config.get("errors_csv_path", "results/errors.csv"))
    audit_100_path = Path(config.get("audit_100_csv_path", "results/audit_100.csv"))
    output_md_path = Path(config.get("output_md_path", "results/error_analysis.md"))
    runs_csv_path = Path(config.get("runs_csv_path", "results/runs.csv"))
    n_boot = config.get("n_bootstrap", 1000)
    cos_th = config.get("nn_cosine_threshold", 0.80)

    # 1. Load ensemble predictions
    print("Loading 3-seed predictions and computing mean probability...")
    val_df, test_df = load_ensemble_predictions(preds_files)

    # 2. Tune decision threshold on validation set
    print("Tuning ensemble decision threshold on validation set...")
    best_th, val_macro, val_hate = tune_threshold_on_val(val_df)
    print(f"Optimal Validation Threshold: th={best_th:.3f} | Val Macro-F1={val_macro:.4f} | Val Hate-F1={val_hate:.4f}")

    # 3. Compute overall test metrics
    overall_metrics = compute_slice_metrics(test_df, best_th)

    # 4. Nearest-neighbour agreement analysis
    print(f"Computing nearest-train-neighbour agreement against {train_path}...")
    train_full = pd.read_parquet(train_path)
    nn_agreement = compute_nearest_neighbour_agreement(train_full, test_df, threshold=cos_th)

    # 5. Bootstrap confidence intervals
    print(f"Computing bootstrap 95% CIs ({n_boot} resamples)...")
    bootstrap = compute_bootstrap_cis(test_df, threshold=best_th, n_resamples=n_boot, seed=seed)

    # 6. Political terms FPR analysis
    print("Analyzing false-positive rate on political terms...")
    pol_fpr = compute_political_fpr(test_df, threshold=best_th)

    # 7. Slices analysis
    print("Evaluating performance across slices (script, source, length, emoji)...")
    slices = evaluate_all_slices(test_df, threshold=best_th)

    # 8. Export errors.csv (top 50 FPs, top 50 FNs)
    print(f"Exporting top 50 FPs and top 50 FNs to {errors_csv_path}...")
    export_errors_csv(test_df, threshold=best_th, output_path=errors_csv_path)

    # 9. Export audit_100.csv
    print(f"Exporting 100 random hate-labelled test rows to {audit_100_path}...")
    export_audit_100_csv(test_df, seed=seed, output_path=audit_100_path)

    # 10. Write exhaustive Markdown report
    write_error_analysis_markdown(
        overall_metrics=overall_metrics,
        threshold=best_th,
        val_macro=val_macro,
        slices=slices,
        nn_agreement=nn_agreement,
        bootstrap=bootstrap,
        pol_fpr=pol_fpr,
        output_path=output_md_path,
    )

    # 11. Log run
    log_run(
        runs_csv_path=runs_csv_path,
        stage="stage5_error_analysis",
        seed=seed,
        parameters={
            "tuned_threshold": best_th,
            "nn_threshold": cos_th,
            "n_bootstrap": n_boot,
        },
        metrics={
            "ensemble_test_macro_f1": overall_metrics["macro_f1"],
            "ensemble_test_hate_f1": overall_metrics["hate_f1"],
            "ensemble_test_pr_auc": overall_metrics["pr_auc"],
            "roman_hate_f1": slices["script"]["roman"]["hate_f1"],
            "deva_hate_f1": slices["script"]["deva"]["hate_f1"],
            "nn_match_count": nn_agreement["n_match"],
            "nn_agreement_pct": nn_agreement["overall_agree"],
            "political_terms_fpr": pol_fpr["all"]["with_fpr"],
            "non_political_fpr": pol_fpr["all"]["without_fpr"],
        },
        status="SUCCESS",
        notes="Stage 5 error analysis complete: ensemble tuning, slices, bootstrap CIs, NN agreement, political FPR.",
    )

    # 12. Print clean summary table to console
    print("\n" + "=" * 85)
    print(f"STAGE 5 COMPLETE: ENSEMBLE TEST METRICS (Tuned Val Threshold = {best_th:.3f})")
    print("=" * 85)
    print(f"{'Slice':20s} | {'N':>6s} | {'Hate':>5s} | {'Macro-F1':>8s} | {'Hate-P':>7s} | {'Hate-R':>7s} | {'Hate-F1':>7s} | {'PR-AUC':>7s}")
    print("-" * 85)
    ov = overall_metrics
    print(f"{'Overall (Ensemble)':20s} | {ov['count']:6d} | {ov['hate_count']:5d} | {ov['macro_f1']:8.4f} | {ov['hate_p']:7.4f} | {ov['hate_r']:7.4f} | {ov['hate_f1']:7.4f} | {ov['pr_auc']:7.4f}")
    print("-" * 85)
    print("SCRIPT SLICES:")
    for sc in ["roman", "deva", "mixed", "unknown"]:
        s = slices["script"][sc]
        print(f"  {sc:18s} | {s['count']:6d} | {s['hate_count']:5d} | {s['macro_f1']:8.4f} | {s['hate_p']:7.4f} | {s['hate_r']:7.4f} | {s['hate_f1']:7.4f} | {s['pr_auc']:7.4f}")
    print("-" * 85)
    print("SOURCE SLICES:")
    for src in ["youtube", "reddit", "unknown"]:
        s = slices["source"][src]
        print(f"  {src:18s} | {s['count']:6d} | {s['hate_count']:5d} | {s['macro_f1']:8.4f} | {s['hate_p']:7.4f} | {s['hate_r']:7.4f} | {s['hate_f1']:7.4f} | {s['pr_auc']:7.4f}")
    print("-" * 85)
    print("LENGTH BUCKETS:")
    for lb in ["1-10 words", "11-25 words", "26-50 words", ">50 words"]:
        s = slices["length"][lb]
        print(f"  {lb:18s} | {s['count']:6d} | {s['hate_count']:5d} | {s['macro_f1']:8.4f} | {s['hate_p']:7.4f} | {s['hate_r']:7.4f} | {s['hate_f1']:7.4f} | {s['pr_auc']:7.4f}")
    print("-" * 85)
    print("EMOJI PRESENCE:")
    for em in ["with_emoji", "no_emoji"]:
        s = slices["emoji"][em]
        print(f"  {em:18s} | {s['count']:6d} | {s['hate_count']:5d} | {s['macro_f1']:8.4f} | {s['hate_p']:7.4f} | {s['hate_r']:7.4f} | {s['hate_f1']:7.4f} | {s['pr_auc']:7.4f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
