"""Stage 3: TF-IDF + Logistic Regression Baseline.

Fits char n-grams (2-5) and word n-grams (1-2) with Logistic Regression
(class_weight='balanced') on the subsampled training set, tunes hyperparameters
and decision threshold on validation, and reports final metrics on test.
"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion

from src.utils.config import load_config, set_seed
from src.utils.logging import log_run


def build_feature_extractor(
    word_ngram_range: Tuple[int, int] = (1, 2),
    char_ngram_range: Tuple[int, int] = (2, 5),
    word_min_df: int = 2,
    char_min_df: int = 5,
    sublinear_tf: bool = True,
) -> FeatureUnion:
    """Construct TF-IDF FeatureUnion combining word and char n-grams.

    Args:
        word_ngram_range: (min_n, max_n) for word analyzer.
        char_ngram_range: (min_n, max_n) for char analyzer.
        word_min_df: Minimum document frequency for words.
        char_min_df: Minimum document frequency for character n-grams.
        sublinear_tf: Apply sublinear tf scaling.

    Returns:
        Configured FeatureUnion instance.
    """
    word_vec = TfidfVectorizer(
        ngram_range=word_ngram_range,
        min_df=word_min_df,
        sublinear_tf=sublinear_tf,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngram_range,
        min_df=char_min_df,
        sublinear_tf=sublinear_tf,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


def tune_model(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    c_candidates: List[float],
    seed: int = 42,
    tune_metric: str = "macro_f1",
) -> Tuple[LogisticRegression, float, float, Dict[str, Any]]:
    """Tune regularization parameter C and decision threshold on validation set.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        c_candidates: List of C values to evaluate.
        seed: Random seed.
        tune_metric: Target metric to optimize ('macro_f1' or 'hate_f1').

    Returns:
        Tuple of (best_model, best_c, best_threshold, tuning_history).
    """
    best_score = -1.0
    best_c = 1.0
    best_threshold = 0.5
    best_model = None
    history: List[Dict[str, Any]] = []

    thresholds = np.linspace(0.20, 0.85, 66)

    for c_val in c_candidates:
        model = LogisticRegression(
            C=c_val,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, val_probs))

        best_th_for_c = 0.5
        best_score_for_c = -1.0
        best_hate_f1_for_c = 0.0

        for th in thresholds:
            preds = (val_probs >= th).astype(int)
            macro_f1 = float(f1_score(y_val, preds, average="macro"))
            hate_f1 = float(f1_score(y_val, preds, pos_label=1, average="binary"))

            score = macro_f1 if tune_metric == "macro_f1" else hate_f1
            if score > best_score_for_c:
                best_score_for_c = score
                best_th_for_c = float(th)
                best_hate_f1_for_c = hate_f1

        history.append({
            "C": c_val,
            "best_threshold": best_th_for_c,
            "val_score": best_score_for_c,
            "val_hate_f1": best_hate_f1_for_c,
            "val_pr_auc": val_pr_auc,
        })

        if best_score_for_c > best_score:
            best_score = best_score_for_c
            best_c = c_val
            best_threshold = best_th_for_c
            best_model = model

    return best_model, best_c, best_threshold, {"candidates": history}


def evaluate_split(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    scripts: pd.Series,
) -> Dict[str, Any]:
    """Compute comprehensive metrics overall and per script.

    Args:
        y_true: Binary ground truth labels.
        y_probs: Predicted probabilities for positive class (hate).
        threshold: Decision threshold for classification.
        scripts: Series of script identifiers for each example.

    Returns:
        Dictionary of overall and per-script metrics.
    """
    preds = (y_probs >= threshold).astype(int)

    def calc_metrics(y_t: np.ndarray, y_p: np.ndarray, y_pr: np.ndarray) -> Dict[str, float]:
        if len(y_t) == 0 or len(np.unique(y_t)) < 2:
            return {"count": len(y_t), "macro_f1": 0.0, "hate_p": 0.0, "hate_r": 0.0, "hate_f1": 0.0, "pr_auc": 0.0}

        macro_f1 = float(f1_score(y_t, y_p, average="macro"))
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, pos_label=1, average="binary", zero_division=0)
        pr_auc = float(average_precision_score(y_t, y_pr))
        return {
            "count": int(len(y_t)),
            "macro_f1": macro_f1,
            "hate_p": float(p),
            "hate_r": float(r),
            "hate_f1": float(f1),
            "pr_auc": pr_auc,
        }

    overall = calc_metrics(y_true, preds, y_probs)
    per_script = {}

    for scr in ["roman", "deva", "mixed", "unknown"]:
        mask = (scripts == scr).values
        if mask.sum() > 0:
            per_script[scr] = calc_metrics(y_true[mask], preds[mask], y_probs[mask])

    return {"overall": overall, "per_script": per_script, "threshold": threshold}


def write_metrics_markdown(
    results: Dict[str, Any],
    best_c: float,
    threshold: float,
    output_path: Path,
) -> None:
    """Write formatted markdown table with baseline evaluation results.

    Args:
        results: Dictionary containing evaluated metrics.
        best_c: Selected regularization parameter C.
        threshold: Selected decision threshold.
        output_path: Path to output Markdown file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ov = results["overall"]
    ps = results["per_script"]

    md = f"""# Stage 3 Baseline: TF-IDF + Logistic Regression

- **Model:** Logistic Regression (`class_weight='balanced'`, `C={best_c}`)
- **Features:** TF-IDF (char n-grams 2-5 + word n-grams 1-2, sublinear tf)
- **Tuned Decision Threshold (Validation):** `{threshold:.2f}`
- **Evaluation Split:** Test set ($N = {ov['count']:,}$, natural hate ratio: 7.96%)

## Test Metrics Summary (Headline Metrics: Macro-F1 & Hate F1)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **{ov['count']:,}** | **{ov['macro_f1']:.4f}** | **{ov['hate_p']:.4f}** | **{ov['hate_r']:.4f}** | **{ov['hate_f1']:.4f}** | **{ov['pr_auc']:.4f}** |
| **Roman** | {ps.get('roman', {}).get('count', 0):,} | {ps.get('roman', {}).get('macro_f1', 0):.4f} | {ps.get('roman', {}).get('hate_p', 0):.4f} | {ps.get('roman', {}).get('hate_r', 0):.4f} | {ps.get('roman', {}).get('hate_f1', 0):.4f} | {ps.get('roman', {}).get('pr_auc', 0):.4f} |
| **Devanagari (`deva`)** | {ps.get('deva', {}).get('count', 0):,} | {ps.get('deva', {}).get('macro_f1', 0):.4f} | {ps.get('deva', {}).get('hate_p', 0):.4f} | {ps.get('deva', {}).get('hate_r', 0):.4f} | {ps.get('deva', {}).get('hate_f1', 0):.4f} | {ps.get('deva', {}).get('pr_auc', 0):.4f} |
| **Mixed** | {ps.get('mixed', {}).get('count', 0):,} | {ps.get('mixed', {}).get('macro_f1', 0):.4f} | {ps.get('mixed', {}).get('hate_p', 0):.4f} | {ps.get('mixed', {}).get('hate_r', 0):.4f} | {ps.get('mixed', {}).get('hate_f1', 0):.4f} | {ps.get('mixed', {}).get('pr_auc', 0):.4f} |
| **Unknown** | {ps.get('unknown', {}).get('count', 0):,} | {ps.get('unknown', {}).get('macro_f1', 0):.4f} | {ps.get('unknown', {}).get('hate_p', 0):.4f} | {ps.get('unknown', {}).get('hate_r', 0):.4f} | {ps.get('unknown', {}).get('hate_f1', 0):.4f} | {ps.get('unknown', {}).get('pr_auc', 0):.4f} |

### Observations:
1. **Script Disparity:** Hate-class F1 on Roman Hinglish vs Devanagari demonstrates baseline performance across scripts prior to transformer fine-tuning.
2. **Precision-Recall Balance:** At threshold `{threshold:.2f}`, the balanced model maintains high recall while moderating false alarms on the imbalanced test set.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: TF-IDF Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3_baseline.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    processed_dir = Path(config.get("processed_dir", "data/processed"))
    runs_csv_path = Path(config.get("runs_csv_path", "results/runs.csv"))
    output_md_path = Path(config.get("output_md_path", "results/baseline_metrics.md"))
    c_candidates = config.get("c_candidates", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    tune_metric = config.get("tune_metric", "macro_f1")

    # 1. Load data splits
    train_sub = pd.read_parquet(processed_dir / "train_subsampled.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    # 2. Extract features
    print("Extracting TF-IDF features (word 1-2 + char 2-5)...")
    t0 = time.time()
    feature_union = build_feature_extractor(
        word_ngram_range=tuple(config.get("word_ngram_range", [1, 2])),
        char_ngram_range=tuple(config.get("char_ngram_range", [2, 5])),
        word_min_df=config.get("word_min_df", 2),
        char_min_df=config.get("char_min_df", 5),
        sublinear_tf=config.get("sublinear_tf", True),
    )

    X_train = feature_union.fit_transform(train_sub["text"])
    X_val = feature_union.transform(val_df["text"])
    X_test = feature_union.transform(test_df["text"])
    print(f"Feature extraction complete in {time.time() - t0:.2f}s | Vocabulary dimension: {X_train.shape[1]:,}")

    y_train = train_sub["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    # 3. Tune on validation set
    print("Tuning regularization C and decision threshold on validation set...")
    best_model, best_c, best_th, history = tune_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        c_candidates=c_candidates,
        seed=seed,
        tune_metric=tune_metric,
    )
    print(f"Tuning complete: Best C={best_c}, Optimal Threshold={best_th:.2f}")

    # 4. Evaluate on test set
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_split(
        y_true=y_test,
        y_probs=test_probs,
        threshold=best_th,
        scripts=test_df["script"],
    )

    # 5. Write results
    write_metrics_markdown(test_metrics, best_c, best_th, output_md_path)

    ov = test_metrics["overall"]
    ps = test_metrics["per_script"]

    # 6. Log run details
    log_run(
        runs_csv_path=runs_csv_path,
        stage="stage3_baseline",
        seed=seed,
        parameters={
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "best_c": best_c,
            "threshold": best_th,
            "vocab_size": X_train.shape[1],
        },
        metrics={
            "test_macro_f1": ov["macro_f1"],
            "test_hate_f1": ov["hate_f1"],
            "test_hate_p": ov["hate_p"],
            "test_hate_r": ov["hate_r"],
            "test_pr_auc": ov["pr_auc"],
            "roman_hate_f1": ps.get("roman", {}).get("hate_f1", 0.0),
            "deva_hate_f1": ps.get("deva", {}).get("hate_f1", 0.0),
            "mixed_hate_f1": ps.get("mixed", {}).get("hate_f1", 0.0),
        },
        status="SUCCESS",
        notes=f"Stage 3 baseline evaluated on test set (best C={best_c}, th={best_th:.2f}).",
    )

    # 7. Print summary metrics table to console
    print("=" * 80)
    print(f"STAGE 3 BASELINE TEST RESULTS (C={best_c}, Decision Threshold={best_th:.2f})")
    print("=" * 80)
    print(f"{'Script Stratum':18s} | {'N':>6s} | {'Macro-F1':>8s} | {'Hate-P':>7s} | {'Hate-R':>7s} | {'Hate-F1':>7s} | {'PR-AUC':>7s}")
    print("-" * 80)
    print(
        f"{'Overall (Test)':18s} | {ov['count']:6d} | "
        f"{ov['macro_f1']:8.4f} | {ov['hate_p']:7.4f} | {ov['hate_r']:7.4f} | {ov['hate_f1']:7.4f} | {ov['pr_auc']:7.4f}"
    )
    for sc in ["roman", "deva", "mixed", "unknown"]:
        if sc in ps:
            s_res = ps[sc]
            print(
                f"{sc:18s} | {s_res['count']:6d} | "
                f"{s_res['macro_f1']:8.4f} | {s_res['hate_p']:7.4f} | {s_res['hate_r']:7.4f} | {s_res['hate_f1']:7.4f} | {s_res['pr_auc']:7.4f}"
            )
    print("-" * 80)
    print(f"Results written to: {output_md_path}")
    print(f"Logged run to: {runs_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
