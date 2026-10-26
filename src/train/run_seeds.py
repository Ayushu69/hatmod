"""Multi-seed runner for MuRIL fine-tuning (Seeds 43 & 44).

Executes training for seeds 43 and 44, aggregates evaluation results across
seeds 42, 43, and 44, computes mean +- std overall and per script,
and writes results/muril_multiseed_summary.md.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

from src.utils.config import load_config
from src.utils.logging import log_run


def run_training_for_seed(config_path: str, seed: int) -> None:
    """Execute training script for a specific seed via subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "src.train.train_muril",
        "--config",
        config_path,
        "--seed",
        str(seed),
    ]
    print(f"\n{'='*70}\nLAUNCHING MURIL TRAINING FOR SEED {seed}\n{'='*70}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for seed {seed} with return code {result.returncode}")


def evaluate_seed_predictions(preds_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate predictions for a seed finding optimal thresholds on val and evaluating on test."""
    df = pd.read_parquet(preds_path)
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    y_val = val["label"].values
    val_probs = val["prob"].values
    y_test = test["label"].values
    test_probs = test["prob"].values

    # Sweep thresholds on validation set (0.10 to 0.99)
    thresholds = np.linspace(0.10, 0.99, 90)
    best_th_macro = 0.5
    best_val_macro = -1.0
    best_th_hate = 0.5
    best_val_hate = -1.0

    for th in thresholds:
        vp = (val_probs >= th).astype(int)
        m_f1 = float(f1_score(y_val, vp, average="macro", zero_division=0))
        h_f1 = float(f1_score(y_val, vp, pos_label=1, average="binary", zero_division=0))
        if m_f1 > best_val_macro:
            best_val_macro = m_f1
            best_th_macro = float(th)
        if h_f1 > best_val_hate:
            best_val_hate = h_f1
            best_th_hate = float(th)

    def calc(y_t: np.ndarray, p: np.ndarray, pr: np.ndarray) -> Dict[str, float]:
        if len(y_t) == 0 or len(np.unique(y_t)) < 2:
            return {"macro_f1": 0.0, "hate_p": 0.0, "hate_r": 0.0, "hate_f1": 0.0, "pr_auc": 0.0}
        macro_f1 = float(f1_score(y_t, p, average="macro", zero_division=0))
        prec, rec, f1, _ = precision_recall_fscore_support(y_t, p, pos_label=1, average="binary", zero_division=0)
        pr_auc = float(average_precision_score(y_t, pr))
        return {
            "macro_f1": macro_f1,
            "hate_p": float(prec),
            "hate_r": float(rec),
            "hate_f1": float(f1),
            "pr_auc": pr_auc,
        }

    def compute_all_metrics(th: float) -> Dict[str, Any]:
        preds = (test_probs >= th).astype(int)
        ov = calc(y_test, preds, test_probs)
        ps = {}
        for scr in ["roman", "deva", "mixed", "unknown"]:
            mask = (test["script"] == scr).values
            if mask.sum() > 0:
                ps[scr] = calc(y_test[mask], preds[mask], test_probs[mask])
        return {"threshold": th, "overall": ov, "per_script": ps}

    res_macro = compute_all_metrics(best_th_macro)
    res_hate = compute_all_metrics(best_th_hate)
    return res_macro, res_hate


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Seed MuRIL Runner")
    parser.add_argument("--config", type=str, default="configs/stage4_muril.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config.get("results_dir", "results"))
    runs_csv_path = Path(config.get("runs_csv_path", "results/runs.csv"))

    # 1. Run training for new seeds
    for s in args.seeds:
        preds_file = results_dir / f"preds_muril_seed{s}.parquet"
        if not preds_file.is_file():
            run_training_for_seed(args.config, s)
        else:
            print(f"Predictions file for seed {s} already exists at {preds_file}, skipping training.")

    # 2. Aggregate all 3 seeds: 42, 43, 44
    all_seeds = [42] + list(args.seeds)
    all_results_macro: Dict[int, Dict[str, Any]] = {}

    for s in all_seeds:
        preds_file = results_dir / f"preds_muril_seed{s}.parquet"
        res_macro, _ = evaluate_seed_predictions(preds_file)
        all_results_macro[s] = res_macro

    # 3. Compute mean +- std across seeds
    strata = ["overall", "roman", "deva", "mixed", "unknown"]
    metrics_list = ["macro_f1", "hate_p", "hate_r", "hate_f1", "pr_auc"]

    summary_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for st in strata:
        summary_stats[st] = {}
        for m in metrics_list:
            vals = []
            for s in all_seeds:
                if st == "overall":
                    v = all_results_macro[s]["overall"][m]
                else:
                    v = all_results_macro[s]["per_script"].get(st, {}).get(m, 0.0)
                vals.append(v)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary_stats[st][m] = (mean_v, std_v)

    # 4. Generate Markdown summary
    out_md = results_dir / "muril_multiseed_summary.md"
    rows = []
    for st in strata:
        label = "Overall (Test)" if st == "overall" else st
        m_f1_m, m_f1_s = summary_stats[st]["macro_f1"]
        p_m, p_s = summary_stats[st]["hate_p"]
        r_m, r_s = summary_stats[st]["hate_r"]
        f1_m, f1_s = summary_stats[st]["hate_f1"]
        auc_m, auc_s = summary_stats[st]["pr_auc"]
        rows.append(
            f"| **{label}** | {m_f1_m:.4f} ± {m_f1_s:.4f} | {p_m:.4f} ± {p_s:.4f} | {r_m:.4f} ± {r_s:.4f} | **{f1_m:.4f} ± {f1_s:.4f}** | {auc_m:.4f} ± {auc_s:.4f} |"
        )

    md_content = f"""# MuRIL Multi-Seed Performance Summary (Seeds 42, 43, 44)

- **Model:** `google/muril-base-cased` (Sequence Classification, 2 labels)
- **Seeds Evaluated:** 42, 43, 44
- **Decision Threshold:** Tuned per-seed on validation set (maximizing val Macro-F1)
- **Test Set Size:** $N = 13,664$ (Natural 7.96% hate ratio)

## Test Metrics Across Seeds (Mean ± Std)

| Script Stratum | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join(rows)}

### Per-Seed Hate-Class F1 Breakdown:
"""
    for s in all_seeds:
        ov = all_results_macro[s]["overall"]
        th = all_results_macro[s]["threshold"]
        md_content += f"- **Seed {s} ($\\tau = {th:.2f}$):** Macro-F1 = {ov['macro_f1']:.4f}, Hate-F1 = {ov['hate_f1']:.4f}, PR-AUC = {ov['pr_auc']:.4f}\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content.strip() + "\n")

    # 5. Log run
    log_run(
        runs_csv_path=runs_csv_path,
        stage="stage4_multiseed",
        seed=42,
        parameters={"seeds": all_seeds, "num_seeds": len(all_seeds)},
        metrics={
            "mean_hate_f1": summary_stats["overall"]["hate_f1"][0],
            "std_hate_f1": summary_stats["overall"]["hate_f1"][1],
            "mean_macro_f1": summary_stats["overall"]["macro_f1"][0],
            "std_macro_f1": summary_stats["overall"]["macro_f1"][1],
            "mean_pr_auc": summary_stats["overall"]["pr_auc"][0],
            "std_pr_auc": summary_stats["overall"]["pr_auc"][1],
        },
        status="SUCCESS",
        notes="Multi-seed MuRIL evaluation completed across seeds 42, 43, 44.",
    )

    # 6. Print summary table to console
    print("\n" + "=" * 85)
    print("STAGE 4 COMPLETE: MURIL MULTI-SEED PERFORMANCE (SEEDS 42, 43, 44)")
    print("=" * 85)
    print(f"{'Script Stratum':18s} | {'Macro-F1 (Mean±Std)':>20s} | {'Hate-F1 (Mean±Std)':>20s} | {'PR-AUC (Mean±Std)':>20s}")
    print("-" * 85)
    for st in strata:
        label = "Overall (Test)" if st == "overall" else st
        m_f1_m, m_f1_s = summary_stats[st]["macro_f1"]
        f1_m, f1_s = summary_stats[st]["hate_f1"]
        auc_m, auc_s = summary_stats[st]["pr_auc"]
        print(f"{label:18s} | {m_f1_m:.4f} ± {m_f1_s:.4f}      | {f1_m:.4f} ± {f1_s:.4f}      | {auc_m:.4f} ± {auc_s:.4f}")
    print("-" * 85)
    print(f"Detailed multi-seed report written to: {out_md}")
    print("=" * 85)


if __name__ == "__main__":
    main()
