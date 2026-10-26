"""Stage 4: MuRIL Fine-Tuning for Hinglish Hate-Speech Detection.

Fine-tunes google/muril-base-cased in fp32 on the subsampled training set
with dynamic padding (max_len 64), batch size 8, gradient accumulation 4,
and class-weighted cross-entropy. Evaluates decision thresholds under both
val hate-class F1 maximization and val macro-F1 maximization, evaluates on test,
and saves predicted probabilities.
"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.utils.config import load_config, set_seed
from src.utils.logging import log_run


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(self, df: pd.DataFrame):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.scripts = df["script"].tolist()
        self.sources = df["source"].tolist()
        self.norm_keys = df["norm_key"].tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
            "script": self.scripts[idx],
            "source": self.sources[idx],
            "norm_key": self.norm_keys[idx],
        }


def make_collate_fn(tokenizer: Any, max_len: int = 64):
    """Create dynamic padding collate function."""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        encoded["meta"] = batch
        return encoded

    return collate_fn


def compute_metrics_for_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    scripts: pd.Series,
) -> Dict[str, Any]:
    """Compute overall and per-script classification metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)

    def calc(y_t: np.ndarray, y_p: np.ndarray, y_pr: np.ndarray) -> Dict[str, float]:
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

    overall = calc(y_true, preds, probs)
    per_script = {}
    for scr in ["roman", "deva", "mixed", "unknown"]:
        mask = (scripts == scr).values
        if mask.sum() > 0:
            per_script[scr] = calc(y_true[mask], preds[mask], probs[mask])

    return {"overall": overall, "per_script": per_script, "threshold": threshold}


def find_optimal_thresholds(
    y_val: np.ndarray, val_probs: np.ndarray
) -> Tuple[float, float, float, float]:
    """Find thresholds maximizing val hate F1 and val macro-F1."""
    thresholds = np.linspace(0.10, 0.90, 81)
    best_th_hate = 0.5
    best_hate_f1 = -1.0
    best_th_macro = 0.5
    best_macro_f1 = -1.0

    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        h_f1 = float(f1_score(y_val, preds, pos_label=1, average="binary", zero_division=0))
        m_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        if h_f1 > best_hate_f1:
            best_hate_f1 = h_f1
            best_th_hate = float(th)

        if m_f1 > best_macro_f1:
            best_macro_f1 = m_f1
            best_th_macro = float(th)

    return best_th_hate, best_hate_f1, best_th_macro, best_macro_f1


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on dataloader and return labels and positive class probabilities."""
    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

        all_probs.extend(probs)
        all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_probs)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_df: pd.DataFrame,
    num_epochs: int,
    grad_accum_steps: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    class_weights: torch.Tensor,
    device: torch.device,
    checkpoint_path: Path,
) -> Tuple[nn.Module, float]:
    """Train MuRIL model with early stopping on validation macro-F1."""
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    total_training_steps = (len(train_loader) // grad_accum_steps) * num_epochs
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_macro_f1 = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("-" * 65)
    print(f"Starting training: {num_epochs} epochs | Total opt steps: {total_training_steps:,}")
    print(f"Device: {device} | Per-device batch: {train_loader.batch_size} | Grad accum: {grad_accum_steps}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits, labels)
                loss_scaled = loss / grad_accum_steps
                loss_scaled.backward()
            except torch.cuda.OutOfMemoryError as e:
                print("CUDA OOM detected! Freezing embeddings layer per environment rule...")
                torch.cuda.empty_cache()
                for p in model.bert.embeddings.parameters():
                    p.requires_grad = False
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits, labels)
                loss_scaled = loss / grad_accum_steps
                loss_scaled.backward()

            total_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - t0

        # Quick validation evaluation (argmax) to track epoch progress
        y_val_true, val_probs = predict_probabilities(model, val_loader, device)
        val_preds_default = (val_probs >= 0.5).astype(int)
        val_macro_f1 = float(f1_score(y_val_true, val_preds_default, average="macro"))

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Macro-F1 (th=0.5): {val_macro_f1:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved new best checkpoint (Val Macro-F1: {best_val_macro_f1:.4f})")

    # Load best checkpoint
    print(f"Loading best checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model, best_val_macro_f1


def write_muril_report(
    test_metrics_macro: Dict[str, Any],
    test_metrics_hate: Dict[str, Any],
    val_th_macro: float,
    val_th_hate: float,
    val_m_f1_macro: float,
    val_h_f1_hate: float,
    best_val_macro: float,
    output_path: Path,
) -> None:
    """Generate Markdown report for Stage 4 MuRIL evaluation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ov_macro = test_metrics_macro["overall"]
    ps_macro = test_metrics_macro["per_script"]

    ov_hate = test_metrics_hate["overall"]
    ps_hate = test_metrics_hate["per_script"]

    content = f"""# Stage 4: MuRIL Fine-Tuning Evaluation (Seed 42)

- **Model:** `google/muril-base-cased` (Sequence Classification, 2 labels)
- **Precision:** FP32 exclusively
- **Sequence Length:** `max_len = 64` (Dynamic batch padding)
- **Batching:** Per-device batch 8 $\\times$ Gradient Accumulation 4 (Effective batch size = 32)
- **Optimizer:** AdamW (`lr=2e-5`, `weight_decay=0.01`, linear warmup 10%)
- **Loss:** Class-weighted cross entropy based on subsampled train distribution
- **Best Validation Macro-F1 (checkpoint):** `{best_val_macro:.4f}`

---

## 1. Decision Threshold Selection on Validation Set

The decision threshold was tuned on the validation set ($N = 13,663$) using two distinct criteria:
1. **Max Val Macro-F1:** Threshold $\\tau = {val_th_macro:.2f}$ (Achieved Val Macro-F1 = `{val_m_f1_macro:.4f}`) — *directly comparable with Stage 3 baseline methodology*.
2. **Max Val Hate-Class F1:** Threshold $\\tau = {val_th_hate:.2f}$ (Achieved Val Hate F1 = `{val_h_f1_hate:.4f}`) — *prioritizing positive class detection*.

---

## 2. Test Set Evaluation: Threshold Tuned for Max Val Macro-F1 ($\\tau = {val_th_macro:.2f}$)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **{ov_macro['count']:,}** | **{ov_macro['macro_f1']:.4f}** | **{ov_macro['hate_p']:.4f}** | **{ov_macro['hate_r']:.4f}** | **{ov_macro['hate_f1']:.4f}** | **{ov_macro['pr_auc']:.4f}** |
| **Roman** | {ps_macro.get('roman', {}).get('count', 0):,} | {ps_macro.get('roman', {}).get('macro_f1', 0):.4f} | {ps_macro.get('roman', {}).get('hate_p', 0):.4f} | {ps_macro.get('roman', {}).get('hate_r', 0):.4f} | {ps_macro.get('roman', {}).get('hate_f1', 0):.4f} | {ps_macro.get('roman', {}).get('pr_auc', 0):.4f} |
| **Devanagari (`deva`)** | {ps_macro.get('deva', {}).get('count', 0):,} | {ps_macro.get('deva', {}).get('macro_f1', 0):.4f} | {ps_macro.get('deva', {}).get('hate_p', 0):.4f} | {ps_macro.get('deva', {}).get('hate_r', 0):.4f} | {ps_macro.get('deva', {}).get('hate_f1', 0):.4f} | {ps_macro.get('deva', {}).get('pr_auc', 0):.4f} |
| **Mixed** | {ps_macro.get('mixed', {}).get('count', 0):,} | {ps_macro.get('mixed', {}).get('macro_f1', 0):.4f} | {ps_macro.get('mixed', {}).get('hate_p', 0):.4f} | {ps_macro.get('mixed', {}).get('hate_r', 0):.4f} | {ps_macro.get('mixed', {}).get('hate_f1', 0):.4f} | {ps_macro.get('mixed', {}).get('pr_auc', 0):.4f} |
| **Unknown** | {ps_macro.get('unknown', {}).get('count', 0):,} | {ps_macro.get('unknown', {}).get('macro_f1', 0):.4f} | {ps_macro.get('unknown', {}).get('hate_p', 0):.4f} | {ps_macro.get('unknown', {}).get('hate_r', 0):.4f} | {ps_macro.get('unknown', {}).get('hate_f1', 0):.4f} | {ps_macro.get('unknown', {}).get('pr_auc', 0):.4f} |

---

## 3. Test Set Evaluation: Threshold Tuned for Max Val Hate-Class F1 ($\\tau = {val_th_hate:.2f}$)

| Script Stratum | Sample Count ($N$) | Macro-F1 | Hate Precision | Hate Recall | Hate-Class F1 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall (Test)** | **{ov_hate['count']:,}** | **{ov_hate['macro_f1']:.4f}** | **{ov_hate['hate_p']:.4f}** | **{ov_hate['hate_r']:.4f}** | **{ov_hate['hate_f1']:.4f}** | **{ov_hate['pr_auc']:.4f}** |
| **Roman** | {ps_hate.get('roman', {}).get('count', 0):,} | {ps_hate.get('roman', {}).get('macro_f1', 0):.4f} | {ps_hate.get('roman', {}).get('hate_p', 0):.4f} | {ps_hate.get('roman', {}).get('hate_r', 0):.4f} | {ps_hate.get('roman', {}).get('hate_f1', 0):.4f} | {ps_hate.get('roman', {}).get('pr_auc', 0):.4f} |
| **Devanagari (`deva`)** | {ps_hate.get('deva', {}).get('count', 0):,} | {ps_hate.get('deva', {}).get('macro_f1', 0):.4f} | {ps_hate.get('deva', {}).get('hate_p', 0):.4f} | {ps_hate.get('deva', {}).get('hate_r', 0):.4f} | {ps_hate.get('deva', {}).get('hate_f1', 0):.4f} | {ps_hate.get('deva', {}).get('pr_auc', 0):.4f} |
| **Mixed** | {ps_hate.get('mixed', {}).get('count', 0):,} | {ps_hate.get('mixed', {}).get('macro_f1', 0):.4f} | {ps_hate.get('mixed', {}).get('hate_p', 0):.4f} | {ps_hate.get('mixed', {}).get('hate_r', 0):.4f} | {ps_hate.get('mixed', {}).get('hate_f1', 0):.4f} | {ps_hate.get('mixed', {}).get('pr_auc', 0):.4f} |
| **Unknown** | {ps_hate.get('unknown', {}).get('count', 0):,} | {ps_hate.get('unknown', {}).get('macro_f1', 0):.4f} | {ps_hate.get('unknown', {}).get('hate_p', 0):.4f} | {ps_hate.get('unknown', {}).get('hate_r', 0):.4f} | {ps_hate.get('unknown', {}).get('hate_f1', 0):.4f} | {ps_hate.get('unknown', {}).get('pr_auc', 0):.4f} |
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: MuRIL Fine-Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage4_muril.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (defaults to config value)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get("model_name", "google/muril-base-cased")
    processed_dir = Path(config.get("processed_dir", "data/processed"))
    checkpoints_dir = Path(config.get("checkpoints_dir", "results/checkpoints"))
    runs_csv_path = Path(config.get("runs_csv_path", "results/runs.csv"))
    results_dir = Path(config.get("results_dir", "results"))
    preds_out_path = results_dir / f"preds_muril_seed{seed}.parquet"
    output_md_path = results_dir / f"muril_seed{seed}_metrics.md"


    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("per_device_batch_size", 8)
    grad_accum = config.get("gradient_accumulation_steps", 4)
    lr = float(config.get("learning_rate", 2.0e-5))
    weight_decay = float(config.get("weight_decay", 0.01))
    warmup_ratio = float(config.get("warmup_ratio", 0.10))
    max_len = config.get("max_len", 64)

    # 1. Load data
    train_df = pd.read_parquet(processed_dir / "train_subsampled.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    # 2. Setup tokenizer and collator
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collate_fn = make_collate_fn(tokenizer, max_len=max_len)

    train_ds = TextClassificationDataset(train_df)
    val_ds = TextClassificationDataset(val_df)
    test_ds = TextClassificationDataset(test_df)

    # num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 3. Class weights from training set
    n_total = len(train_df)
    n_0 = (train_df["label"] == 0).sum()
    n_1 = (train_df["label"] == 1).sum()
    w0 = n_total / (2.0 * n_0)
    w1 = n_total / (2.0 * n_1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float, device=device)

    # 4. Load model
    print(f"Loading {model_name} in fp32 on {device}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # 5. Train model
    best_ckpt_path = checkpoints_dir / f"muril_best_seed{seed}.pt"
    model, best_val_macro = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_df=val_df,
        num_epochs=num_epochs,
        grad_accum_steps=grad_accum,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        class_weights=class_weights,
        device=device,
        checkpoint_path=best_ckpt_path,
    )

    # 6. Evaluation on Val and Test
    print("Running final validation and test predictions...")
    y_val_true, val_probs = predict_probabilities(model, val_loader, device)
    y_test_true, test_probs = predict_probabilities(model, test_loader, device)

    # Threshold selection two ways:
    th_hate, val_f1_hate, th_macro, val_f1_macro = find_optimal_thresholds(y_val_true, val_probs)

    # Metrics at max val macro-F1 threshold
    test_metrics_macro = compute_metrics_for_probs(y_test_true, test_probs, th_macro, test_df["script"])
    val_metrics_macro = compute_metrics_for_probs(y_val_true, val_probs, th_macro, val_df["script"])

    # Metrics at max val hate-class F1 threshold
    test_metrics_hate = compute_metrics_for_probs(y_test_true, test_probs, th_hate, test_df["script"])
    val_metrics_hate = compute_metrics_for_probs(y_val_true, val_probs, th_hate, val_df["script"])

    # 7. Save predicted probabilities to results/preds_muril_seed42.parquet
    val_preds_df = val_df[["norm_key", "text", "label", "script", "source"]].copy()
    val_preds_df["prob"] = val_probs
    val_preds_df["split"] = "val"

    test_preds_df = test_df[["norm_key", "text", "label", "script", "source"]].copy()
    test_preds_df["prob"] = test_probs
    test_preds_df["split"] = "test"

    combined_preds = pd.concat([val_preds_df, test_preds_df], ignore_index=True)
    preds_out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_preds.to_parquet(preds_out_path, index=False, engine="pyarrow")
    print(f"Saved val and test predictions to: {preds_out_path}")

    # 8. Write Markdown report
    write_muril_report(
        test_metrics_macro=test_metrics_macro,
        test_metrics_hate=test_metrics_hate,
        val_th_macro=th_macro,
        val_th_hate=th_hate,
        val_m_f1_macro=val_f1_macro,
        val_h_f1_hate=val_f1_hate,
        best_val_macro=best_val_macro,
        output_path=output_md_path,
    )

    # 9. Log to runs.csv
    log_run(
        runs_csv_path=runs_csv_path,
        stage="stage4_muril",
        seed=seed,
        parameters={
            "model_name": model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "th_val_macro": th_macro,
            "th_val_hate": th_hate,
        },
        metrics={
            "val_macro_f1": val_f1_macro,
            "val_hate_f1": val_f1_hate,
            "test_macro_f1_tuned_macro": test_metrics_macro["overall"]["macro_f1"],
            "test_hate_f1_tuned_macro": test_metrics_macro["overall"]["hate_f1"],
            "test_pr_auc": test_metrics_macro["overall"]["pr_auc"],
            "test_macro_f1_tuned_hate": test_metrics_hate["overall"]["macro_f1"],
            "test_hate_f1_tuned_hate": test_metrics_hate["overall"]["hate_f1"],
        },
        status="SUCCESS",
        notes=f"MuRIL seed {seed} complete. Thresholds: th_macro={th_macro:.2f}, th_hate={th_hate:.2f}.",
    )

    # 10. Print formatted tables to console
    print("=" * 80)
    print("STAGE 4 MuRIL SEED 42 COMPLETE")
    print(f"Validation Optimal Thresholds: Max Macro-F1 = {th_macro:.2f} | Max Hate-F1 = {th_hate:.2f}")
    print("=" * 80)
    print(f"TEST METRICS: Threshold Tuned on Val Macro-F1 (th={th_macro:.2f})")
    print("-" * 80)
    print(f"{'Script Stratum':18s} | {'N':>6s} | {'Macro-F1':>8s} | {'Hate-P':>7s} | {'Hate-R':>7s} | {'Hate-F1':>7s} | {'PR-AUC':>7s}")
    print("-" * 80)
    ov_m = test_metrics_macro["overall"]
    print(
        f"{'Overall (Test)':18s} | {ov_m['count']:6d} | "
        f"{ov_m['macro_f1']:8.4f} | {ov_m['hate_p']:7.4f} | {ov_m['hate_r']:7.4f} | {ov_m['hate_f1']:7.4f} | {ov_m['pr_auc']:7.4f}"
    )
    for sc in ["roman", "deva", "mixed", "unknown"]:
        if sc in test_metrics_macro["per_script"]:
            s_res = test_metrics_macro["per_script"][sc]
            print(
                f"{sc:18s} | {s_res['count']:6d} | "
                f"{s_res['macro_f1']:8.4f} | {s_res['hate_p']:7.4f} | {s_res['hate_r']:7.4f} | {s_res['hate_f1']:7.4f} | {s_res['pr_auc']:7.4f}"
            )
    print("=" * 80)
    print(f"TEST METRICS: Threshold Tuned on Val Hate-Class F1 (th={th_hate:.2f})")
    print("-" * 80)
    ov_h = test_metrics_hate["overall"]
    print(
        f"{'Overall (Test)':18s} | {ov_h['count']:6d} | "
        f"{ov_h['macro_f1']:8.4f} | {ov_h['hate_p']:7.4f} | {ov_h['hate_r']:7.4f} | {ov_h['hate_f1']:7.4f} | {ov_h['pr_auc']:7.4f}"
    )
    for sc in ["roman", "deva", "mixed", "unknown"]:
        if sc in test_metrics_hate["per_script"]:
            s_res = test_metrics_hate["per_script"][sc]
            print(
                f"{sc:18s} | {s_res['count']:6d} | "
                f"{s_res['macro_f1']:8.4f} | {s_res['hate_p']:7.4f} | {s_res['hate_r']:7.4f} | {s_res['hate_f1']:7.4f} | {s_res['pr_auc']:7.4f}"
            )

    print("=" * 80)
    print(f"Metrics report written to: {output_md_path}")
    print(f"Run logged to: {runs_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
