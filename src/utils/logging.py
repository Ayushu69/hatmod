"""Run logging utilities for pipeline reproducibility tracking."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json
import pandas as pd


def log_run(
    runs_csv_path: str | Path,
    stage: str,
    seed: int,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    status: str = "SUCCESS",
    notes: str = "",
) -> None:
    """Log pipeline run details to a central CSV tracking file.

    Args:
        runs_csv_path: Path to the runs.csv file.
        stage: Pipeline stage name (e.g. stage1_audit).
        seed: Random seed used.
        parameters: Dictionary of parameters used for the run.
        metrics: Dictionary of key metrics produced.
        status: Run status ('SUCCESS', 'FAILED', etc.).
        notes: Optional extra notes or observations.
    """
    path = Path(runs_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "seed": seed,
        "parameters": json.dumps(parameters),
        "metrics": json.dumps(metrics),
        "status": status,
        "notes": notes,
    }

    new_df = pd.DataFrame([row])
    if path.is_file():
        new_df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        new_df.to_csv(path, mode="w", header=True, index=False, encoding="utf-8")
