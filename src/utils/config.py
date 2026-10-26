"""Configuration and seed utilities."""

from pathlib import Path
from typing import Any, Dict
import random
import numpy as np
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary with parsed configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across standard libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
