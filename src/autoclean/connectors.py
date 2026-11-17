
from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)

# --- Pandalyzer extension: exports ---
import json

def save_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Save a DataFrame to CSV, creating parent directories, and return the path."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=index)
    return destination


def save_quality_report(report: dict, path: str | Path) -> Path:
    """Save a quality report as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def safe(value):
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="index")
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if value is pd.NA:
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        return value
    with destination.open("w", encoding="utf-8") as file:
        json.dump(safe(report), file, indent=2, ensure_ascii=False, default=str)
    return destination
