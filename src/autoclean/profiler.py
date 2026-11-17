import pandas as pd

from autoclean._utils import distinct_counts, duplicated_mask


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Generate a read-only data quality profile."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = df.isna()
    column_report = pd.DataFrame(
        {
            "data_type": df.dtypes.astype(str),
            "missing_count": missing.sum(),
            # An empty frame has no missing cells: report 0.0, not NaN.
            "missing_percent": (missing.mean() * 100).fillna(0.0).round(2),
            "distinct_count": distinct_counts(df),
        }
    )

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(duplicated_mask(df).sum()),
        "columns": column_report,
    }


def _quality_metrics(df: pd.DataFrame) -> dict:
    missing = int(df.isna().sum().sum())
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_cell_count": missing,
        "missing_percent": round(missing / df.size * 100, 2) if df.size else 0.0,
        "duplicate_row_count": int(duplicated_mask(df).sum()),
    }


def compare_data_quality(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Compare basic data-quality metrics before and after cleaning."""
    if not isinstance(before, pd.DataFrame) or not isinstance(after, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")
    old, new = _quality_metrics(before), _quality_metrics(after)
    change = {
        # Round so float subtraction doesn't leak noise like -16.659999999999997.
        key: round(new[key] - old[key], 2) if isinstance(new[key], float) else new[key] - old[key]
        for key in old
    }
    return {"before": old, "after": new, "change": change}
