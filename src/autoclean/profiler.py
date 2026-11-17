import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Generate a read-only data quality profile."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    column_report = pd.DataFrame(
        {
            "data_type": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
            "distinct_count": df.nunique(dropna=True),
        }
    )

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
        "columns": column_report,
    }

# --- Pandalyzer extension: quality comparison ---
def compare_data_quality(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Compare basic data-quality metrics before and after cleaning."""
    if not isinstance(before, pd.DataFrame) or not isinstance(after, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")
    def metrics(df: pd.DataFrame) -> dict:
        missing = int(df.isna().sum().sum())
        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "missing_cell_count": missing,
            "missing_percent": round(missing / df.size * 100, 2) if df.size else 0.0,
            "duplicate_row_count": int(df.duplicated().sum()),
        }
    old, new = metrics(before), metrics(after)
    return {"before": old, "after": new, "change": {key: new[key] - old[key] for key in old}}
