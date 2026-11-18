import pandas as pd

from autoclean.emails import email_columns, is_valid_email


def validate_missing_values(
    df: pd.DataFrame,
    threshold: float = 50,
) -> list[str]:
    """Return column names whose missing-value percentage exceeds the threshold."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100.")

    missing_percent = df.isna().mean() * 100

    return missing_percent[missing_percent > threshold].index.tolist()


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Return required column names that are absent from the DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(required_columns, list):
        raise TypeError("required_columns must be a list of column names.")
    return [column for column in required_columns if column not in df.columns]


def validate_column_types(df: pd.DataFrame, expected_types: dict[str, object]) -> dict[str, str]:
    """Return missing columns or columns whose dtype differs from the expected dtype."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(expected_types, dict):
        raise TypeError("expected_types must be a dictionary.")
    # Resolve every expected dtype first so an invalid spec always raises,
    # even when its column happens to be missing from the DataFrame.
    resolved = {}
    for column, expected in expected_types.items():
        try:
            resolved[column] = pd.api.types.pandas_dtype(expected)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid expected dtype for column {column!r}: {expected!r}") from exc

    problems = {}
    for column, expected_dtype in resolved.items():
        if column not in df.columns:
            problems[column] = "missing column"
            continue
        if not pd.api.types.is_dtype_equal(df[column].dtype, expected_dtype):
            problems[column] = f"expected {expected_dtype}, found {df[column].dtype}"
    return problems


def validate_numeric_ranges(
    df: pd.DataFrame, ranges: dict[str, tuple[float | None, float | None]]
) -> dict[str, list]:
    """Return index labels of non-missing numeric values outside inclusive bounds."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(ranges, dict):
        raise TypeError("ranges must be a dictionary.")
    problems = {}
    for column, bounds in ranges.items():
        if column not in df.columns:
            raise KeyError(f"Column not found: {column}")
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError("Each range must be a (minimum, maximum) tuple.")
        minimum, maximum = bounds
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"Minimum exceeds maximum for column {column!r}.")
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column {column!r} must have a numeric dtype.")
        values = df[column]
        present = values.notna()
        mask = pd.Series(False, index=df.index)
        if minimum is not None:
            mask |= present & (values < minimum)
        if maximum is not None:
            mask |= present & (values > maximum)
        if mask.any():
            problems[column] = df.index[mask].tolist()
    return problems


def validate_emails(df: pd.DataFrame, columns: list | None = None) -> dict[str, list]:
    """Return index labels of malformed email addresses, per column.

    ``columns`` defaults to every column that looks like an email column.
    Surrounding spaces, letter case, ``mailto:`` and display names
    (``"Ann <ann@x.com>"``) are tolerated; missing values are not flagged.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if columns is None:
        columns = email_columns(df)
    problems = {}
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"Column not found: {column}")
        values = df[column]
        invalid = values.notna() & ~values.map(is_valid_email).astype(bool)
        if invalid.any():
            problems[column] = df.index[invalid].tolist()
    return problems
