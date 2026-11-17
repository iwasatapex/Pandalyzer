
import pandas as pd


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows while preserving the original DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return df.drop_duplicates().copy()


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop_rows",
    value=None,
) -> pd.DataFrame:
    """Handle missing values using the selected strategy."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if strategy == "drop_rows":
        return df.dropna().copy()

    if strategy == "fill_constant":
        return df.fillna(value=value).copy()
     
    if strategy == "fill_mean":
        numeric_columns = df.select_dtypes(include="number").columns

        if len(numeric_columns) == 0:
            raise ValueError(
                "Mean imputation requires at least one numeric column."
            )

        means = df[numeric_columns].mean()
        return df.fillna(means).copy()
      
    if strategy == "fill_median":
        numeric_columns = df.select_dtypes(include="number").columns

        if len(numeric_columns) == 0:
            raise ValueError(
                "Median imputation requires at least one numeric column."
            )

        medians = df[numeric_columns].median()
        return df.fillna(medians).copy()



    raise ValueError(f"Unsupported missing-value strategy: {strategy}")

# --- Pandalyzer extension: end-to-end cleaning helper ---
def clean_dataframe(df: pd.DataFrame, *, remove_duplicates: bool = True, missing_strategy: str = "drop_rows", fill_value=None) -> pd.DataFrame:
    """Apply the existing duplicate and missing-value cleaners in sequence."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    result = df.copy()
    if remove_duplicates:
        result = remove_duplicate_rows(result)
    return handle_missing_values(result, strategy=missing_strategy, value=fill_value)
