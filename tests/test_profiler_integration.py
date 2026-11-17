
from pathlib import Path

import pandas as pd

from autoclean.profiler import profile_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "database" / "messy_retail_orders.csv"


def test_profiler_on_real_dataset():
    # Verify that the dataset exists.
    assert DATASET_PATH.is_file(), (
        f"Dataset not found: {DATASET_PATH}"
    )

    # Load the real dataset.
    df = pd.read_csv(DATASET_PATH)
    original = df.copy(deep=True)

    # Generate the profile.
    report = profile_dataframe(df)

    # Verify overall dataset metrics.
    assert report["row_count"] == len(df)
    assert report["column_count"] == len(df.columns)
    assert report["duplicate_row_count"] == int(df.duplicated().sum())

    columns = report["columns"]

    # Verify column count and column order.
    assert len(columns) == len(df.columns)
    assert columns.index.tolist() == df.columns.tolist()

    # Verify reported data types.
    assert columns["data_type"].tolist() == (
        df.dtypes.astype(str).tolist()
    )

    # Verify missing-value metrics.
    assert columns["missing_count"].tolist() == (
        df.isna().sum().tolist()
    )

    expected_missing_percent = (
        (df.isna().mean() * 100).round(2).tolist()
    )

    assert columns["missing_percent"].tolist() == (
        expected_missing_percent
    )

    # Verify distinct-value counts.
    assert columns["distinct_count"].tolist() == (
        df.nunique(dropna=True).tolist()
    )

    # Verify that metrics fall within valid bounds.
    assert columns["missing_count"].between(
        0, len(df)
    ).all()

    assert columns["missing_percent"].between(
        0, 100
    ).all()

    assert columns["distinct_count"].between(
        0, len(df)
    ).all()

    # Verify profiling did not modify the original DataFrame.
    pd.testing.assert_frame_equal(df, original)


# --- Pandalyzer end-to-end workflow tests ---
def test_csv_to_cleaned_export_and_quality_report(tmp_path):
    import json
    import pandas as pd

    from autoclean.cleaner import clean_dataframe
    from autoclean.connectors import load_csv, save_csv, save_quality_report
    from autoclean.profiler import compare_data_quality, profile_dataframe
    from autoclean.validators import (
        validate_missing_values,
        validate_numeric_ranges,
        validate_required_columns,
    )

    # Start with realistic but deliberately imperfect order data.
    source = pd.DataFrame(
        {
            "order_id": [101, 102, 102, 103],
            "amount": [10.0, None, None, 30.0],
            "status": ["paid", "pending", "pending", "paid"],
        }
    )
    input_path = tmp_path / "messy_orders.csv"
    source.to_csv(input_path, index=False)

    # 1. Load the CSV and profile the original data.
    loaded = load_csv(input_path)
    before_profile = profile_dataframe(loaded)
    assert before_profile["row_count"] == 4
    assert before_profile["duplicate_row_count"] == 1
    assert before_profile["columns"].loc["amount", "missing_count"] == 2

    # 2. Validate the required schema and inspect missingness.
    assert validate_required_columns(
        loaded, ["order_id", "amount", "status"]
    ) == []
    assert validate_missing_values(loaded, threshold=20) == ["amount"]

    # 3. Clean duplicate rows, then impute the remaining numeric missing value.
    cleaned = clean_dataframe(
        loaded,
        remove_duplicates=True,
        missing_strategy="fill_median",
    )
    assert len(cleaned) == 3
    assert cleaned["amount"].isna().sum() == 0
    assert cleaned["amount"].tolist() == [10.0, 20.0, 30.0]

    # 4. Verify the cleaned data and compare before/after quality metrics.
    assert validate_numeric_ranges(cleaned, {"amount": (0, 100)}) == {}
    after_profile = profile_dataframe(cleaned)
    comparison = compare_data_quality(loaded, cleaned)
    assert after_profile["duplicate_row_count"] == 0
    assert comparison["before"]["missing_cell_count"] == 2
    assert comparison["after"]["missing_cell_count"] == 0
    assert comparison["change"]["row_count"] == -1

    # 5. Export the cleaned CSV and a JSON quality report.
    cleaned_path = save_csv(cleaned, tmp_path / "exports" / "cleaned_orders.csv")
    report_path = save_quality_report(
        {
            "before_profile": before_profile,
            "after_profile": after_profile,
            "comparison": comparison,
            "validation": {
                "missing_columns": validate_required_columns(
                    cleaned, ["order_id", "amount", "status"]
                ),
                "out_of_range_rows": validate_numeric_ranges(
                    cleaned, {"amount": (0, 100)}
                ),
            },
        },
        tmp_path / "reports" / "quality_report.json",
    )

    exported = load_csv(cleaned_path)
    assert len(exported) == 3
    assert exported["amount"].tolist() == [10.0, 20.0, 30.0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["comparison"]["after"]["missing_cell_count"] == 0
    assert report["validation"]["missing_columns"] == []
    assert report["validation"]["out_of_range_rows"] == {}
