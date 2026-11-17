import pandas as pd
import pandas.testing as pdt
import pytest

from autoclean.profiler import compare_data_quality, profile_dataframe


def test_profile_counts_and_column_metrics():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Alice"],
        "age": [25, None, 25],
    })

    report = profile_dataframe(df)

    assert report["row_count"] == 3
    assert report["column_count"] == 2
    assert report["duplicate_row_count"] == 1

    columns = report["columns"]
    assert columns.loc["age", "missing_count"] == 1
    assert columns.loc["age", "missing_percent"] == pytest.approx(33.33)
    assert columns.loc["name", "distinct_count"] == 2


def test_profiler_does_not_modify_input():
    df = pd.DataFrame({
        "name": ["Alice", None, "Bob"],
        "age": [25, 30, 35],
    })
    original = df.copy(deep=True)

    profile_dataframe(df)

    pdt.assert_frame_equal(df, original)


def test_profiler_rejects_non_dataframe():
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        profile_dataframe([1, 2, 3])

def test_profiler_handles_empty_dataframe():
    df = pd.DataFrame()

    report = profile_dataframe(df)

    assert report["row_count"] == 0
    assert report["column_count"] == 0
    assert report["duplicate_row_count"] == 0
    assert report["columns"].empty

def test_profiler_handles_empty_dataframe_with_columns():
    df = pd.DataFrame(columns=["name", "age", "city"])

    report = profile_dataframe(df)

    assert report["row_count"] == 0
    assert report["column_count"] == 3
    assert report["duplicate_row_count"] == 0

    columns = report["columns"]

    assert list(columns.index) == ["name", "age", "city"]
    assert (columns["missing_count"] == 0).all()
    assert (columns["distinct_count"] == 0).all()

def test_profiler_handles_entirely_missing_column():
    df = pd.DataFrame({
        "customer": ["Alice", "Bob", "Charlie", "David"],
        "discount": [None, None, None, None],
    })

    report = profile_dataframe(df)
    columns = report["columns"]

    assert columns.loc["discount", "missing_count"] == 4
    assert columns.loc["discount", "missing_percent"] == 100.0
    assert columns.loc["discount", "distinct_count"] == 0

    assert columns.loc["customer", "missing_count"] == 0

def test_profiler_counts_extra_duplicate_occurrences():
    df = pd.DataFrame({
        "order_id": [101, 101, 101, 102, 103, 103],
        "amount": [50, 50, 50, 75, 90, 90],
    })

    report = profile_dataframe(df)

    assert report["row_count"] == 6
    assert report["duplicate_row_count"] == 3

def test_profiler_excludes_missing_values_from_distinct_count():
    df = pd.DataFrame({
        "category": ["A", "A", "B", None, None],
    })

    report = profile_dataframe(df)

    columns = report["columns"]

    assert columns.loc["category", "missing_count"] == 2
    assert columns.loc["category", "missing_percent"] == 40.0
    assert columns.loc["category", "distinct_count"] == 2

def test_profiler_reports_column_data_types():
    df = pd.DataFrame({
        "customer": ["Alice", "Bob"],
        "quantity": pd.Series([2, 5], dtype="int64"),
        "price": pd.Series([10.5, 20.0], dtype="float64"),
        "active": [True, False],
    })

    report = profile_dataframe(df)
    columns = report["columns"]

    assert columns.loc["customer", "data_type"] in {"object","str","string"}
    assert columns.loc["quantity", "data_type"] == "int64"
    assert columns.loc["price", "data_type"] == "float64"
    assert columns.loc["active", "data_type"] == "bool"

def test_profiler_calculates_missing_percentages_per_column():
    df = pd.DataFrame({
        "a": [1, None, 3, None],
        "b": [10, 20, None, 40],
        "c": [None, None, None, None],
    })

    report = profile_dataframe(df)
    columns = report["columns"]

    assert columns.loc["a", "missing_percent"] == 50.0
    assert columns.loc["b", "missing_percent"] == 25.0
    assert columns.loc["c", "missing_percent"] == 100.0

def test_profiler_handles_non_sequential_index():
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice"],
            "amount": [100, 200, 100],
        },
        index=[10, 25, 90],
    )

    report = profile_dataframe(df)

    assert report["row_count"] == 3
    assert report["column_count"] == 2
    assert report["duplicate_row_count"] == 1

@pytest.mark.parametrize(
    "invalid_input",
    [
        [1, 2, 3],
        {"name": ["Alice", "Bob"]},
        "not a dataframe",
        None,
        123,
    ],
)
def test_profiler_rejects_invalid_input_types(invalid_input):
    with pytest.raises(
        TypeError,
        match="Input must be a pandas DataFrame",
    ):
        profile_dataframe(invalid_input)

def test_profiler_handles_mixed_missing_values():
    df = pd.DataFrame({
        "value": [1.0, None, float("nan"), 4.0],
    })

    report = profile_dataframe(df)
    columns = report["columns"]

    assert columns.loc["value", "missing_count"] == 2
    assert columns.loc["value", "missing_percent"] == 50.0
    assert columns.loc["value", "distinct_count"] == 2

def test_profiler_handles_unusual_column_names():
    df = pd.DataFrame({
        "": [1, 2],
        "order id": [101, 102],
        "amount ($)": [10.5, 20.5],
    })

    report = profile_dataframe(df)
    columns = report["columns"]

    assert report["column_count"] == 3
    assert "" in columns.index
    assert "order id" in columns.index
    assert "amount ($)" in columns.index

def test_profiler_handles_duplicate_column_names():
    df = pd.DataFrame(
        [[10, 20], [30, 40]],
        columns=["amount", "amount"],
    )

    report = profile_dataframe(df)

    assert report["row_count"] == 2
    assert report["column_count"] == 2

def test_profiler_missing_counts_are_exact():
    df = pd.DataFrame({
        "a": [None, 1, None, 2, None],
        "b": [10, 20, 30, 40, 50],
    })

    report = profile_dataframe(df)
    columns = report["columns"]

    assert columns.loc["a", "missing_count"] == 3
    assert columns.loc["a", "missing_percent"] == 60.0
    assert columns.loc["b", "missing_count"] == 0
    assert columns.loc["b", "missing_percent"] == 0.0

def test_repeated_profiling_does_not_modify_input():
    df = pd.DataFrame({
        "product": ["A", None, "B"],
        "price": [10.5, 20.0, None],
    })

    original = df.copy(deep=True)

    first_report = profile_dataframe(df)
    second_report = profile_dataframe(df)

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(
        first_report["columns"],
        second_report["columns"],
    )

    assert first_report["row_count"] == second_report["row_count"]
    assert (
        first_report["duplicate_row_count"]
        == second_report["duplicate_row_count"]
    )

def test_profiler_preserves_metrics_for_duplicate_column_names():
    df = pd.DataFrame(
        [[10, None], [20, 30], [None, 40]],
        columns=["amount", "amount"],
    )

    report = profile_dataframe(df)

    assert report["column_count"] == 2

    columns = report["columns"]

    assert len(columns) == 2
    assert columns["missing_count"].tolist() == [1, 1]
    assert columns["missing_percent"].tolist() == [33.33, 33.33]


# --- Pandalyzer extension tests: quality comparison ---

def test_compare_data_quality_reports_changes():
    result = compare_data_quality(pd.DataFrame({"x": [1, None, 1]}), pd.DataFrame({"x": [1, 2]}))
    assert result["before"]["row_count"] == 3
    assert result["after"]["row_count"] == 2
    assert result["change"]["missing_cell_count"] == -1

def test_compare_data_quality_handles_empty_frames():
    assert compare_data_quality(pd.DataFrame(), pd.DataFrame())["before"]["missing_percent"] == 0.0

def test_compare_data_quality_rejects_non_dataframes():
    with pytest.raises(TypeError): compare_data_quality([], pd.DataFrame())
