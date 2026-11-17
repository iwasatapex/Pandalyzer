
import pandas as pd
from autoclean.validators import validate_missing_values


def test_flags_columns_above_missing_threshold():
    df = pd.DataFrame({
        "customer": ["Alice", None, None, None, "Eve"],
        "amount": [100, 200, 300, 400, 500],
    })

    result = validate_missing_values(df, threshold=50)

    assert result == ["customer"]


def test_does_not_flag_column_at_exact_threshold():
    df = pd.DataFrame({
        "customer": ["Alice", None, "Charlie", None],
        "amount": [100, 200, 300, 400],
    })

    result = validate_missing_values(df, threshold=50)

    assert result == []


def test_does_not_flag_column_at_exact_threshold():
    df = pd.DataFrame({
        "customer": ["Alice", None, "Charlie", None],
        "amount": [100, 200, 300, 400],
    })

    result = validate_missing_values(df, threshold=50)

    assert result == []

import pytest


def test_rejects_threshold_above_100():
    df = pd.DataFrame({"customer": ["Alice", None]})

    with pytest.raises(ValueError, match="Threshold must be between 0 and 100"):
        validate_missing_values(df, threshold=110)


def test_rejects_negative_threshold():
    df = pd.DataFrame({"customer": ["Alice", None]})

    with pytest.raises(ValueError, match="Threshold must be between 0 and 100"):
        validate_missing_values(df, threshold=-1)


def test_handles_empty_dataframe():
    df = pd.DataFrame(columns=["customer", "amount"])

    result = validate_missing_values(df, threshold=50)

    assert result == []



def test_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        validate_missing_values([1, None, 3], threshold=50)


def test_does_not_modify_original_dataframe():
    df = pd.DataFrame({
        "customer": ["Alice", None, None, None, "Eve"],
        "amount": [100, 200, 300, 400, 500],
    })
    original = df.copy(deep=True)

    validate_missing_values(df, threshold=50)

    pd.testing.assert_frame_equal(df, original)


def test_flags_any_missing_value_when_threshold_is_zero():
    df = pd.DataFrame({
        "customer": ["Alice", None, "Charlie"],
        "amount": [100, 200, 300],
        "region": ["North", "South", "East"],
    })

    result = validate_missing_values(df, threshold=0)

    assert result == ["customer"]


def test_does_not_flag_columns_at_100_percent_threshold():
    df = pd.DataFrame({
        "customer": [None, None, None],
        "amount": [100, None, 300],
    })

    result = validate_missing_values(df, threshold=100)

    assert result == []


def test_does_not_flag_columns_at_100_percent_threshold():
    df = pd.DataFrame({
        "customer": [None, None, None],
        "amount": [100, None, 300],
    })

    result = validate_missing_values(df, threshold=100)

    assert result == []

# --- Pandalyzer extension tests: schema and range validation ---
import pandas as pd
import pytest
from autoclean.validators import validate_required_columns, validate_column_types, validate_numeric_ranges

def test_validate_required_columns_returns_missing_names():
    assert validate_required_columns(pd.DataFrame({"id": [1]}), ["id", "status"]) == ["status"]

def test_validate_required_columns_rejects_non_dataframe():
    with pytest.raises(TypeError): validate_required_columns([], ["id"])

def test_validate_column_types_reports_mismatch_and_missing_column():
    df = pd.DataFrame({"amount": [2.5]})
    result = validate_column_types(df, {"amount": "object", "status": "string"})
    assert result["status"] == "missing column"
    assert result["amount"] == "expected object, found float64"

def test_validate_column_types_accepts_matching_dtype():
    assert validate_column_types(pd.DataFrame({"x": pd.Series([1], dtype="int64")}), {"x": "int64"}) == {}

def test_validate_numeric_ranges_returns_out_of_range_index_labels():
    df = pd.DataFrame({"amount": [-1, 10, 120]}, index=[10, 20, 30])
    assert validate_numeric_ranges(df, {"amount": (0, 100)}) == {"amount": [10, 30]}

def test_validate_numeric_ranges_ignores_missing_values():
    assert validate_numeric_ranges(pd.DataFrame({"amount": [None, 10.0]}), {"amount": (0, 100)}) == {}

def test_validate_numeric_ranges_rejects_non_numeric_column():
    with pytest.raises(TypeError): validate_numeric_ranges(pd.DataFrame({"x": ["bad"]}), {"x": (0, 10)})

def test_validate_numeric_ranges_rejects_invalid_bounds():
    with pytest.raises(ValueError): validate_numeric_ranges(pd.DataFrame({"x": [1]}), {"x": (10, 0)})


# --- Integrated additional validator coverage tests ---
def test_required_columns_rejects_non_list():
    with pytest.raises(TypeError, match="required_columns"):
        validate_required_columns(pd.DataFrame({"x": [1]}), "x")


def test_column_types_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_column_types([], {})


def test_column_types_rejects_non_dictionary():
    with pytest.raises(TypeError, match="expected_types"):
        validate_column_types(pd.DataFrame({"x": [1]}), [])


def test_column_types_rejects_invalid_dtype():
    with pytest.raises(ValueError, match="Invalid expected dtype"):
        validate_column_types(pd.DataFrame({"x": [1]}), {"x": "not_a_real_pandas_dtype"})


def test_numeric_ranges_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_numeric_ranges([], {})


def test_numeric_ranges_rejects_non_dictionary():
    with pytest.raises(TypeError, match="ranges"):
        validate_numeric_ranges(pd.DataFrame({"x": [1]}), [])


def test_numeric_ranges_rejects_missing_column():
    with pytest.raises(KeyError, match="Column not found"):
        validate_numeric_ranges(pd.DataFrame({"x": [1]}), {"missing": (0, 10)})


@pytest.mark.parametrize("bounds", [(0,), [0, 10], (0, 1, 2)])
def test_numeric_ranges_rejects_malformed_bounds(bounds):
    with pytest.raises(ValueError, match="range must be"):
        validate_numeric_ranges(pd.DataFrame({"x": [1]}), {"x": bounds})


def test_numeric_ranges_checks_minimum_only():
    df = pd.DataFrame({"x": [-1, 0, 5, None]})
    assert validate_numeric_ranges(df, {"x": (0, None)}) == {"x": [0]}


def test_numeric_ranges_checks_maximum_only():
    df = pd.DataFrame({"x": [1, 5, 11, None]})
    assert validate_numeric_ranges(df, {"x": (None, 10)}) == {"x": [2]}


def test_numeric_ranges_with_no_bounds_finds_no_violations():
    df = pd.DataFrame({"x": [-1, 5, 11, None]})
    assert validate_numeric_ranges(df, {"x": (None, None)}) == {}


def test_numeric_ranges_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="Minimum exceeds maximum"):
        validate_numeric_ranges(pd.DataFrame({"x": [5]}), {"x": (10, 0)})


def test_numeric_ranges_rejects_nonnumeric_column():
    with pytest.raises(TypeError, match="numeric dtype"):
        validate_numeric_ranges(pd.DataFrame({"x": ["bad"]}), {"x": (0, 10)})


def test_numeric_ranges_no_violations_returns_empty_dict():
    assert validate_numeric_ranges(
        pd.DataFrame({"x": [2, 5, 8]}),
        {"x": (0, 10)},
    ) == {}
