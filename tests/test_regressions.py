"""Regression tests for bugs found during the code audit."""

import json

import numpy as np
import pandas as pd
import pytest

from autoclean.cleaner import clean_dataframe, handle_missing_values
from autoclean.connectors import save_quality_report
from autoclean.profiler import compare_data_quality, profile_dataframe
from autoclean.validators import validate_column_types


def _strict_json_load(path):
    def reject(token):
        raise ValueError(f"non-standard JSON token: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.mark.parametrize("strategy, expected", [("fill_mean", 1.5), ("fill_median", 1.5)])
def test_statistic_fill_handles_nullable_integer_with_fractional_result(strategy, expected):
    df = pd.DataFrame({"qty": pd.array([1, 2, None], dtype="Int64")})

    result = handle_missing_values(df, strategy=strategy)

    assert result["qty"].tolist() == [1.0, 2.0, expected]
    assert not result["qty"].isna().any()


def test_statistic_fill_keeps_nullable_integer_when_result_is_whole():
    df = pd.DataFrame({"qty": pd.array([1, 3, None], dtype="Int64")})

    result = handle_missing_values(df, strategy="fill_mean")

    assert str(result["qty"].dtype) == "Int64"
    assert result["qty"].tolist() == [1, 3, 2]


def test_statistic_fill_leaves_non_numeric_and_all_missing_columns():
    df = pd.DataFrame(
        {
            "amount": [10.0, None, 30.0],
            "empty": [np.nan, np.nan, np.nan],
            "name": ["a", None, "c"],
        }
    )

    result = handle_missing_values(df, strategy="fill_mean")

    assert result["amount"].tolist() == [10.0, 20.0, 30.0]
    assert result["empty"].isna().all()
    assert pd.isna(result.loc[1, "name"])


def test_statistic_fill_does_not_modify_original():
    df = pd.DataFrame({"qty": pd.array([1, 2, None], dtype="Int64")})
    original = df.copy(deep=True)

    handle_missing_values(df, strategy="fill_mean")

    pd.testing.assert_frame_equal(df, original)


def test_fill_constant_requires_a_value():
    df = pd.DataFrame({"a": [1, None]})

    with pytest.raises(ValueError, match="requires a fill value"):
        handle_missing_values(df, strategy="fill_constant")

    with pytest.raises(ValueError, match="requires a fill value"):
        clean_dataframe(df, missing_strategy="fill_constant")


def test_unsupported_strategy_error_lists_valid_options():
    with pytest.raises(ValueError, match="fill_median"):
        handle_missing_values(pd.DataFrame({"a": [1]}), strategy="nope")


def test_profile_of_empty_frame_reports_zero_missing_percent():
    profile = profile_dataframe(pd.DataFrame({"a": [], "b": []}))

    assert profile["columns"]["missing_percent"].tolist() == [0.0, 0.0]


def test_quality_report_is_strict_json_for_nan_inf_and_nat(tmp_path):
    path = save_quality_report(
        {
            "nan": np.float64("nan"),
            "inf": float("inf"),
            "nat": pd.NaT,
            "frame": pd.DataFrame({"pct": [np.nan, 50.0]}, index=["a", "b"]),
            "series": pd.Series([np.nan], index=["x"]),
        },
        tmp_path / "report.json",
    )

    assert _strict_json_load(path) == {
        "nan": None,
        "inf": None,
        "nat": None,
        "frame": {"a": {"pct": None}, "b": {"pct": 50.0}},
        "series": {"x": None},
    }


def test_quality_report_serialises_timestamps_as_iso_strings(tmp_path):
    path = save_quality_report(
        {
            "ts": pd.Timestamp("2024-01-02 03:04:05"),
            "np_ts": np.datetime64("2024-01-02T03:04:05"),
        },
        tmp_path / "ts.json",
    )

    assert _strict_json_load(path) == {
        "ts": "2024-01-02T03:04:05",
        "np_ts": "2024-01-02T03:04:05",
    }


def test_profile_of_empty_frame_round_trips_through_report(tmp_path):
    path = save_quality_report(profile_dataframe(pd.DataFrame({"a": []})), tmp_path / "p.json")

    assert _strict_json_load(path)["columns"]["a"]["missing_percent"] == 0.0


def test_quality_report_rejects_non_dict(tmp_path):
    with pytest.raises(TypeError):
        save_quality_report([1, 2], tmp_path / "bad.json")


def test_compare_data_quality_change_has_no_float_noise():
    before = pd.DataFrame({"a": [None, 1, 1]})
    after = pd.DataFrame({"a": [None, 1, 1, 1, 1, 1]})

    change = compare_data_quality(before, after)["change"]

    assert change["missing_percent"] == -16.66
    assert repr(change["missing_percent"]) == "-16.66"


def test_invalid_dtype_spec_raises_even_when_column_missing():
    with pytest.raises(ValueError, match="Invalid expected dtype"):
        validate_column_types(pd.DataFrame({"a": [1]}), {"b": "not_a_dtype"})
