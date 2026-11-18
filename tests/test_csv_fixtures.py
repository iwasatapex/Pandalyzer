"""Run the whole project end to end on 10 CSV files with different schemas and dtypes.

The files live in ``tests/data`` and are produced by ``tests/csv_fixtures.py``,
which also records the ground truth checked here. Every test runs once per file.
"""

import json
import warnings

import pandas as pd
import pytest

from autoclean import (
    auto_clean,
    clean_dataframe,
    compare_data_quality,
    handle_missing_values,
    load_csv,
    profile_dataframe,
    save_csv,
    save_quality_report,
    validate_column_types,
    validate_emails,
    validate_missing_values,
    validate_numeric_ranges,
    validate_required_columns,
)
from tests.csv_fixtures import DATA_DIR, FIXTURES, CsvFixture


def _kind(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return "text"
    return str(dtype)


def _load(fixture: CsvFixture, path=None) -> pd.DataFrame:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_csv(path or DATA_DIR / fixture.file_name, **fixture.load_kwargs)
    parser_warnings = [w for w in caught if issubclass(w.category, pd.errors.ParserWarning)]
    assert bool(parser_warnings) == fixture.expect_parser_warning, [str(w.message) for w in caught]
    return df


def _clean(fixture: CsvFixture, df: pd.DataFrame, **overrides):
    return auto_clean(df, **{**fixture.clean_kwargs, **overrides})


@pytest.fixture(params=FIXTURES, ids=[fixture.file_name for fixture in FIXTURES])
def case(request) -> CsvFixture:
    return request.param


@pytest.fixture
def raw(case) -> pd.DataFrame:
    return _load(case)


@pytest.fixture
def cleaned(case, raw):
    return _clean(case, raw, return_log=True)


# ---------------------------------------------------------------------------


def test_fixture_files_match_generator(case):
    on_disk = case.payload((DATA_DIR / case.file_name).read_bytes())
    assert on_disk == case.payload(case.to_bytes()), (
        "tests/data is out of date: run `python -m tests.csv_fixtures`"
    )


def test_load_detects_format_and_schema(case, raw):
    assert raw.shape == (case.raw_rows, len(case.raw_columns))
    assert raw.columns.tolist() == case.raw_columns
    assert raw.attrs.get("preamble", []) == case.preamble


def test_profile_of_raw_data_is_consistent(raw):
    profile = profile_dataframe(raw)
    columns = profile["columns"]

    assert profile["row_count"] == len(raw)
    assert profile["column_count"] == raw.shape[1]
    assert columns["missing_count"].tolist() == raw.isna().sum().tolist()
    assert columns["missing_percent"].between(0, 100).all()
    assert (columns["distinct_count"] <= len(raw)).all()


def test_auto_clean_gives_expected_dtypes(case, cleaned):
    result, _ = cleaned
    kinds = {column: _kind(result[column]) for column in result.columns}

    assert kinds == case.clean_kinds


def test_auto_clean_drops_only_duplicates_and_empty_rows(case, raw, cleaned):
    result, log = cleaned

    assert len(result) == case.clean_rows
    assert set(raw.index) - set(result.index) == case.dropped
    assert not result.isna().all(axis=1).any()
    assert not result.duplicated().any()
    assert all(entry["step"] for entry in log)


def test_auto_clean_produces_exact_values(case, cleaned):
    result, _ = cleaned
    for (label, column), expected in case.checks.items():
        actual = result.loc[label, column]
        if expected is None or expected is pd.NA:
            assert pd.isna(actual), f"{column}[{label}] = {actual!r}, expected missing"
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected), f"{column}[{label}]"
        else:
            assert actual == expected, f"{column}[{label}] = {actual!r}, expected {expected!r}"


def test_auto_clean_does_not_modify_loaded_frame(case, raw):
    before = raw.copy()

    _clean(case, raw, missing="auto")

    pd.testing.assert_frame_equal(raw, before)


def test_validators_on_cleaned_data(case, cleaned):
    result, _ = cleaned

    assert validate_required_columns(result, list(case.clean_kinds)) == []
    expected_types = {column: str(result[column].dtype) for column in result.columns}
    assert validate_column_types(result, expected_types) == {}
    flagged = validate_missing_values(result, threshold=0)
    assert flagged == [c for c in result.columns if result[c].isna().any()]

    for column, (low, high, violations) in case.ranges.items():
        problems = validate_numeric_ranges(result, {column: (low, high)})
        assert len(problems.get(column, [])) == violations, column


def test_email_validation_before_and_after_cleaning(case, raw, cleaned):
    result, _ = cleaned

    flagged = validate_emails(raw)

    assert {column: len(rows) for column, rows in flagged.items()} == case.invalid_emails
    assert validate_emails(result, list(case.invalid_emails)) == {}


def test_every_missing_value_strategy(case, cleaned):
    result, _ = cleaned
    numeric = [
        c
        for c in result.columns
        if _kind(result[c]) in ("int", "float") and result[c].notna().any()
    ]

    assert not handle_missing_values(result, "drop_rows").isna().any().any()
    filled = handle_missing_values(result, "fill_constant", 0)
    assert not filled.isna().any().any()
    for strategy in ("fill_mean", "fill_median"):
        if numeric:
            assert not handle_missing_values(result, strategy)[numeric].isna().any().any()
        else:
            with pytest.raises(ValueError):
                handle_missing_values(result, strategy)

    auto = _clean(case, result, missing="auto")
    assert not auto[numeric].isna().any().any()
    assert len(clean_dataframe(result, missing_strategy="fill_constant", fill_value="x")) == len(
        result
    )


def test_clean_save_reload_round_trip(case, cleaned, tmp_path):
    result, _ = cleaned
    path = save_csv(result, tmp_path / "clean.csv")

    reloaded = _clean(case, _load_plain(case, path))

    assert reloaded.columns.tolist() == result.columns.tolist()
    assert {c: _kind(reloaded[c]) for c in reloaded.columns} == case.clean_kinds
    pd.testing.assert_frame_equal(
        reloaded.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
    )


def _load_plain(case: CsvFixture, path) -> pd.DataFrame:
    # A file we saved ourselves is clean UTF-8 CSV: no parser warnings expected.
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        return load_csv(path, **case.load_kwargs)


def test_auto_clean_is_idempotent(case, cleaned):
    result, _ = cleaned

    again, log = _clean(case, result, return_log=True)

    pd.testing.assert_frame_equal(again, result)
    assert log == []


def test_quality_report_is_strict_json(case, raw, cleaned, tmp_path):
    result, log = cleaned
    comparison = compare_data_quality(raw, result)

    path = save_quality_report(
        {
            "file": case.file_name,
            "raw_profile": profile_dataframe(raw),
            "clean_profile": profile_dataframe(result),
            "comparison": comparison,
            "log": log,
            "sample": result.head(3),
        },
        tmp_path / "report.json",
    )

    report = json.loads(path.read_text(encoding="utf-8"), parse_constant=pytest.fail)
    assert report["comparison"]["change"]["row_count"] == case.clean_rows - case.raw_rows
    assert set(report["clean_profile"]["columns"]) == set(case.clean_kinds)
    assert len(report["sample"]) == min(3, case.clean_rows)
