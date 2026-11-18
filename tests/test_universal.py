"""Tests for schema-agnostic loading, type inference and automatic cleaning."""

import bz2
import gzip
import json
import warnings
import zipfile

import pandas as pd
import pytest

from autoclean import (
    auto_clean,
    clean_column_names,
    compare_data_quality,
    fill_missing_auto,
    handle_missing_values,
    infer_column_types,
    infer_series_type,
    load_csv,
    profile_dataframe,
    remove_duplicate_rows,
    save_quality_report,
    standardize_missing_values,
    strip_whitespace,
    unify_case_variants,
    validate_missing_values,
)
from autoclean.connectors import detect_delimiter, detect_encoding

# ---------------------------------------------------------------------------
# Loading: delimiters, encodings, compression, malformed rows
# ---------------------------------------------------------------------------

EXPECTED = pd.DataFrame({"id": [1, 2], "city": ["São Paulo", "Zürich"]})


@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_load_csv_detects_delimiter(tmp_path, sep):
    path = tmp_path / "data.csv"
    path.write_text(f"id{sep}city\n1{sep}São Paulo\n2{sep}Zürich\n", encoding="utf-8")

    pd.testing.assert_frame_equal(load_csv(path), EXPECTED)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"])
def test_load_csv_detects_encoding(tmp_path, encoding):
    path = tmp_path / "data.csv"
    path.write_bytes("id,city\n1,São Paulo\n2,Zürich\n".encode(encoding))

    pd.testing.assert_frame_equal(load_csv(path), EXPECTED)


def test_load_csv_falls_back_when_non_utf8_byte_is_past_the_sample(tmp_path):
    path = tmp_path / "late.csv"
    rows = "".join(f"{i},Paris\n" for i in range(30_000))
    path.write_bytes(("id,city\n" + rows + "99,São Paulo\n").encode("cp1252"))

    result = load_csv(path)

    assert result["city"].iloc[-1] == "São Paulo"


@pytest.mark.parametrize(
    "suffix, opener",
    [(".csv.gz", gzip.open), (".csv.bz2", bz2.open)],
)
def test_load_csv_reads_compressed_files(tmp_path, suffix, opener):
    path = tmp_path / f"data{suffix}"
    with opener(path, "wt", encoding="utf-8") as file:
        file.write("id;city\n1;São Paulo\n2;Zürich\n")

    pd.testing.assert_frame_equal(load_csv(path), EXPECTED)


def test_load_csv_reads_single_file_zip(tmp_path):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("inner.csv", "id\tcity\n1\tSão Paulo\n2\tZürich\n")

    pd.testing.assert_frame_equal(load_csv(path), EXPECTED)


def test_load_csv_keeps_rows_with_extra_fields(tmp_path):
    path = tmp_path / "ragged.csv"
    path.write_text("id,name\n1,Ana\n2,Bob,EXTRA\n3,Cy\n", encoding="utf-8")

    with pytest.warns(pd.errors.ParserWarning):
        result = load_csv(path)

    assert result.to_dict(orient="list") == {"id": [1, 2, 3], "name": ["Ana", "Bob", "Cy"]}


def test_load_csv_does_not_shift_columns_when_first_row_has_extra_field(tmp_path):
    path = tmp_path / "ragged.csv"
    path.write_text("id,name\n1,Ana,EXTRA\n2,Bob\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = load_csv(path)

    assert result["id"].tolist() == [1, 2]
    assert result["name"].tolist() == ["Ana", "Bob"]


def test_load_csv_respects_explicit_options(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a;b\n1;2\n", encoding="utf-8")

    assert load_csv(path, sep=",").columns.tolist() == ["a;b"]
    assert load_csv(path, dtype=str)["a"].tolist() == ["1"]


def test_load_csv_whitespace_only_file_is_empty(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("  \n\n", encoding="utf-8")

    with pytest.raises(pd.errors.EmptyDataError):
        load_csv(path)


def test_detect_delimiter_ignores_delimiters_inside_quotes():
    text = 'name;note\n"Smith, J";"a, b, c"\n"Doe, J";"d, e"\n'
    assert detect_delimiter(text) == ";"


def test_detect_delimiter_defaults_to_comma_for_single_column():
    assert detect_delimiter("value\n1\n2\n") == ","


def test_detect_encoding_handles_truncated_multibyte_sample():
    sample = "é".encode()[:1]
    assert detect_encoding(b"abc" + sample) == "utf-8-sig"


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, kind, expected",
    [
        (["1", " 2 ", "3"], "integer", [1, 2, 3]),
        (["1.0", "2.0", None], "integer", [1, 2, pd.NA]),
        (["1.5", "2", "-3"], "float", [1.5, 2.0, -3.0]),
        (["₹1,234.50", "$20", "€3"], "float", [1234.5, 20.0, 3.0]),
        (["15%", "7.5%", "0%"], "float", [15.0, 7.5, 0.0]),
        (["(42)", "10", "5"], "integer", [-42, 10, 5]),
        (["Yes", "no", "Y", "N"], "boolean", [True, False, True, False]),
        (["true", "FALSE", "1", "0"], "boolean", [True, False, True, False]),
    ],
)
def test_infer_series_type_numbers_and_booleans(values, kind, expected):
    result, detected = infer_series_type(pd.Series(values, dtype=object))

    assert detected == kind
    assert [None if pd.isna(v) else v for v in result.tolist()] == [
        None if v is pd.NA else v for v in expected
    ]


def test_infer_series_type_parses_mixed_date_formats():
    series = pd.Series(["2024-02-19 10:16:45", "03-Mar-2024", "07/27/2024 08:35:07", None])

    result, kind = infer_series_type(series)

    assert kind == "datetime"
    assert result.tolist()[:3] == [
        pd.Timestamp("2024-02-19 10:16:45"),
        pd.Timestamp("2024-03-03"),
        pd.Timestamp("2024-07-27 08:35:07"),
    ]
    assert pd.isna(result.iloc[3])


def test_infer_series_type_respects_dayfirst():
    series = pd.Series(["03/04/2024", "25/12/2024"])

    result, kind = infer_series_type(series, dayfirst=True)

    assert kind == "datetime"
    assert result.iloc[0] == pd.Timestamp("2024-04-03")


@pytest.mark.parametrize(
    "values",
    [
        ["00123", "00456", "789"],  # Leading-zero identifiers.
        ["1234567890123456789", "9876543210987654321"],  # Too long for exact floats.
        ["apple", "banana", "cherry"],  # Plain text.
        ["0", "1", "1"],  # Pure 0/1 stays numeric, not boolean.
        ["10.0.0.1", "192.168.1.1"],  # IP addresses are not dates.
        ["1", "2", "x", "y"],  # Below threshold: don't destroy half the data.
    ],
)
def test_infer_series_type_leaves_ambiguous_text_alone(values):
    series = pd.Series(values, dtype=object)

    result, kind = infer_series_type(series)

    if values == ["0", "1", "1"]:
        assert kind == "integer"
    else:
        assert kind is None
        pd.testing.assert_series_equal(result, series)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, 2]),
        pd.Series([1.5, None]),
        pd.Series([True, False]),
        pd.Series(pd.to_datetime(["2024-01-01"])),
        pd.Series(["1", "2"], dtype="category"),
    ],
)
def test_infer_series_type_never_touches_typed_columns(series):
    result, kind = infer_series_type(series)

    assert kind is None
    pd.testing.assert_series_equal(result, series)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["3,5", "1.234,50", "-12,25"], [3.5, 1234.5, -12.25]),
        (["€ 1.200,00", "€ 15,50", "€ 7"], [1200.0, 15.5, 7.0]),
    ],
)
def test_infer_series_type_reads_european_decimal_comma(values, expected):
    result, kind = infer_series_type(pd.Series(values))

    assert kind == "float"
    assert result.tolist() == expected


def test_infer_series_type_keeps_us_reading_when_it_works():
    result, _ = infer_series_type(pd.Series(["1,234", "2,500", "10"]))

    assert result.tolist() == [1234, 2500, 10]


def test_infer_series_type_narrows_whole_number_floats_to_int():
    result, kind = infer_series_type(pd.Series([1.0, None, 3.0]))

    assert kind == "integer"
    assert str(result.dtype) == "Int64"
    assert result.tolist()[0] == 1


def test_infer_series_type_turns_oversized_python_ints_into_text():
    big = 95908301661318609139
    result, kind = infer_series_type(pd.Series([big, None], dtype=object))

    assert kind == "text"
    assert result.iloc[0] == str(big)


def test_dayfirst_does_not_flip_iso_dates():
    series = pd.Series(["2024-03-05", "05.03.2024", "2024-12-01"])

    result, kind = infer_series_type(series, dayfirst=True)

    assert kind == "datetime"
    assert result.tolist() == [pd.Timestamp("2024-03-05")] * 2 + [pd.Timestamp("2024-12-01")]


def test_infer_column_types_does_not_modify_input():
    df = pd.DataFrame({"n": ["1", "2"], "t": ["a", "b"]})
    original = df.copy()

    result = infer_column_types(df)

    pd.testing.assert_frame_equal(df, original)
    assert str(result["n"].dtype) == "Int64"


def test_infer_series_type_rejects_bad_threshold():
    with pytest.raises(ValueError):
        infer_series_type(pd.Series(["1"]), threshold=0)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [object, "string", "str"])
def test_text_steps_work_on_every_text_dtype(dtype):
    df = pd.DataFrame({"v": pd.Series(["  Cash ", "CASH", "\tcash", " N/A ", None], dtype=dtype)})

    result = unify_case_variants(standardize_missing_values(strip_whitespace(df)))

    assert result["v"].tolist()[:3] == ["Cash", "Cash", "Cash"]
    assert result["v"].isna().tolist() == [False, False, False, True, True]


def test_strip_whitespace_leaves_non_strings_in_object_columns():
    df = pd.DataFrame({"v": pd.Series([" a ", 5, [1, 2], None], dtype=object)})

    result = strip_whitespace(df)

    assert result["v"].tolist()[:3] == ["a", 5, [1, 2]]


def test_standardize_missing_values_accepts_custom_markers():
    df = pd.DataFrame({"v": ["ok", "Unknown", "-", "SKIP"]})

    result = standardize_missing_values(df, markers={"skip"})

    assert result["v"].tolist()[:3] == ["ok", "Unknown", "-"]
    assert pd.isna(result["v"].iloc[3])


def test_unify_case_variants_picks_most_common_spelling():
    df = pd.DataFrame({"v": ["Store", "store", "STORE", "store", "Web"]})

    assert unify_case_variants(df)["v"].tolist() == ["store", "store", "store", "store", "Web"]


def test_unify_case_variants_does_not_merge_distinct_words():
    df = pd.DataFrame({"v": ["COD", "Cash on Delivery", "Net Banking", "NetBanking"]})

    assert unify_case_variants(df)["v"].tolist() == df["v"].tolist()


@pytest.mark.parametrize(
    "style, expected",
    [
        ("strip", ["Order ID", "Order ID_2", "column_3", "a"]),
        ("snake", ["order_id", "order_id_2", "column_3", "a"]),
    ],
)
def test_clean_column_names_makes_unique_names(style, expected):
    df = pd.DataFrame([[1, 2, 3, 4]], columns=[" Order ID ", "Order ID", "  ", "a"])

    assert clean_column_names(df, style=style).columns.tolist() == expected


# ---------------------------------------------------------------------------
# Every function on every dtype
# ---------------------------------------------------------------------------

DTYPE_COLUMNS = {
    "str": pd.Series(["a", None, "a"], dtype="str"),
    "object": pd.Series(["a", None, "a"], dtype=object),
    "string": pd.Series(["a", None, "a"], dtype="string"),
    "int64": pd.Series([1, 2, 1], dtype="int64"),
    "Int64": pd.Series([1, None, 1], dtype="Int64"),
    "float64": pd.Series([1.5, None, 1.5]),
    "Float64": pd.Series([1.5, None, 1.5], dtype="Float64"),
    "bool": pd.Series([True, False, True]),
    "boolean": pd.Series([True, None, True], dtype="boolean"),
    "datetime": pd.Series(pd.to_datetime(["2024-01-01", None, "2024-01-01"])),
    "datetime_tz": pd.Series(pd.to_datetime(["2024-01-01", None, "2024-01-01"]).tz_localize("UTC")),
    "timedelta": pd.Series(pd.to_timedelta(["1D", None, "1D"])),
    "category": pd.Series(["a", None, "a"], dtype="category"),
    "lists": pd.Series([[1], None, [1]], dtype=object),
    "mixed_object": pd.Series([1, "x", None], dtype=object),
}


@pytest.fixture(params=list(DTYPE_COLUMNS), ids=list(DTYPE_COLUMNS))
def typed_frame(request):
    return pd.DataFrame({"c": DTYPE_COLUMNS[request.param], "n": [1.0, None, 1.0]})


def test_every_function_accepts_every_dtype(typed_frame, tmp_path):
    original = typed_frame.copy()

    remove_duplicate_rows(typed_frame)
    handle_missing_values(typed_frame)
    handle_missing_values(typed_frame, "fill_constant", "X")
    handle_missing_values(typed_frame, "fill_constant", 0)
    handle_missing_values(typed_frame, "fill_mean")
    handle_missing_values(typed_frame, "fill_median")
    fill_missing_auto(typed_frame)
    validate_missing_values(typed_frame)
    compare_data_quality(typed_frame, typed_frame)
    _, log = auto_clean(typed_frame, return_log=True)
    auto_clean(typed_frame, missing="auto")
    path = save_quality_report(
        {"profile": profile_dataframe(typed_frame), "frame": typed_frame, "log": log},
        tmp_path / "r.json",
    )

    json.loads(path.read_text(encoding="utf-8"), parse_constant=pytest.fail)
    pd.testing.assert_frame_equal(typed_frame, original)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean", "category"])
def test_fill_constant_works_for_values_outside_the_dtype(dtype):
    values = (
        ["a", None] if dtype == "category" else [1, None] if dtype != "boolean" else [True, None]
    )
    df = pd.DataFrame({"c": pd.Series(values, dtype=dtype)})

    result = handle_missing_values(df, "fill_constant", "missing")

    assert result["c"].tolist()[1] == "missing"


def test_fill_constant_accepts_per_column_dict():
    df = pd.DataFrame({"a": [1, None], "b": ["x", None], "c": [None, 2.0]})

    result = handle_missing_values(df, "fill_constant", {"a": 0, "b": "?"})

    assert result["a"].tolist() == [1.0, 0.0]
    assert result["b"].tolist() == ["x", "?"]
    assert pd.isna(result.loc[0, "c"])


def test_duplicates_and_profile_with_unhashable_cells():
    df = pd.DataFrame({"tags": [["a"], ["a"], ["b"]]})

    assert len(remove_duplicate_rows(df)) == 2
    profile = profile_dataframe(df)
    assert profile["duplicate_row_count"] == 1
    assert profile["columns"].loc["tags", "distinct_count"] == 2


# ---------------------------------------------------------------------------
# auto_clean pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def messy():
    return pd.DataFrame(
        {
            " Order ID ": ["A1", "A2", " A2 ", "A3", None, "A4"],
            "Amount": ["₹1,200.50", "300", "300", "TBD", None, "45"],
            "Paid": ["Yes", "no", "NO", "yes", None, "N/A"],
            "Date": ["2024-01-05", "05-Jan-2024", "05-Jan-2024", "01/07/2024", None, "2024-02-30"],
            "Channel": ["Web", "web", " WEB ", "Store", None, "store"],
            "Unnamed: 5": [None] * 6,
        }
    )


def test_auto_clean_end_to_end(messy):
    result, log = auto_clean(messy, return_log=True)

    assert result.columns.tolist() == ["Order ID", "Amount", "Paid", "Date", "Channel"]
    assert result["Order ID"].tolist() == ["A1", "A2", "A3", "A4"]  # dup + empty row gone
    assert str(result["Amount"].dtype) == "float64"
    assert result["Amount"].tolist()[:2] == [1200.5, 300.0]
    assert pd.isna(result["Amount"].iloc[2])
    assert str(result["Paid"].dtype) == "boolean"
    assert result["Paid"].tolist()[:3] == [True, False, True]
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result["Date"].tolist()[:2] == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05")]
    assert pd.isna(result["Date"].iloc[3])  # 2024-02-30 doesn't exist.
    assert result["Channel"].tolist() == ["Web", "Web", "Store", "Store"]
    steps = {entry["step"] for entry in log}
    assert {"rename_column", "strip_whitespace", "standardize_missing", "unify_case",
            "infer_type", "drop_empty", "remove_duplicates"} <= steps  # fmt: skip


def test_auto_clean_does_not_modify_input(messy):
    original = messy.copy()

    auto_clean(messy, missing="auto")

    pd.testing.assert_frame_equal(messy, original)


def test_auto_clean_keeps_missing_by_default_and_fills_with_auto():
    df = pd.DataFrame(
        {
            "id": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "qty": ["1", "2", "3", "4", "5", None],
            "status": ["ok", "ok", "ok", "bad", "bad", None],
            "email": ["a@x", "b@x", "c@x", "d@x", "e@x", None],
            "when": ["2024-01-01"] * 5 + [None],
        }
    )

    kept = auto_clean(df)
    filled = auto_clean(df, missing="auto")

    assert kept.isna().sum().sum() == 4
    assert filled["qty"].tolist() == [1, 2, 3, 4, 5, 3]
    assert filled["status"].tolist() == ["ok", "ok", "ok", "bad", "bad", "ok"]
    assert pd.isna(filled["email"].iloc[5])  # High-cardinality text is not guessed.
    assert pd.isna(filled["when"].iloc[5])  # Dates are not guessed.


def test_auto_clean_supports_existing_strategies():
    df = pd.DataFrame({"a": ["1", None, "3"], "b": ["x", "y", "z"]})

    assert len(auto_clean(df, missing="drop_rows")) == 2
    assert auto_clean(df, missing="fill_mean")["a"].tolist() == [1, 2, 3]
    assert auto_clean(df, missing="fill_constant", fill_value=0)["a"].tolist() == [1, 0, 3]


def test_auto_clean_steps_can_be_switched_off(messy):
    result = auto_clean(
        messy,
        column_names=None,
        strip_text=False,
        missing_markers=None,
        unify_case=False,
        infer_types=False,
        remove_duplicates=False,
        drop_empty=False,
    )

    pd.testing.assert_frame_equal(result, messy)


def test_auto_clean_snake_case_names():
    df = pd.DataFrame({"Customer Name": ["a"], "Order-ID": ["1"]})

    assert auto_clean(df, column_names="snake").columns.tolist() == ["customer_name", "order_id"]


def test_auto_clean_rejects_bad_input():
    with pytest.raises(TypeError):
        auto_clean([1, 2])
    with pytest.raises(ValueError, match="missing option"):
        auto_clean(pd.DataFrame({"a": [1]}), missing="nope")


def test_auto_clean_handles_empty_frames():
    assert auto_clean(pd.DataFrame()).empty
    result = auto_clean(pd.DataFrame({"a": pd.Series([], dtype=object)}))
    assert result.columns.tolist() == ["a"]


def test_auto_clean_handles_duplicate_column_names():
    df = pd.DataFrame([["1", "x"], ["2", "y"]], columns=["a", "a"])

    result = auto_clean(df, column_names=None)

    assert result.columns.tolist() == ["a", "a"]
    assert str(result.iloc[:, 0].dtype) == "Int64"


def test_load_then_auto_clean_round_trip(tmp_path):
    path = tmp_path / "export.csv"
    path.write_bytes(
        "Name;Price;Active;\n Ana ;€1.200,00;Yes;\nBob;€15;no;\n;;;\n".encode("cp1252")
    )

    result = auto_clean(load_csv(path), missing="keep")

    assert result.columns.tolist() == ["Name", "Price", "Active"]
    assert result["Name"].tolist() == ["Ana", "Bob"]
    assert str(result["Active"].dtype) == "boolean"


def test_profile_and_report_after_auto_clean(tmp_path, messy):
    cleaned, log = auto_clean(messy, return_log=True)

    path = save_quality_report(
        {
            "profile": profile_dataframe(cleaned),
            "comparison": compare_data_quality(messy, cleaned),
            "log": log,
            "sample": cleaned.head(),
        },
        tmp_path / "report.json",
    )

    report = json.loads(path.read_text(encoding="utf-8"), parse_constant=pytest.fail)
    assert report["profile"]["columns"]["Paid"]["data_type"] == "boolean"
    assert report["sample"]["0"]["Date"] == "2024-01-05T00:00:00"
