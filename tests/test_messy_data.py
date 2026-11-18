"""Unit tests for the messy-export handling: emails, report preambles and footers,
repeated headers, international number formats and invisible characters."""

import pandas as pd
import pytest

from autoclean import (
    DEFAULT_MISSING_MARKERS,
    auto_clean,
    clean_column_names,
    count_preamble_lines,
    drop_repeated_headers,
    drop_total_rows,
    email_columns,
    infer_series_type,
    is_valid_email,
    load_csv,
    normalize_email,
    normalize_emails,
    strip_whitespace,
    unwrap_excel_text,
    validate_emails,
)

TEXT_DTYPES = ["str", "string", "string[python]", object]


def _series(values, dtype="str"):
    return pd.Series(values, dtype=dtype)


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ann@Example.COM", "ann@example.com"),
        ("  bob@example.org ", "bob@example.org"),
        ("mailto:Cy@Example.io", "cy@example.io"),
        ("Dee Lee <dee@example.com>", "dee@example.com"),
        ("\u200beve@example.com", "eve@example.com"),
        ("fay@example.com.", "fay@example.com"),
    ],
)
def test_normalize_email(raw, expected):
    assert normalize_email(raw) == expected
    assert is_valid_email(raw)


@pytest.mark.parametrize(
    "bad",
    [
        "@example.com",
        "user@",
        "a@@example.com",
        "not an email",
        "user@example",
        "user@-example.com",
        "user..name@example.com",
        ".user@example.com",
        "user@exa_mple.com",
        "",
        None,
        42,
    ],
)
def test_invalid_emails(bad):
    assert not is_valid_email(bad)


def test_email_column_detection():
    df = pd.DataFrame(
        {
            "email": ["a@x.com", "b@x.com", "bad@", "c@y.org", "d@z.io"],
            "handle": ["@ann", "@bob", "@cy", "@dee", "@eve"],  # Has "@", but no addresses.
            "name": ["Ann", "Bob", "Cy", "Dee", "Eve"],
            "count": [1, 2, 3, 4, 5],
        }
    )
    assert email_columns(df) == ["email"]


@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_normalize_emails_sets_invalid_to_missing(dtype):
    df = pd.DataFrame({"e": _series(["A@X.COM", "mailto:b@x.com", "bad@", None], dtype)})

    result = normalize_emails(df)

    assert result["e"].tolist()[:2] == ["a@x.com", "b@x.com"]
    assert result["e"].isna().tolist() == [False, False, True, True]
    assert result["e"].dtype == df["e"].dtype


def test_normalize_emails_can_keep_invalid():
    df = pd.DataFrame({"e": ["A@X.COM", "bad@", "c@y.org"]})

    result = normalize_emails(df, invalid="keep")

    assert result["e"].tolist() == ["a@x.com", "bad@", "c@y.org"]
    with pytest.raises(ValueError):
        normalize_emails(df, invalid="drop")


def test_validate_emails():
    df = pd.DataFrame(
        {"email": [" A@X.COM", "bad@", None, "c@y.org", "x@@y.com"], "n": [1, 2, 3, 4, 5]},
        index=[10, 11, 12, 13, 14],
    )
    assert validate_emails(df) == {"email": [11, 14]}
    assert validate_emails(df, ["email"]) == {"email": [11, 14]}
    with pytest.raises(KeyError):
        validate_emails(df, ["missing"])
    with pytest.raises(TypeError):
        validate_emails([1, 2])


def test_auto_clean_email_step_and_log():
    df = pd.DataFrame(
        {
            "email": ["Ann@X.com", "ann@x.com ", "bad@", "bo@y.org", "cy@z.io"],
            "name": ["Ann", "Ann", "Bad", "Bo", "Cy"],
        }
    )

    result, log = auto_clean(df, return_log=True)
    kept, _ = auto_clean(df, emails="keep", return_log=True)
    skipped = auto_clean(df, emails=None)

    values = result["email"].tolist()
    assert len(values) == 4  # "Ann@X.com" and "ann@x.com " are the same contact.
    assert values[0] == "ann@x.com" and pd.isna(values[1])
    assert values[2:] == ["bo@y.org", "cy@z.io"]
    entry = next(e for e in log if e["step"] == "normalize_emails")
    assert "1 invalid addresses set to missing" in entry["detail"]
    assert "bad@" in kept["email"].tolist()
    assert "Ann@X.com" in skipped["email"].tolist()
    with pytest.raises(ValueError):
        auto_clean(df, emails="drop")


# ---------------------------------------------------------------------------
# Report preambles, repeated headers and total rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Title\nGenerated: Jan 5, 2024\n\na,b,c\n1,2,3\n4,5,6\n", 3),
        ("a,b,c\n1,2,3\n", 0),
        ("a,b,c\n1,2,3,\n4,5,6,\n", 0),  # Trailing delimiter on data rows only.
        ("a,b,,\n1,2,3,4\n5,6,7,8\n", 0),  # Blank header names.
        ('Code,"Hours\nWorked",Rate\n1,2,3\n4,5,6\n', 0),  # Line break inside a header.
        ('Report\n\nCode,"Hours\nWorked",Rate\n1,2,3\n', 2),
        ("only\n1\n2\n", 0),  # Single column.
        ("", 0),
    ],
)
def test_count_preamble_lines(text, expected):
    assert count_preamble_lines(text, ",") == expected


def test_load_csv_skips_preamble(tmp_path):
    path = tmp_path / "statement.csv"
    path.write_text(
        "Account statement\nGenerated: Jan 5, 2024\n\ndate,amount,ref\n2024-01-01,1,A\n"
    )

    df = load_csv(path)

    assert df.columns.tolist() == ["date", "amount", "ref"]
    assert df.attrs["preamble"] == ["Account statement", "Generated: Jan 5, 2024", ""]
    assert load_csv(path, skiprows=3).columns.tolist() == ["date", "amount", "ref"]
    assert "preamble" not in load_csv(path, skiprows=3).attrs  # Explicit options win.


def test_drop_repeated_headers():
    df = pd.DataFrame(
        {
            "Order ID": ["1", "Order ID", "2", " order  id ", "id"],
            "Qty": ["5", "Qty", "6", "QTY", "Qty"],
            "Note": ["x", None, "y", "note", None],
        }
    )
    result = drop_repeated_headers(df)

    # Row 4 has "id" (not the header name) in the first column, so it stays.
    assert result.index.tolist() == [0, 2, 4]
    renamed = df.set_axis(["order_id", "qty", "note"], axis=1)
    assert drop_repeated_headers(renamed, header=df.columns).index.tolist() == [0, 2, 4]
    with pytest.raises(ValueError):
        drop_repeated_headers(df, header=["a"])


def test_drop_repeated_headers_matches_pandas_deduplicated_names():
    df = pd.DataFrame([["1", "2"], ["id", "id"], ["3", "4"]], columns=["id", "id.1"])
    assert drop_repeated_headers(df).index.tolist() == [0, 2]


def test_drop_total_rows_only_at_the_end():
    df = pd.DataFrame(
        {
            "label": ["Total", "a", "b", None, "Grand Total", "SUBTOTAL:", None],
            "value": [9, 1, 2, None, 3, 3, None],
        }
    )
    result = drop_total_rows(df)

    assert result.index.tolist() == [0, 1, 2, 3, 6]  # The first "Total" is a real row.
    assert drop_total_rows(df.iloc[:0]).empty
    assert drop_total_rows(pd.DataFrame({"x": ["Totally", "Sumatra"]})).shape[0] == 2


def test_drop_total_rows_with_duplicate_index():
    df = pd.DataFrame({"a": ["x", "y", "Total"]}, index=[0, 0, 0])
    assert drop_total_rows(df)["a"].tolist() == ["x", "y"]


def test_auto_clean_drops_artifacts_before_inference():
    df = pd.DataFrame(
        {
            "Store": ["A", "Store", "B", "Total"],
            "Units": ["5", "Units", "7", "12"],
        }
    )
    result, log = auto_clean(df, column_names="snake", return_log=True)

    assert result["units"].tolist() == [5, 7]
    assert str(result["units"].dtype) == "Int64"
    assert [e["detail"] for e in log if e["step"] == "drop_artifacts"] == [
        "dropped 1 header rows",
        "dropped 1 total rows",
    ]
    assert len(auto_clean(df, drop_artifacts=False)) == 4


# ---------------------------------------------------------------------------
# Whitespace, invisible characters, Excel wrappers, placeholders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_strip_whitespace_collapses_and_removes_invisible(dtype):
    df = pd.DataFrame(
        {
            "t": _series(
                ["John   Smith", "Ann\tLee", "\u200b42\u200b", "a\u00a0b", "line1\nline2 ", None],
                dtype,
            )
        }
    )
    result = strip_whitespace(df)

    assert result["t"].tolist()[:5] == ["John Smith", "Ann Lee", "42", "a b", "line1\nline2"]
    assert result["t"].dtype == df["t"].dtype
    assert strip_whitespace(df, collapse=False)["t"].iloc[0] == "John   Smith"


@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_unwrap_excel_text(dtype):
    df = pd.DataFrame({"code": _series(['="00123"', "00456", '="A=B"', "=SUM(A1)"], dtype)})
    assert unwrap_excel_text(df)["code"].tolist() == ["00123", "00456", "A=B", "=SUM(A1)"]


def test_excel_codes_keep_leading_zeros_through_auto_clean():
    df = pd.DataFrame({"code": ['="00123"', '="00456"', '="01000"']})
    result = auto_clean(df)
    assert result["code"].tolist() == ["00123", "00456", "01000"]


def test_clean_column_names_collapses_line_breaks():
    df = pd.DataFrame(columns=["Hours\r\nWorked", "  Start   Date "])
    assert clean_column_names(df).columns.tolist() == ["Hours Worked", "Start Date"]


@pytest.mark.parametrize("marker", ["\u2014", "\u2013", "(blank)", "<NA>", "(null)", "NaT"])
def test_new_missing_markers(marker):
    assert marker.casefold() in DEFAULT_MISSING_MARKERS
    df = pd.DataFrame({"v": ["1", marker, "3"] * 10, "id": [str(i) for i in range(30)]})
    result = auto_clean(df)
    assert result["v"].isna().sum() == 10
    assert str(result["v"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# Number and boolean formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["USD 1,234.50", "1,200.00 USD", "-USD 12.00", "USD9.99"], [1234.5, 1200.0, -12.0, 9.99]),
        (["EUR 15,00", "15,00 EUR", "1.234,50 EUR", "2,5 EUR"], [15.0, 15.0, 1234.5, 2.5]),
        (["1 234 567", "12 345", "999", "-1 000"], [1234567, 12345, 999, -1000]),
        (["1'234.50", "12'345.00", "99.10", "CHF 5.50"], [1234.5, 12345.0, 99.1, 5.5]),
        (["1\u2019234.50", "99.10", "1\u2019000.00", "5"], [1234.5, 99.1, 1000.0, 5.0]),
        (
            ["1\u202f234,56 €", "12,00 €", "999,99 €", "1\u00a0000,00 €"],
            [1234.56, 12.0, 999.99, 1000.0],
        ),
        (["12.5 %", "3 %", "100%", "0 %"], [12.5, 3.0, 100.0, 0.0]),
    ],
)
@pytest.mark.parametrize("dtype", ["str", "string[python]", object])
def test_international_number_formats(values, expected, dtype):
    converted, kind = infer_series_type(_series(values * 5, dtype))
    assert kind in ("integer", "float")
    assert converted.tolist()[:4] == pytest.approx(expected)


@pytest.mark.parametrize(
    "values",
    [
        ["SKU 123", "SKU 456", "SKU 789"],  # Not a currency code.
        ["1,234 567", "12 345", "1,000"],  # Mixed separators within one value.
        ["555 1234", "555 9876", "555 0000"],  # Not 3-digit groups.
        ["+1 (555) 123-4567", "555-123-4567", "555.123.4567"],
    ],
)
def test_lookalike_numbers_stay_text(values):
    _, kind = infer_series_type(_series(values * 5))
    assert kind is None


def test_on_off_booleans():
    converted, kind = infer_series_type(_series(["on", "OFF", "On", "off", None]))
    assert kind == "boolean"
    assert converted.tolist()[:4] == [True, False, True, False]


def test_object_column_of_booleans_with_blanks_becomes_boolean():
    # How pandas loads a TRUE/FALSE column that has empty cells.
    series = pd.Series([True, False, None, True], dtype=object)

    converted, kind = infer_series_type(series)

    assert kind == "boolean"
    assert str(converted.dtype) == "boolean"
    assert converted.isna().tolist() == [False, False, True, False]


def test_boolean_column_with_blanks_survives_save_and_reload(tmp_path):
    df = pd.DataFrame({"flag": ["TRUE", "FALSE", "", "TRUE"], "n": ["1", "2", "3", "4"]})
    path = tmp_path / "flags.csv"
    df.to_csv(path, index=False)

    result = auto_clean(load_csv(path))

    assert str(result["flag"].dtype) == "boolean"


# ---------------------------------------------------------------------------
# Regressions found by the re-audit (one test per bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "options"),
    [
        ("a  b  c\n1  2  3\n4  5  6\n", {"sep": r"\s+"}),
        ("a::b::c\n1::2::3\n4::5::6\n", {"sep": "::", "engine": "python"}),
        ("a,b,c\n1,2,3\n4,5,6\n", {"engine": "python"}),
    ],
)
def test_load_csv_regex_multichar_separators_and_python_engine(tmp_path, text, options):
    path = tmp_path / "data.txt"
    path.write_text(text)

    df = load_csv(path, **options)

    assert df.columns.tolist() == ["a", "b", "c"]
    assert len(df) == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a,b,c\n1,2,3,4\n5,6,7,8\n9,10,11,12\n", 0),  # Data rows wider than the header.
        ("Title\n\na,b,c\n1,2,3,4\n5,6,7,8\n", 2),
        ("# exported by tool\n# v2\na,b,c\n1,2,3\n", 2),  # Tie between 1 and 3 fields.
        ("Account statement\nGenerated: Jan 5, 2024\n\ndate,amount\n2024-01-01,1\n", 3),
        ("Generated: Jan 5, 2024\ndate,amount,ref\n2024-01-01,1,A\n2024-01-02,2,B\n", 1),
        ("Prepared by Ann, Finance\n\na,b,c\n1,2,3\n", 2),
        ("region,2023,2024\nN,1,2\nS,3,4\n", 0),  # Numbers in a header are not a note.
        ("2023,2024\n1,2\n\n3,4\n", 0),
    ],
)
def test_count_preamble_lines_regressions(text, expected):
    assert count_preamble_lines(text, ",") == expected


def test_load_csv_keeps_header_above_wider_rows(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("a,b,c\n1,2,3,4\n5,6,7,8\n")

    with pytest.warns(pd.errors.ParserWarning):  # The extra fields are dropped, as documented.
        df = load_csv(path)

    assert df.columns.tolist() == ["a", "b", "c"]
    assert len(df) == 2


def test_headerless_frames_keep_rows_that_match_column_numbers():
    df = pd.DataFrame([["0", "1", "2"], ["5", "6", "7"]])  # As load_csv(header=None) gives.
    assert auto_clean(df).values.tolist() == [[0, 1, 2], [5, 6, 7]]
    assert len(drop_repeated_headers(df)) == 2


def test_repeated_header_needs_two_matching_cells_in_two_columns():
    df = pd.DataFrame({"name": ["Ann", "name", "Bo"], "nickname": ["A", None, "B"]})
    assert drop_repeated_headers(df).index.tolist() == [0, 1, 2]
    df.loc[1, "nickname"] = "Nickname"
    assert drop_repeated_headers(df).index.tolist() == [0, 2]


def test_last_row_named_like_a_total_is_kept():
    df = pd.DataFrame({"store": ["Fit Co", "Gym Max", "Total Gym"], "sales": [10, 20, 30]})
    assert auto_clean(df)["store"].tolist() == ["Fit Co", "Gym Max", "Total Gym"]


def test_total_rows_that_leave_columns_empty_are_dropped():
    df = pd.DataFrame(
        {
            "date": ["d1", "d2", "d3", None, "Total debits", "Total credits"],
            "desc": ["a", "b", "c", None, None, None],
            "amount": [1, 2, 3, None, 6, 9],
        }
    )
    assert drop_total_rows(df)["date"].tolist()[:3] == ["d1", "d2", "d3"]
    assert len(drop_total_rows(df)) == 4  # The empty row is drop_empty_rows' job.


def test_phone_numbers_in_an_email_column_are_kept():
    contacts = [f"u{i}@x.com" for i in range(9)] + ["+1 (555) 123-4567"]
    df = pd.DataFrame({"contact": contacts, "id": range(10)})

    result = auto_clean(df)

    assert result["contact"].iloc[-1] == "+1 (555) 123-4567"
    assert validate_emails(df) == {"contact": [9]}  # Still listed for review.


def test_address_with_inner_space_is_not_rewritten():
    assert normalize_email(" john smith@example.com ") == "john smith@example.com"
    assert not is_valid_email("john smith@example.com")
    df = pd.DataFrame({"e": ["john smith@example.com", "a@x.com", "b@x.com"], "i": [1, 2, 3]})
    assert pd.isna(auto_clean(df)["e"].iloc[0])


@pytest.mark.parametrize(
    "address",
    ["user@example.xn--p1ai", "josé@example.com", "user@例え.jp", "straße@x.de", "a_b@x.co.uk"],
)
def test_international_addresses_are_valid(address):
    assert is_valid_email(address)
    assert normalize_email(address) == address.lower()  # lower(), so "ß" stays "ß".


@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_infer_handles_nbsp_and_zero_width_without_auto_clean(dtype):
    values = ["€\u00a012.50", "€\u00a03.10", "\u200b5", "1\u00a0234.00"] * 5

    converted, kind = infer_series_type(_series(values, dtype))

    assert kind == "float"
    assert converted.tolist()[:4] == [12.5, 3.1, 5.0, 1234.0]


def test_leading_zero_codes_after_a_currency_symbol_stay_text():
    _, kind = infer_series_type(_series(["$0123", "$0456", "$0789"] * 5))
    assert kind is None
    _, kind = infer_series_type(_series(["$0.50", "$1.25"] * 5))
    assert kind == "float"


def test_validators_handle_repeated_column_names():
    from autoclean import validate_column_types, validate_numeric_ranges

    numbers = pd.DataFrame([[1, 200], [5, 6]], columns=["a", "a"])
    emails = pd.DataFrame([["a@x.com", "b@x.com"], ["bad@", "c@x.com"]], columns=["e", "e"])

    assert validate_numeric_ranges(numbers, {"a": (0, 10)}) == {"a": [0]}
    assert validate_column_types(numbers, {"a": "int64"}) == {}
    assert validate_emails(emails) == {"e": [1]}


def test_categorical_text_is_cleaned_and_stays_categorical():
    categories = [" b ", "A", "N/A", "a", "b"]
    df = pd.DataFrame(
        {
            "c": pd.Categorical(["a", "N/A", " b ", "A", "b"], categories=categories, ordered=True),
            "i": range(5),
        }
    )
    result, log = auto_clean(df, return_log=True)

    assert result["c"].tolist()[:1] == ["a"] and pd.isna(result["c"].iloc[1])
    assert result["c"].tolist()[2:] == ["b", "a", "b"]
    assert list(result["c"].cat.categories) == ["b", "a"]  # Original order kept.
    assert result["c"].cat.ordered
    assert {entry["step"] for entry in log} >= {"strip_whitespace", "standardize_missing"}
    again, log_again = auto_clean(result, return_log=True)
    pd.testing.assert_frame_equal(again, result)
    assert log_again == []


def test_clean_categorical_is_returned_unchanged():
    series = pd.Categorical(["x", "y", "x"], categories=["y", "x", "z"])
    df = pd.DataFrame({"c": series, "i": [1, 2, 3]})
    pd.testing.assert_series_equal(auto_clean(df)["c"], df["c"])


def test_inference_on_repeated_values_matches_value_by_value():
    # Over 20% repeats takes the parse-each-distinct-value-once path.
    prices, kind = infer_series_type(_series(["$1,200.50", "$3.10", "(0.99)"] * 50))
    assert kind == "float"
    assert prices.tolist() == pytest.approx([1200.5, 3.1, -0.99] * 50)

    dates, kind = infer_series_type(_series(["2024-01-05", "05 Feb 2024", "bad"]))
    assert kind is None  # One bad value in three is under the 95% threshold.
    dates, kind = infer_series_type(_series(["2024-01-05", "05 Feb 2024"] * 50))
    assert kind == "datetime"
    assert dates.tolist()[:2] == [pd.Timestamp(2024, 1, 5), pd.Timestamp(2024, 2, 5)]
    assert dates.index.equals(pd.RangeIndex(100))


def test_notebooks_do_not_point_outside_the_project():
    from pathlib import Path

    for notebook in (Path(__file__).parents[1] / "notebooks").glob("*.ipynb"):
        assert "/mnt/data" not in notebook.read_text(encoding="utf-8"), notebook.name
