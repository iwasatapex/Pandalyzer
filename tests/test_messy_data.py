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
