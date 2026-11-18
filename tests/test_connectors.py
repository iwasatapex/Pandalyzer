import json

import pandas as pd
import pytest

from autoclean.connectors import load_csv, save_csv, save_quality_report


def test_loads_csv_into_dataframe(tmp_path):
    csv_file = tmp_path / "orders.csv"
    csv_file.write_text(
        "order_id,amount\n1,100\n2,250\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert result["order_id"].tolist() == [1, 2]
    assert result["amount"].tolist() == [100, 250]


def test_raises_error_for_missing_csv(tmp_path):
    csv_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(csv_file)


def test_loads_csv_with_missing_values(tmp_path):
    csv_file = tmp_path / "missing_values.csv"
    csv_file.write_text(
        "order_id,customer\n1,Alice\n2,\n3,Charlie\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert pd.isna(result.loc[1, "customer"])
    assert result["customer"].isna().sum() == 1


def test_raises_error_when_path_is_a_directory(tmp_path):
    with pytest.raises((IsADirectoryError, PermissionError, OSError)):
        load_csv(tmp_path)


def test_preserves_column_names(tmp_path):
    csv_file = tmp_path / "column_names.csv"
    csv_file.write_text(
        "Order ID,Customer Name,Total Amount\n1,Alice,100\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert list(result.columns) == [
        "Order ID",
        "Customer Name",
        "Total Amount",
    ]


def test_raises_error_for_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")

    with pytest.raises(pd.errors.EmptyDataError):
        load_csv(csv_file)


def test_handles_duplicate_column_names(tmp_path):
    csv_file = tmp_path / "duplicate_columns.csv"
    csv_file.write_text(
        "id,amount,amount\n1,100,200\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert list(result.columns) == ["id", "amount", "amount.1"]
    assert result.loc[0, "amount"] == 100
    assert result.loc[0, "amount.1"] == 200


def test_loads_csv_with_headers_but_no_rows(tmp_path):
    csv_file = tmp_path / "headers_only.csv"
    csv_file.write_text(
        "order_id,customer,amount\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["order_id", "customer", "amount"]
    assert result.shape == (0, 3)


def test_infers_numeric_column_types(tmp_path):
    csv_file = tmp_path / "numeric_types.csv"
    csv_file.write_text(
        "order_id,amount\n1,100.5\n2,250.75\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert pd.api.types.is_integer_dtype(result["order_id"])
    assert pd.api.types.is_float_dtype(result["amount"])


def test_loads_quoted_values_containing_commas(tmp_path):
    csv_file = tmp_path / "quoted_commas.csv"
    csv_file.write_text(
        'customer_id,customer_name,amount\n1,"Smith, John",150\n2,"Doe, Jane",200\n',
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert result.shape == (2, 3)
    assert result.loc[0, "customer_name"] == "Smith, John"
    assert result.loc[1, "customer_name"] == "Doe, Jane"


def test_default_loading_infers_numeric_identifiers(tmp_path):
    csv_file = tmp_path / "leading_zeros.csv"
    csv_file.write_text(
        "customer_id,amount\n00123,100\n00456,200\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert result["customer_id"].tolist() == [123, 456]


def test_loads_csv_with_utf8_text(tmp_path):
    csv_file = tmp_path / "utf8.csv"
    csv_file.write_text(
        "customer,city\nJosé,Dehradun\nMüller,Delhi\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert result["customer"].tolist() == ["José", "Müller"]
    assert result["city"].tolist() == ["Dehradun", "Delhi"]


def test_loads_single_column_csv(tmp_path):
    csv_file = tmp_path / "single_column.csv"
    csv_file.write_text(
        "product_id\n101\n102\n103\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert result.shape == (3, 1)
    assert result["product_id"].tolist() == [101, 102, 103]


def test_loads_single_row_csv(tmp_path):
    csv_file = tmp_path / "single_row.csv"
    csv_file.write_text(
        "order_id,amount\n1,250\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert result.shape == (1, 2)
    assert result.loc[0, "order_id"] == 1
    assert result.loc[0, "amount"] == 250


def test_preserves_row_order(tmp_path):
    csv_file = tmp_path / "row_order.csv"
    csv_file.write_text(
        "order_id\n30\n10\n20\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert result["order_id"].tolist() == [30, 10, 20]


def test_loads_boolean_like_values_as_strings_or_objects(tmp_path):
    csv_file = tmp_path / "boolean_values.csv"
    csv_file.write_text(
        "status\nTrue\nFalse\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert result.shape == (2, 1)
    assert result["status"].astype(str).tolist() == ["True", "False"]


def test_loads_csv_with_whitespace_in_values(tmp_path):
    csv_file = tmp_path / "whitespace.csv"
    csv_file.write_text(
        "customer,city\n Alice , Dehradun \n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert result.loc[0, "customer"] == " Alice "
    assert result.loc[0, "city"] == " Dehradun "


def test_load_csv_does_not_modify_source_file(tmp_path):
    csv_file = tmp_path / "source.csv"
    original_content = "id,amount\n1,100\n2,200\n"
    csv_file.write_text(original_content, encoding="utf-8")

    load_csv(csv_file)

    assert csv_file.read_text(encoding="utf-8") == original_content


# --- Pandalyzer extension tests: exports ---


def test_save_csv_writes_csv_and_returns_path(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    path = save_csv(df, tmp_path / "nested" / "clean.csv")
    assert path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(path), df)


def test_save_csv_rejects_non_dataframe(tmp_path):
    with pytest.raises(TypeError):
        save_csv([1], tmp_path / "bad.csv")


def test_save_quality_report_writes_json(tmp_path):
    path = save_quality_report({"rows": 3, "issues": ["missing amount"]}, tmp_path / "report.json")
    assert json.loads(path.read_text(encoding="utf-8"))["rows"] == 3


def test_save_quality_report_converts_dataframe(tmp_path):
    path = save_quality_report(
        {"columns": pd.DataFrame({"missing_count": [2]}, index=["amount"])},
        tmp_path / "report.json",
    )
    assert json.loads(path.read_text(encoding="utf-8"))["columns"]["amount"]["missing_count"] == 2


# --- Integrated additional connector coverage tests ---
def test_quality_report_converts_series(tmp_path):
    path = save_quality_report(
        {"amount": pd.Series([10, 20], index=["a", "b"])},
        tmp_path / "series.json",
    )
    assert json.loads(path.read_text())["amount"] == {"a": 10, "b": 20}


def test_quality_report_converts_dataframe(tmp_path):
    frame = pd.DataFrame({"amount": [10, 20]}, index=["a", "b"])
    path = save_quality_report({"table": frame}, tmp_path / "frame.json")
    assert json.loads(path.read_text())["table"] == {
        "a": {"amount": 10},
        "b": {"amount": 20},
    }


def test_quality_report_converts_nested_lists_and_tuples(tmp_path):
    path = save_quality_report(
        {"nested": ([1, 2], {"values": (3, 4)})},
        tmp_path / "nested.json",
    )
    assert json.loads(path.read_text()) == {
        "nested": [[1, 2], {"values": [3, 4]}],
    }


def test_quality_report_converts_pandas_na(tmp_path):
    path = save_quality_report({"value": pd.NA}, tmp_path / "na.json")
    assert json.loads(path.read_text()) == {"value": None}


def test_quality_report_converts_numpy_scalars(tmp_path):
    import numpy as np

    path = save_quality_report(
        {"integer": np.int64(7), "floating": np.float64(2.5)},
        tmp_path / "numpy.json",
    )
    assert json.loads(path.read_text()) == {"integer": 7, "floating": 2.5}


@pytest.mark.parametrize("error", [ValueError("cannot convert"), TypeError("cannot convert")])
def test_quality_report_falls_back_when_item_raises(tmp_path, error):
    class BrokenItem:
        def item(self):
            raise error

        def __str__(self):
            return "fallback-value"

    path = save_quality_report(
        {"value": BrokenItem()},
        tmp_path / "fallback.json",
    )
    assert json.loads(path.read_text()) == {"value": "fallback-value"}
