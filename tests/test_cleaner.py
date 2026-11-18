import pandas as pd
import pandas.testing as pdt
import pytest

from autoclean.cleaner import (
    clean_dataframe,
    handle_missing_values,
    remove_duplicate_rows,
)


def test_removes_exact_duplicate_rows():
    df = pd.DataFrame(
        {
            "order_id": [101, 102, 102, 103, 103, 103],
            "amount": [50, 75, 75, 90, 90, 90],
        }
    )

    result = remove_duplicate_rows(df)

    expected = pd.DataFrame(
        {
            "order_id": [101, 102, 103],
            "amount": [50, 75, 90],
        },
        index=[0, 1, 3],
    )

    pdt.assert_frame_equal(result, expected)


def test_preserves_first_occurrence_and_row_order():
    df = pd.DataFrame(
        {
            "item": ["B", "A", "B", "C", "A"],
            "quantity": [2, 1, 2, 3, 1],
        }
    )

    result = remove_duplicate_rows(df)

    expected = pd.DataFrame(
        {
            "item": ["B", "A", "C"],
            "quantity": [2, 1, 3],
        },
        index=[0, 1, 3],
    )

    pdt.assert_frame_equal(result, expected)


def test_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "item": ["A", "A", "B"],
            "price": [10, 10, 20],
        }
    )
    original = df.copy(deep=True)

    remove_duplicate_rows(df)

    pdt.assert_frame_equal(df, original)


def test_handles_non_sequential_index():
    df = pd.DataFrame(
        {
            "item": ["A", "B", "A", "C"],
            "price": [10, 20, 10, 30],
        },
        index=[10, 25, 40, 90],
    )

    result = remove_duplicate_rows(df)

    expected = pd.DataFrame(
        {
            "item": ["A", "B", "C"],
            "price": [10, 20, 30],
        },
        index=[10, 25, 90],
    )

    pdt.assert_frame_equal(result, expected)


def test_handles_dataframe_without_duplicates():
    df = pd.DataFrame(
        {
            "item": ["A", "B", "C"],
            "price": [10, 20, 30],
        }
    )

    result = remove_duplicate_rows(df)

    pdt.assert_frame_equal(result, df)


def test_rejects_non_dataframe_input():
    with pytest.raises(
        TypeError,
        match="Input must be a pandas DataFrame",
    ):
        remove_duplicate_rows([1, 2, 3])


def test_handles_empty_dataframe():
    df = pd.DataFrame(
        {
            "item": pd.Series(dtype="object"),
            "price": pd.Series(dtype="int64"),
        }
    )

    result = remove_duplicate_rows(df)

    pdt.assert_frame_equal(result, df)


def test_keeps_one_row_when_all_rows_are_duplicates():
    df = pd.DataFrame(
        {
            "item": ["A", "A", "A", "A"],
            "price": [100, 100, 100, 100],
        }
    )

    result = remove_duplicate_rows(df)

    expected = pd.DataFrame(
        {
            "item": ["A"],
            "price": [100],
        },
        index=[0],
    )

    pdt.assert_frame_equal(result, expected)


def test_preserves_column_names_and_data_types():
    df = pd.DataFrame(
        {
            "Order ID": pd.Series([101, 101, 102], dtype="int64"),
            "Item Name": pd.Series(["Book", "Book", "Pen"], dtype="object"),
        }
    )

    result = remove_duplicate_rows(df)
    expected = pd.DataFrame(
        {
            "Order ID": [101, 102],
            "Item Name": ["Book", "Pen"],
        },
        index=[0, 2],
    ).astype(
        {
            "Order ID": "int64",
            "Item Name": "object",
        }
    )
    pdt.assert_frame_equal(result, expected)


def test_drop_rows_removes_rows_with_missing_values():
    df = pd.DataFrame(
        {
            "item": ["Book", None, "Pen", "Bag"],
            "price": [100, 200, None, 400],
        }
    )

    result = handle_missing_values(df, strategy="drop_rows")

    expected = pd.DataFrame(
        {
            "item": ["Book", "Bag"],
            "price": [100.0, 400.0],
        },
        index=[0, 3],
    )

    pdt.assert_frame_equal(result, expected)


def test_drop_rows_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "item": ["Book", None, "Pen"],
            "price": [100, 200, 300],
        }
    )
    original = df.copy(deep=True)

    handle_missing_values(df, strategy="drop_rows")

    pdt.assert_frame_equal(df, original)


def test_drop_rows_rejects_non_dataframe_input():
    with pytest.raises(
        TypeError,
        match="Input must be a pandas DataFrame",
    ):
        handle_missing_values([1, 2, 3], strategy="drop_rows")


def test_rejects_unsupported_missing_value_strategy():
    df = pd.DataFrame(
        {
            "item": ["Book", None],
            "price": [100, 200],
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported missing-value strategy",
    ):
        handle_missing_values(df, strategy="unknown")


def test_fill_constant_replaces_missing_values():
    df = pd.DataFrame(
        {
            "item": ["Book", None, "Pen"],
            "price": [100, 200, None],
        }
    )

    result = handle_missing_values(
        df,
        strategy="fill_constant",
        value="Unknown",
    )

    expected = pd.DataFrame(
        {
            "item": ["Book", "Unknown", "Pen"],
            "price": [100, 200, "Unknown"],
        }
    )

    pdt.assert_frame_equal(result, expected)


def test_fill_constant_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "item": ["Book", None, "Pen"],
            "price": [100, 200, 300],
        }
    )
    original = df.copy(deep=True)

    handle_missing_values(
        df,
        strategy="fill_constant",
        value="Unknown",
    )

    pdt.assert_frame_equal(df, original)


def test_fill_constant_rejects_non_dataframe_input():
    with pytest.raises(
        TypeError,
        match="Input must be a pandas DataFrame",
    ):
        handle_missing_values(
            [1, 2, 3],
            strategy="fill_constant",
            value=0,
        )


def test_fill_constant_handles_dataframe_without_missing_values():
    df = pd.DataFrame(
        {
            "item": ["Book", "Pen"],
            "price": [100, 200],
        }
    )

    result = handle_missing_values(
        df,
        strategy="fill_constant",
        value="Unknown",
    )

    pdt.assert_frame_equal(result, df)


def test_fill_mean_replaces_missing_numeric_values():
    df = pd.DataFrame(
        {
            "price": [10.0, None, 20.0, 30.0],
        }
    )

    result = handle_missing_values(
        df,
        strategy="fill_mean",
    )

    expected = pd.DataFrame(
        {
            "price": [10.0, 20.0, 20.0, 30.0],
        }
    )

    pdt.assert_frame_equal(result, expected)


def test_fill_mean_leaves_non_numeric_columns_unchanged():
    df = pd.DataFrame(
        {
            "price": [10.0, None, 30.0],
            "category": ["Book", None, "Pen"],
        }
    )

    result = handle_missing_values(
        df,
        strategy="fill_mean",
    )

    expected = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0],
            "category": ["Book", None, "Pen"],
        }
    )

    pdt.assert_frame_equal(result, expected)


def test_fill_mean_leaves_entirely_missing_numeric_column_unchanged():
    df = pd.DataFrame(
        {
            "price": pd.Series([None, None, None], dtype="float64"),
        }
    )

    result = handle_missing_values(
        df,
        strategy="fill_mean",
    )

    expected = pd.DataFrame(
        {
            "price": pd.Series([None, None, None], dtype="float64"),
        }
    )

    pdt.assert_frame_equal(result, expected)


def test_fill_mean_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "price": [10.0, None, 30.0],
        }
    )

    original = df.copy(deep=True)

    handle_missing_values(
        df,
        strategy="fill_mean",
    )

    pdt.assert_frame_equal(df, original)


def test_fill_mean_rejects_dataframe_without_numeric_columns():
    df = pd.DataFrame(
        {
            "category": ["Book", None, "Pen"],
        }
    )

    with pytest.raises(ValueError, match="numeric"):
        handle_missing_values(
            df,
            strategy="fill_mean",
        )


def test_fill_mean_handles_numeric_column_without_missing_values():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0],
            "category": ["Book", "Pen", "Bag"],
        }
    )

    result = handle_missing_values(df, strategy="fill_mean")
    pdt.assert_frame_equal(result, df)


def test_fill_median_replaces_missing_numeric_values():
    df = pd.DataFrame(
        {
            "price": [10.0, None, 30.0, 100.0],
        }
    )

    result = handle_missing_values(
        df,
        strategy="fill_median",
    )

    expected = pd.DataFrame(
        {
            "price": [10.0, 30.0, 30.0, 100.0],
        }
    )

    pdt.assert_frame_equal(result, expected)


def test_fill_median_leaves_non_numeric_columns_unchanged():
    df = pd.DataFrame(
        {
            "price": [10.0, None, 30.0],
            "category": ["Book", None, "Pen"],
        }
    )

    result = handle_missing_values(
        df,
        strategy="fill_median",
    )

    expected = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0],
            "category": ["Book", None, "Pen"],
        }
    )

    pdt.assert_frame_equal(result, expected)


def test_fill_median_leaves_entirely_missing_numeric_column_unchanged():
    df = pd.DataFrame(
        {
            "price": pd.Series([None, None, None], dtype="float64"),
        }
    )

    result = handle_missing_values(
        df,
        strategy="fill_median",
    )

    expected = pd.DataFrame(
        {
            "price": pd.Series([None, None, None], dtype="float64"),
        }
    )

    pdt.assert_frame_equal(result, expected)


def test_fill_median_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "price": [10.0, None, 30.0],
        }
    )
    original = df.copy(deep=True)

    handle_missing_values(
        df,
        strategy="fill_median",
    )

    pdt.assert_frame_equal(df, original)


def test_fill_median_rejects_dataframe_without_numeric_columns():
    df = pd.DataFrame(
        {
            "category": ["Book", None, "Pen"],
        }
    )

    with pytest.raises(ValueError, match="numeric"):
        handle_missing_values(
            df,
            strategy="fill_median",
        )


def test_clean_dataframe_removes_duplicates_then_drops_missing_rows():
    df = pd.DataFrame({"id": [1, 1, 2], "amount": [10.0, 10.0, None]})
    result = clean_dataframe(df)
    assert result.to_dict(orient="records") == [{"id": 1, "amount": 10.0}]
    assert len(df) == 3


def test_clean_dataframe_can_fill_constant_without_deduplication():
    df = pd.DataFrame({"id": [1, 1], "status": ["ok", None]})
    result = clean_dataframe(
        df, remove_duplicates=False, missing_strategy="fill_constant", fill_value="unknown"
    )
    assert result["status"].tolist() == ["ok", "unknown"]


def test_clean_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError):
        clean_dataframe([])
