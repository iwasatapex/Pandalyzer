"""Infer the most specific dtype (number, boolean, datetime) for each column."""

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_float_dtype, is_object_dtype

from autoclean._utils import is_text_dtype

TRUE_TOKENS = frozenset({"true", "t", "yes", "y"})
FALSE_TOKENS = frozenset({"false", "f", "no", "n"})
_BINARY_DIGITS = frozenset({"1", "0", "1.0", "0.0"})

_CURRENCY = r"[$€£₹¥]"
_GROUPED_NUMBER = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
_EUROPEAN_NUMBER = r"[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?"
_LEADING_ZERO = r"[+-]?0\d"  # "007", "00123": identifiers, not numbers.
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
# Requires a 4-digit year (or "/"/"-" with a 2-digit year, or a month name), so
# version numbers ("1.2.3") and IP addresses ("10.0.0.1") don't look like dates.
_DATE_LIKE = (
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2}(?!\d)"
    r"|\d{1,2}-\d{1,2}-\d{2}(?!\d)"
    rf"|\d{{1,2}}[-/ ]?{_MONTH}"
    rf"|{_MONTH}[a-z]*\.? \d{{1,2}})"
)
_MIN_DATE_PARSE_RATE = 0.5  # Guard against date-looking text that isn't dates.
_FLOAT_EXACT_LIMIT = 2**53  # Largest range where float64 holds every integer exactly.
_INT64_MAX = np.iinfo(np.int64).max


def _as_text(series: pd.Series) -> pd.Series:
    """Non-missing values as stripped strings in a ``string`` Series."""
    return series.dropna().astype("string").str.strip()


def _strip_number_decorations(text: pd.Series) -> pd.Series:
    """Remove currency symbols and % and turn accounting "(42)" into "-42"."""
    cleaned = text.str.replace(r"^\((.+)\)$", r"-\1", regex=True)
    cleaned = cleaned.str.replace(rf"^([+-]?)\s*{_CURRENCY}\s*", r"\1", regex=True)
    return cleaned.str.replace(rf"\s*(?:{_CURRENCY}|%)$", "", regex=True)


def _parse_numbers(text: pd.Series) -> pd.Series:
    """Parse numbers written with currency symbols, "1,234.5" grouping, % or (negatives)."""
    cleaned = _strip_number_decorations(text)
    grouped = cleaned.str.fullmatch(_GROUPED_NUMBER).fillna(False)
    cleaned = cleaned.where(~grouped, cleaned.str.replace(",", "", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_european_numbers(text: pd.Series) -> pd.Series:
    """Parse numbers written with a decimal comma: "3,5", "1.234,50", "€ 1.200"."""
    cleaned = _strip_number_decorations(text)
    valid = cleaned.str.fullmatch(_EUROPEAN_NUMBER).fillna(False)
    cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned.where(valid), errors="coerce")


def _as_boolean(text: pd.Series, threshold: float):
    tokens = text.str.casefold()
    if not tokens.isin(TRUE_TOKENS | FALSE_TOKENS).any():
        return None  # Pure 0/1 columns stay numeric.
    valid = tokens.isin(TRUE_TOKENS | FALSE_TOKENS | _BINARY_DIGITS)
    if valid.mean() < threshold:
        return None
    values = tokens.isin(TRUE_TOKENS | {"1", "1.0"})
    return values.where(valid).astype("boolean")


def _as_number(text: pd.Series, threshold: float):
    numbers = _parse_numbers(text)
    if numbers.notna().mean() < threshold:
        # Only if the usual "1,234.5" reading fails: try the decimal-comma
        # convention used across Europe, consistently for the whole column.
        numbers = _parse_european_numbers(text)
        if numbers.notna().mean() < threshold:
            return None
    if text[numbers.notna()].str.match(_LEADING_ZERO).any():
        return None
    return _narrow_numbers(numbers.astype("float64"))


def _narrow_numbers(numbers: pd.Series):
    """Return ``Int64`` when every value is whole, float otherwise; None if precision would be lost."""
    present = numbers.dropna()
    if (present.abs() > _FLOAT_EXACT_LIMIT).any():
        return None  # Long digit strings (card numbers, big IDs) would lose precision.
    if (present % 1 == 0).all():
        return numbers.astype("Int64")
    return numbers


def _as_datetime(text: pd.Series, threshold: float, dayfirst: bool):
    # Decide on shape, not parse success: impossible dates such as
    # "2024-02-30" or "0000-00-00" should become missing, not block conversion.
    if text.str.casefold().str.contains(_DATE_LIKE, regex=True).mean() < threshold:
        return None
    dates = _parse_dates(text, dayfirst, utc=False)
    if dates is None:
        # Mixed time-zone offsets (or aware + naive): normalise everything to UTC.
        dates = _parse_dates(text, dayfirst, utc=True)
    if dates.notna().mean() < _MIN_DATE_PARSE_RATE:
        return None
    return dates


def _parse_dates(text: pd.Series, dayfirst: bool, *, utc: bool):
    """Parse mixed-format dates; None if the result can't share one datetime dtype.

    Year-first values ("2024-03-05") are always read as year-month-day: dateutil
    would otherwise turn them into 2024-05-03 when ``dayfirst`` is set.
    """
    year_first = text.str.match(r"\d{4}[-/.]").fillna(False).astype(bool)
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mask, first in ((year_first, False), (~year_first, dayfirst)):
            if not mask.any():
                continue
            try:
                parsed = pd.to_datetime(
                    text[mask], errors="coerce", format="mixed", dayfirst=first, utc=utc
                )
            except (TypeError, ValueError):
                return None
            if not is_datetime64_any_dtype(parsed.dtype):
                return None  # Older pandas returns object for mixed offsets.
            parts.append(parsed)
    if len(parts) == 2 and parts[0].dtype != parts[1].dtype:
        if utc:
            parts = [part.astype(parts[0].dtype) for part in parts]
        else:
            return None
    return pd.concat(parts).reindex(text.index)


def _narrow_typed_column(series: pd.Series):
    """Fix non-text columns that pandas loaded with an awkward dtype.

    - float columns holding only whole numbers (an integer column with blanks,
      which pandas reads as float64) become ``Int64``;
    - object columns of Python ints too big for int64 (long IDs) become text.
    """
    present = series.dropna()
    if present.empty:
        return None, None
    if is_float_dtype(series.dtype):
        narrowed = _narrow_numbers(series)
        if narrowed is not None and str(narrowed.dtype) == "Int64":
            return narrowed, "integer"
        return None, None
    if is_object_dtype(series.dtype) and present.map(
        lambda v: isinstance(v, int) and not isinstance(v, bool)
    ).all():
        if present.abs().max() > _INT64_MAX:
            # Same text dtype pandas gives any column of strings.
            as_text = series.map(str, na_action="ignore")
            return as_text.astype(pd.Series(["x"]).dtype), "text"
        return series.astype("Int64"), "integer"
    return None, None


def infer_series_type(
    series: pd.Series,
    *,
    threshold: float = 0.95,
    dayfirst: bool = False,
) -> tuple[pd.Series, str | None]:
    """Give a Series the most specific dtype its values allow.

    Text columns become boolean, integer (``Int64``), float or datetime when
    at least ``threshold`` of the non-missing values parse as that type; the
    rest become missing. Numbers may use "1,234.5" or European "1.234,5"
    notation, currency symbols, % or accounting "(42)" negatives.

    Typed columns are only narrowed: whole-number floats (integer columns
    pandas loaded as float because of blanks) become ``Int64``, and object
    columns of integers too big for int64 become text.

    Returns ``(series, kind)`` where kind is ``"boolean"``, ``"integer"``,
    ``"float"``, ``"datetime"``, ``"text"`` or ``None`` when nothing changed.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    if not is_text_dtype(series.dtype) or (
        is_object_dtype(series.dtype) and not series.dropna().map(type).eq(str).any()
    ):
        narrowed, kind = _narrow_typed_column(series)
        return (series, None) if narrowed is None else (narrowed.rename(series.name), kind)
    text = _as_text(series)
    text = text[text != ""]
    if text.empty:
        return series, None

    for kind, convert in (
        ("boolean", lambda: _as_boolean(text, threshold)),
        ("number", lambda: _as_number(text, threshold)),
        ("datetime", lambda: _as_datetime(text, threshold, dayfirst)),
    ):
        converted = convert()
        if converted is None:
            continue
        result = converted.reindex(series.index)
        if kind == "number":
            kind = "integer" if str(result.dtype) == "Int64" else "float"
        return result.rename(series.name), kind
    return series, None


def infer_column_types(
    df: pd.DataFrame,
    *,
    threshold: float = 0.95,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Return a copy of ``df`` with text columns converted to their inferred types.

    See :func:`infer_series_type` for the rules.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    result = df.copy()
    for position in range(result.shape[1]):
        converted, kind = infer_series_type(
            result.iloc[:, position], threshold=threshold, dayfirst=dayfirst
        )
        if kind is not None:
            result.isetitem(position, converted)
    return result
