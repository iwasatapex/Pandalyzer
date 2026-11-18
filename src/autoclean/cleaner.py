import math
import re

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from autoclean._utils import (
    HORIZONTAL_SPACE,
    INVISIBLE,
    LINE_BREAKS,
    distinct_counts,
    duplicated_mask,
    is_text_dtype,
)
from autoclean.emails import email_columns, normalize_email_series
from autoclean.inference import infer_series_type

MISSING_VALUE_STRATEGIES = ("drop_rows", "fill_constant", "fill_mean", "fill_median")

# Compared case-insensitively after stripping whitespace.
DEFAULT_MISSING_MARKERS = frozenset(
    {
        "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--", "?",
        "unknown", "missing", "undefined", "not available", "tbd", "tba", "#n/a", "#na",
        "#value!", "#ref!", "#div/0!", "#null!", "#num!", "#name?", "nat", "<na>",
        "<null>", "(blank)", "(empty)", "(null)", "(none)", "\u2013", "\u2014",
    }
)  # fmt: skip

_AUTO_NAMED_COLUMN = re.compile(r"^Unnamed: \d+(?:_level_\d+)?$")
# Whitespace plus zero-width characters that web and Excel exports leave behind.
_EDGE_CHARS = HORIZONTAL_SPACE + LINE_BREAKS + INVISIBLE
_EDGE_SPACE = f"^[{_EDGE_CHARS}]+|[{_EDGE_CHARS}]+$"
# Runs of horizontal whitespace, and any single tab/NBSP; line breaks are kept.
_INNER_SPACE = f"[{HORIZONTAL_SPACE}]{{2,}}|[{HORIZONTAL_SPACE.removeprefix(' ')}]"
_EXCEL_TEXT = r'^="(.*)"$'  # Excel's ="00123" trick for keeping leading zeros.
_TOTAL_LABEL = re.compile(r"^(?:grand\s+|sub-?\s?)?totals?\b|^sum\b", re.IGNORECASE)


def _is_numeric_column(dtype) -> bool:
    return is_numeric_dtype(dtype) and not is_bool_dtype(dtype)


# --------------------------------------------------------------------------
# Duplicates and missing values
# --------------------------------------------------------------------------


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows while preserving the original DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return df[~duplicated_mask(df)].copy()


def _fill_with_statistic(df: pd.DataFrame, statistic: str) -> pd.DataFrame:
    """Fill missing values in numeric columns with each column's mean or median.

    Non-numeric (and boolean) columns are left untouched. Columns whose
    statistic is undefined (all values missing) are left as-is. Integer
    columns are upcast to a float dtype when the statistic is fractional,
    instead of failing on nullable integer dtypes such as ``Int64``.
    """
    numeric_positions = [
        position for position, dtype in enumerate(df.dtypes) if _is_numeric_column(dtype)
    ]

    if not numeric_positions:
        raise ValueError(
            f"{statistic.capitalize()} imputation requires at least one numeric column."
        )

    result = df.copy()
    for position in numeric_positions:
        _fill_position(result, position, statistic)
    return result


def _fill_position(df: pd.DataFrame, position: int, statistic: str) -> bool:
    """Fill one numeric column in place with its mean/median. Return True if filled."""
    series = df.iloc[:, position]
    if not series.isna().any():
        return False
    fill = getattr(series, statistic)()
    if pd.isna(fill):
        return False
    try:
        filled = series.fillna(fill)
    except (TypeError, ValueError):
        filled = series.astype("Float64").fillna(fill)
    df.isetitem(position, filled)
    return True


def _fill_series_constant(series: pd.Series, value) -> pd.Series:
    """``series.fillna(value)`` that also works when ``value`` doesn't fit the dtype."""
    try:
        return series.fillna(value)
    except (TypeError, ValueError):
        if isinstance(series.dtype, pd.CategoricalDtype):
            return series.cat.add_categories([value]).fillna(value)
        # e.g. "unknown" into Int64/boolean: fall back to a generic object column.
        return series.astype(object).fillna(value)


def _fill_constant(df: pd.DataFrame, value) -> pd.DataFrame:
    result = df.copy()
    for position, column in enumerate(result.columns):
        fill = value.get(column) if isinstance(value, dict) else value
        if fill is None:
            continue
        series = result.iloc[:, position]
        if series.isna().any():
            result.isetitem(position, _fill_series_constant(series, fill))
    return result


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop_rows",
    value=None,
) -> pd.DataFrame:
    """Handle missing values using the selected strategy.

    Strategies:
        ``drop_rows``      drop any row that contains a missing value.
        ``fill_constant``  replace missing values with ``value`` (required); a
                           dict maps column names to per-column fill values.
        ``fill_mean``      fill numeric columns with their mean.
        ``fill_median``    fill numeric columns with their median.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if strategy == "drop_rows":
        return df.dropna().copy()

    if strategy == "fill_constant":
        if value is None:
            raise ValueError("The 'fill_constant' strategy requires a fill value.")
        return _fill_constant(df, value)

    if strategy == "fill_mean":
        return _fill_with_statistic(df, "mean")

    if strategy == "fill_median":
        return _fill_with_statistic(df, "median")

    raise ValueError(
        f"Unsupported missing-value strategy: {strategy!r}. "
        f"Expected one of: {', '.join(MISSING_VALUE_STRATEGIES)}."
    )


def clean_dataframe(
    df: pd.DataFrame,
    *,
    remove_duplicates: bool = True,
    missing_strategy: str = "drop_rows",
    fill_value=None,
) -> pd.DataFrame:
    """Apply the duplicate and missing-value cleaners in sequence.

    For messy real-world files prefer :func:`auto_clean`, which also fixes
    whitespace, placeholder values, case variants and column types.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    result = remove_duplicate_rows(df) if remove_duplicates else df
    return handle_missing_values(result, strategy=missing_strategy, value=fill_value)


# --------------------------------------------------------------------------
# Text and schema normalisation
# --------------------------------------------------------------------------


def _map_strings(series: pd.Series, func) -> pd.Series:
    """Apply ``func`` to the string values of a text column, leaving other values alone."""
    if isinstance(series.dtype, pd.StringDtype):
        return func(series.str)
    # pandas 3 infers "str" for a map that returns strings; keep the column's dtype.
    return series.map(lambda value: func(value) if isinstance(value, str) else value).astype(
        series.dtype
    )


def _require_dataframe(df) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")


def clean_column_names(df: pd.DataFrame, style: str = "strip") -> pd.DataFrame:
    """Return a copy with tidied column names.

    ``style="strip"`` trims whitespace and collapses inner runs (including line
    breaks in Excel headers) to one space; ``style="snake"`` also
    lower-cases and replaces non-alphanumeric runs with ``_`` (``"Order ID"``
    becomes ``"order_id"``). Blank names become ``column_<n>`` and names that
    collide get ``_2``, ``_3``… suffixes, so the result always has unique names.
    """
    _require_dataframe(df)
    if style not in ("strip", "snake"):
        raise ValueError("style must be 'strip' or 'snake'.")

    names, seen = [], set()
    for position, column in enumerate(df.columns, start=1):
        name = " ".join(str(column).split())
        if style == "snake":
            name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
        name = name or f"column_{position}"
        candidate, suffix = name, 2
        while candidate in seen:
            candidate, suffix = f"{name}_{suffix}", suffix + 1
        seen.add(candidate)
        names.append(candidate)

    result = df.copy()
    result.columns = names
    return result


def _replace_text(series: pd.Series, pattern: str, replacement: str) -> pd.Series:
    if isinstance(series.dtype, pd.StringDtype):
        return series.str.replace(pattern, replacement, regex=True)
    compiled = re.compile(pattern)
    return _map_strings(series, lambda value: compiled.sub(replacement, value))


def _map_text_columns(df: pd.DataFrame, func) -> pd.DataFrame:
    result = df.copy()
    for position in range(result.shape[1]):
        series = result.iloc[:, position]
        if is_text_dtype(series.dtype):
            result.isetitem(position, func(series))
    return result


def strip_whitespace(df: pd.DataFrame, *, collapse: bool = True) -> pd.DataFrame:
    """Return a copy with whitespace trimmed from text values.

    Trims spaces, tabs, non-breaking spaces and zero-width characters at both
    ends. With ``collapse`` (default) runs of spaces/tabs inside a value become
    one space (``"John   Smith"`` -> ``"John Smith"``); line breaks are kept.
    """
    _require_dataframe(df)

    def tidy(series):
        series = _replace_text(series, _EDGE_SPACE, "")
        return _replace_text(series, _INNER_SPACE, " ") if collapse else series

    return _map_text_columns(df, tidy)


def unwrap_excel_text(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where Excel text formulas such as ``="00123"`` become ``00123``."""
    _require_dataframe(df)
    return _map_text_columns(df, lambda series: _replace_text(series, _EXCEL_TEXT, r"\1"))


def standardize_missing_values(
    df: pd.DataFrame,
    markers=DEFAULT_MISSING_MARKERS,
) -> pd.DataFrame:
    """Return a copy where placeholder text such as ``"N/A"``, ``"unknown"`` or ``""`` is missing.

    Matching ignores case and surrounding whitespace. Pass your own ``markers``
    to change what counts as missing.
    """
    _require_dataframe(df)
    markers = {str(marker).strip().casefold() for marker in markers}
    result = df.copy()
    for position in range(result.shape[1]):
        series = result.iloc[:, position]
        if not is_text_dtype(series.dtype):
            continue
        if isinstance(series.dtype, pd.StringDtype):
            is_marker = series.str.strip().str.casefold().isin(markers).fillna(False)
        else:
            is_marker = series.map(
                lambda value: isinstance(value, str) and value.strip().casefold() in markers
            )
        is_marker = is_marker.astype(bool)
        if is_marker.any():
            result.isetitem(position, series.mask(is_marker))
    return result


def _case_variant_mapping(series: pd.Series) -> dict:
    # sort=False keeps first-appearance order, so ties go to the earliest spelling.
    counts = series.value_counts(dropna=True, sort=False)
    canonical, mapping = {}, {}
    for spelling in counts.sort_values(ascending=False, kind="stable").index:
        if not isinstance(spelling, str):
            continue
        key = " ".join(spelling.split()).casefold()
        if key not in canonical:
            canonical[key] = spelling
        elif spelling != canonical[key]:
            mapping[spelling] = canonical[key]
    return mapping


def unify_case_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where text values differing only by case/spacing use one spelling.

    ``"Cash"``, ``"CASH"`` and ``"cash"`` all become whichever spelling is most
    common in the column. Distinct words (``"COD"`` vs ``"Cash on Delivery"``)
    are not merged; use ``DataFrame.replace`` for synonyms.
    """
    _require_dataframe(df)
    result = df.copy()
    for position in range(result.shape[1]):
        series = result.iloc[:, position]
        if not is_text_dtype(series.dtype):
            continue
        try:
            mapping = _case_variant_mapping(series)
        except TypeError:  # Unhashable values (lists, dicts) in an object column.
            continue
        if mapping:
            # One dict-lookup pass; Series.replace scans once per mapping key.
            replacement = series.map(mapping)
            result.isetitem(position, series.mask(replacement.notna(), replacement))
    return result


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without rows in which every value is missing."""
    _require_dataframe(df)
    return df.dropna(how="all").copy()


def _header_keys(name) -> set[str]:
    text = " ".join(str(name).split()).casefold()
    # pandas renames a repeated header "id" to "id.1"; the data row still says "id".
    return {text, re.sub(r"\.\d+$", "", text)}


def drop_repeated_headers(df: pd.DataFrame, header=None) -> pd.DataFrame:
    """Return a copy without rows that repeat the header (left by concatenated exports).

    A row counts as a header when every value in it matches its column's name
    (ignoring case and spacing) and at least half of the columns are filled.
    ``header`` gives the names to compare with; it defaults to ``df.columns``
    (pass the original names if the columns were renamed).
    """
    _require_dataframe(df)
    header = list(df.columns if header is None else header)
    if len(header) != df.shape[1]:
        raise ValueError("header must have one name per column.")
    if df.empty:
        return df.copy()
    matches = pd.Series(0, index=df.index)
    mismatch = pd.Series(False, index=df.index)
    for position, name in enumerate(header):
        series = df.iloc[:, position]
        present = series.notna().astype(bool)
        text = series.map(lambda value: " ".join(str(value).split()).casefold())
        same = text.isin(_header_keys(name)) & present
        matches += same.astype(int)
        mismatch |= present & ~same
    is_header = ~mismatch & (matches >= max(1, math.ceil(df.shape[1] / 2)))
    return df[~is_header].copy()


def drop_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without the summary rows at the end of a report export.

    Only trailing rows are considered (empty rows between them are skipped):
    a row is a summary when its first filled cell starts with "Total",
    "Subtotal", "Grand total" or "Sum". Matching rows elsewhere are kept.
    """
    _require_dataframe(df)
    keep = [True] * len(df)
    for position in range(len(df) - 1, -1, -1):  # Stops at the first ordinary row.
        filled = [value for value in df.iloc[position] if not _is_missing_scalar(value)]
        if not filled:
            continue
        if not (isinstance(filled[0], str) and _TOTAL_LABEL.match(filled[0].strip())):
            break
        keep[position] = False
    return df.iloc[[p for p, kept in enumerate(keep) if kept]].copy()


def _is_missing_scalar(value) -> bool:
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def normalize_emails(df: pd.DataFrame, columns=None, *, invalid: str = "missing") -> pd.DataFrame:
    """Return a copy with email addresses trimmed, lower-cased and validated.

    ``columns`` defaults to every column that looks like an email column (nearly
    all values contain "@" and most are valid). ``mailto:`` prefixes and display
    names (``"Ann <ann@x.com>"``) are removed. Malformed addresses become missing
    with ``invalid="missing"`` (default) or stay as they are with ``"keep"``;
    use :func:`autoclean.validators.validate_emails` to list them.
    """
    _require_dataframe(df)
    columns = email_columns(df) if columns is None else list(columns)
    result = df.copy()
    for position, column in enumerate(result.columns):
        if column in columns:
            result.isetitem(
                position, normalize_email_series(result.iloc[:, position], invalid=invalid)
            )
    return result


# --------------------------------------------------------------------------
# Automatic pipeline
# --------------------------------------------------------------------------


def _is_low_cardinality(series: pd.Series) -> bool:
    present = series.notna().sum()
    if present == 0:
        return False
    distinct = int(distinct_counts(series.to_frame()).iloc[0])
    return distinct <= 50 and distinct <= present / 2


def fill_missing_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using a sensible default per column type.

    - Numeric columns: the median.
    - Boolean columns and low-cardinality text/category columns (at most 50
      distinct values, each used twice on average): the most frequent value.
    - Datetimes, free text and identifiers: left missing, since any guess
      would invent data.
    """
    return _fill_missing_auto(df, log=None)


def _fill_missing_auto(df: pd.DataFrame, log: list | None) -> pd.DataFrame:
    _require_dataframe(df)
    result = df.copy()
    for position, column in enumerate(result.columns):
        series = result.iloc[:, position]
        missing = int(series.isna().sum())
        if not missing:
            continue
        dtype = series.dtype
        if _is_numeric_column(dtype):
            if _fill_position(result, position, "median") and log is not None:
                log.append(_entry("fill_missing", column, f"filled {missing} with median"))
            continue
        if is_datetime64_any_dtype(dtype):
            continue
        is_categorical = isinstance(dtype, pd.CategoricalDtype)
        if is_bool_dtype(dtype) or (
            (is_text_dtype(dtype) or is_categorical) and _is_low_cardinality(series)
        ):
            try:
                modes = series.mode(dropna=True)
            except TypeError:  # Unhashable values (lists, dicts).
                continue
            if len(modes) and pd.api.types.is_scalar(modes.iloc[0]):
                result.isetitem(position, _fill_series_constant(series, modes.iloc[0]))
                if log is not None:
                    log.append(
                        _entry("fill_missing", column, f"filled {missing} with {modes.iloc[0]!r}")
                    )
    return result


def _entry(step: str, column, detail: str) -> dict:
    return {"step": step, "column": None if column is None else str(column), "detail": detail}


def _changed_cells(before: pd.Series, after: pd.Series) -> int:
    same = (before == after).fillna(False).astype(bool) | (before.isna() & after.isna())
    return int((~same).sum())


def _per_column_step(df, func, step, log, describe):
    """Run a column-wise cleaning function and log which columns changed."""
    result = func(df)
    for position, column in enumerate(result.columns):
        before, after = df.iloc[:, position], result.iloc[:, position]
        if before.dtype != after.dtype:
            changed = len(before)
        else:
            try:
                changed = _changed_cells(before, after)
            except (TypeError, ValueError):  # Uncomparable cell values (e.g. arrays).
                changed = 0
        if changed:
            log.append(_entry(step, column, describe(changed)))
    return result


def auto_clean(
    df: pd.DataFrame,
    *,
    column_names: str | None = "strip",
    strip_text: bool = True,
    missing_markers=DEFAULT_MISSING_MARKERS,
    drop_artifacts: bool = True,
    emails: str | None = "missing",
    unify_case: bool = True,
    infer_types: bool = True,
    type_threshold: float = 0.95,
    dayfirst: bool = False,
    remove_duplicates: bool = True,
    drop_empty: bool = True,
    missing: str = "keep",
    fill_value=None,
    return_log: bool = False,
):
    """Clean any DataFrame without knowing its schema in advance.

    Steps, in order (each can be switched off):

    1. ``column_names``: ``"strip"`` (default) or ``"snake"`` tidy the
       headers and make them unique; ``None`` leaves them alone.
    2. ``strip_text``: trim whitespace and zero-width characters around text
       values, collapse inner runs of spaces, and unwrap Excel ``="00123"``.
    3. ``missing_markers``: turn placeholders like ``"N/A"``, ``"unknown"`` or
       ``""`` into real missing values (``None`` to skip).
    4. ``drop_artifacts``: drop rows that repeat the header (concatenated
       exports) and trailing "Total"/"Sum" summary rows.
    5. ``emails``: in columns of email addresses, trim and lower-case them;
       malformed addresses become missing (``"missing"``, default), stay as
       they are (``"keep"``), or the step is skipped (``None``).
    6. ``unify_case``: merge spellings that differ only by case/spacing.
    7. ``infer_types``: convert text columns to boolean, integer (``Int64``),
       float or datetime when at least ``type_threshold`` of their values
       parse. Handles ``"₹1,234.50"``, ``"USD 1 234.50"``, ``"1'234.50"``,
       ``"15%"``, ``"(42)"``, ``"Yes"/"No"``, ``"on"/"off"`` and mixed date
       formats. Leading-zero codes such as ``"00123"`` stay text.
    8. ``drop_empty``: drop rows that are entirely empty, and empty columns
       pandas auto-named ``"Unnamed: N"`` (from trailing delimiters).
    9. ``remove_duplicates``: drop exact duplicate rows (after the fixes above,
       so near-duplicates that differed only by spacing/case are caught).
    10. ``missing``: ``"keep"`` (default) leaves missing values in place;
       ``"auto"`` uses :func:`fill_missing_auto`; or any
       :func:`handle_missing_values` strategy (``"drop_rows"``,
       ``"fill_mean"``, ``"fill_median"``, ``"fill_constant"`` + ``fill_value``).

    Columns that already have a non-text dtype (int, float, bool, datetime,
    category…) are never converted, only deduplicated and filled.

    Returns the cleaned DataFrame, or ``(DataFrame, log)`` when ``return_log``
    is true. The log is a list of ``{"step", "column", "detail"}`` dicts.
    """
    _require_dataframe(df)
    if missing not in ("keep", "auto", *MISSING_VALUE_STRATEGIES):
        raise ValueError(
            f"Unsupported missing option: {missing!r}. Expected 'keep', 'auto' or one of: "
            f"{', '.join(MISSING_VALUE_STRATEGIES)}."
        )
    if emails not in ("missing", "keep", None):
        raise ValueError("emails must be 'missing', 'keep' or None.")
    log: list[dict] = []
    result = df.copy()
    original_header = list(df.columns)

    if column_names is not None:
        renamed = clean_column_names(result, style=column_names)
        for old, new in zip(result.columns, renamed.columns, strict=True):
            if str(old) != new:
                log.append(_entry("rename_column", old, f"renamed to {new!r}"))
        result = renamed

    if strip_text:
        result = _per_column_step(
            result, strip_whitespace, "strip_whitespace", log,
            lambda n: f"trimmed whitespace in {n} values",
        )  # fmt: skip
        result = _per_column_step(
            result, unwrap_excel_text, "unwrap_excel_text", log,
            lambda n: f'unwrapped {n} Excel ="..." values',
        )  # fmt: skip

    if missing_markers:
        result = _per_column_step(
            result, lambda d: standardize_missing_values(d, missing_markers),
            "standardize_missing", log,
            lambda n: f"{n} placeholder values set to missing",
        )  # fmt: skip

    if drop_artifacts:
        rows_before = len(result)
        result = drop_repeated_headers(result, header=original_header)
        if len(result) < rows_before:
            log.append(
                _entry("drop_artifacts", None, f"dropped {rows_before - len(result)} header rows")
            )
        rows_before = len(result)
        result = drop_total_rows(result)
        if len(result) < rows_before:
            log.append(
                _entry("drop_artifacts", None, f"dropped {rows_before - len(result)} total rows")
            )

    if emails is not None:
        for position, column in enumerate(result.columns):
            series = result.iloc[:, position]
            if column not in email_columns(series.to_frame()):
                continue
            normalized = normalize_email_series(series, invalid=emails)
            invalid = int(normalized.isna().sum() - series.isna().sum())
            changed = _changed_cells(series, normalized) - invalid
            detail = f"normalized {changed} addresses"
            if invalid:
                detail += f"; {invalid} invalid addresses set to missing"
            if changed or invalid:
                log.append(_entry("normalize_emails", column, detail))
                result.isetitem(position, normalized)

    if unify_case:
        result = _per_column_step(
            result, unify_case_variants, "unify_case", log,
            lambda n: f"{n} values respelled to the most common variant",
        )  # fmt: skip

    if infer_types:
        for position, column in enumerate(result.columns):
            series = result.iloc[:, position]
            converted, kind = infer_series_type(series, threshold=type_threshold, dayfirst=dayfirst)
            if kind is None:
                continue
            lost = int(converted.isna().sum() - series.isna().sum())
            detail = f"converted to {kind} ({converted.dtype})"
            if lost:
                detail += f"; {lost} unparseable values set to missing"
            log.append(_entry("infer_type", column, detail))
            result.isetitem(position, converted)

    if drop_empty:
        empty_auto_named = [
            column
            for position, column in enumerate(result.columns)
            if _AUTO_NAMED_COLUMN.match(str(column)) and result.iloc[:, position].isna().all()
        ]
        if empty_auto_named:
            result = result.drop(columns=empty_auto_named)
            for column in empty_auto_named:
                log.append(_entry("drop_empty", column, "dropped empty auto-named column"))
        rows_before = len(result)
        result = drop_empty_rows(result)
        if len(result) < rows_before:
            log.append(
                _entry("drop_empty", None, f"dropped {rows_before - len(result)} empty rows")
            )

    if remove_duplicates:
        rows_before = len(result)
        result = remove_duplicate_rows(result)
        if len(result) < rows_before:
            log.append(
                _entry(
                    "remove_duplicates", None, f"dropped {rows_before - len(result)} duplicate rows"
                )
            )

    if missing == "auto":
        result = _fill_missing_auto(result, log)
    elif missing != "keep":
        rows_before = len(result)
        result = handle_missing_values(result, strategy=missing, value=fill_value)
        log.append(_entry("handle_missing", None, f"applied {missing!r}"))
        if len(result) < rows_before:
            log[-1]["detail"] += f"; dropped {rows_before - len(result)} rows"

    return (result, log) if return_log else result
