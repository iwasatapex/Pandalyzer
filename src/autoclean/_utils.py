"""Internal helpers shared across autoclean modules."""

from collections.abc import Hashable

import pandas as pd
from pandas.api.types import is_object_dtype

# Regex character-class contents written as literal characters, not "\\s" or
# "\\u" escapes: pandas runs regexes on pyarrow strings with RE2, whose "\\s"
# is ASCII-only and which rejects "\\u" escapes, lookarounds and backreferences.
HORIZONTAL_SPACE = " \t\f\v\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000"
LINE_BREAKS = "\n\r\u2028\u2029"
INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff"  # Zero-width characters and stray BOMs.


def is_text_dtype(dtype) -> bool:
    """True for object and string dtypes (``object``, ``string``, pandas 3 ``str``)."""
    return is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)


def _hashable_column(series: pd.Series) -> pd.Series:
    if not is_object_dtype(series.dtype):
        return series
    if series.map(lambda value: isinstance(value, Hashable)).all():
        return series
    return series.map(lambda value: value if isinstance(value, Hashable) else repr(value))


def hashable_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with unhashable cells (lists, dicts, sets) replaced by their repr.

    Used only for comparisons such as duplicate detection and distinct counts,
    which pandas cannot compute on unhashable values.
    """
    result = df.copy()
    for position in range(result.shape[1]):
        result.isetitem(position, _hashable_column(result.iloc[:, position]))
    return result


def duplicated_mask(df: pd.DataFrame) -> pd.Series:
    """Like ``df.duplicated()``, but tolerant of unhashable cell values."""
    try:
        return df.duplicated()
    except TypeError:
        return hashable_frame(df).duplicated()


def distinct_counts(df: pd.DataFrame) -> pd.Series:
    """Like ``df.nunique(dropna=True)``, but tolerant of unhashable cell values."""
    try:
        return df.nunique(dropna=True)
    except TypeError:
        return hashable_frame(df).nunique(dropna=True)
