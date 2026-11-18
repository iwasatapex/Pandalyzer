"""Recognise, normalise and validate email address columns."""

import re

import pandas as pd

from autoclean._utils import INVISIBLE, is_text_dtype

# Practical subset of RFC 5322: dot-separated atoms before the "@", and a
# domain of hyphenated labels ending in an alphabetic TLD. Quoted local parts
# and IP-literal domains are rejected; they are almost always data errors.
_ATOM = r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+"
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_VALID_EMAIL = re.compile(rf"{_ATOM}(?:\.{_ATOM})*@(?:{_LABEL}\.)+[a-z]{{2,63}}")
_DROP_INVISIBLE = dict.fromkeys(map(ord, INVISIBLE))
_DISPLAY_NAME = re.compile(r"^[^<>]*<([^<>]+)>$")  # "Ann Lee <ann@example.com>"

# A column is an email column when at least 80% of its values contain "@" and
# most are valid addresses (so a column of "@handles" doesn't qualify).
_AT_SIGN_SHARE = 0.8
_VALID_SHARE = 0.5


def normalize_email(value: str) -> str:
    """Return ``value`` as a bare, lower-case address: no ``mailto:``, display name or spaces."""
    text = value.strip()
    match = _DISPLAY_NAME.match(text)
    if match:
        text = match.group(1)
    if text[:7].casefold() == "mailto:":
        text = text[7:]
    text = text.translate(_DROP_INVISIBLE)
    # Trailing punctuation comes from prose ("write to ann@x.com."); a leading dot is an error.
    return "".join(text.split()).strip("<>").rstrip(".,;").casefold()


def is_valid_email(value) -> bool:
    """True if ``value`` is a well-formed email address (after :func:`normalize_email`)."""
    if not isinstance(value, str):
        return False
    return _VALID_EMAIL.fullmatch(normalize_email(value)) is not None


def _present_strings(series: pd.Series) -> pd.Series:
    if not is_text_dtype(series.dtype):
        return series.iloc[0:0]
    present = series.dropna()
    return present[present.map(lambda value: isinstance(value, str) and bool(value.strip()))]


def is_email_column(series: pd.Series) -> bool:
    """True if ``series`` holds email addresses, judged from its values alone."""
    values = _present_strings(series)
    if values.empty:
        return False
    if values.str.contains("@", regex=False).mean() < _AT_SIGN_SHARE:
        return False
    return values.map(is_valid_email).mean() >= _VALID_SHARE


def email_columns(df: pd.DataFrame) -> list:
    """Names of the columns of ``df`` that hold email addresses."""
    return [
        column
        for position, column in enumerate(df.columns)
        if is_email_column(df.iloc[:, position])
    ]


def normalize_email_series(series: pd.Series, *, invalid: str = "missing") -> pd.Series:
    """Normalise every address in ``series``; invalid ones become missing or stay as they are.

    ``invalid="missing"`` (default) replaces malformed addresses with a missing
    value; ``invalid="keep"`` leaves them untouched so they can be reviewed.
    """
    if invalid not in ("missing", "keep"):
        raise ValueError("invalid must be 'missing' or 'keep'.")

    def fix(value):
        if not isinstance(value, str):
            return value
        address = normalize_email(value)
        if _VALID_EMAIL.fullmatch(address):
            return address
        return pd.NA if invalid == "missing" else value

    # pandas 3 infers "str" for a map that returns strings; keep the column's dtype.
    return series.map(fix).astype(series.dtype)
