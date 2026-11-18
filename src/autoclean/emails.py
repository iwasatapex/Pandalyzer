"""Recognise, normalise and validate email address columns."""

import re

import numpy as np
import pandas as pd

from autoclean._utils import INVISIBLE, is_text_dtype

# Practical subset of RFC 5322 plus internationalised addresses (RFC 6531):
# dot-separated atoms before the "@" (letters in any script allowed), and a
# domain of hyphenated labels ending in a letter TLD or a punycode one
# ("xn--p1ai"). Quoted local parts, IP-literal domains and spaces are rejected;
# in real exports they are almost always data errors.
_ATOM = r"[\w!#$%&'*+/=?^`{|}~-]+"
_LABEL = r"[^\W_](?:(?:[^\W_]|-){0,61}[^\W_])?"  # Letters/digits in any script, inner "-".
_TLD = r"(?:[^\W\d_]{2,63}|xn--[a-z0-9-]{1,59})"
_VALID_EMAIL = re.compile(rf"{_ATOM}(?:\.{_ATOM})*@(?:{_LABEL}\.)+{_TLD}")
# Phone numbers often share a "contact" column with emails; they are kept.
_PHONE = re.compile(r"\+?[\d\s().\-/]{7,20}")
_SAMPLE = 5000  # Values checked when deciding whether a column holds emails.
_DROP_INVISIBLE = dict.fromkeys(map(ord, INVISIBLE))
_DISPLAY_NAME = re.compile(r"^[^<>]*<([^<>]+)>$")  # "Ann Lee <ann@example.com>"

# A column is an email column when at least 80% of its values contain "@" and
# most are valid addresses (so a column of "@handles" doesn't qualify).
_AT_SIGN_SHARE = 0.8
_VALID_SHARE = 0.5


def normalize_email(value: str) -> str:
    """Return ``value`` as a bare, lower-case address without ``mailto:`` or a display name.

    Only surrounding whitespace is removed: an address with a space inside
    (``"john smith@x.com"``) stays invalid rather than becoming a different one.
    """
    text = value.translate(_DROP_INVISIBLE).strip()
    match = _DISPLAY_NAME.match(text)
    if match:
        text = match.group(1).strip()
    if text[:7].casefold() == "mailto:":
        text = text[7:].strip()
    # Trailing punctuation comes from prose ("write to ann@x.com."); a leading dot is an error.
    # lower(), not casefold(): casefold turns "ß" into "ss", which is a different mailbox.
    return text.strip("<>").rstrip(".,;").lower()


def looks_like_phone(value) -> bool:
    """True for values such as ``"+1 (555) 123-4567"``: 7 to 15 digits and phone punctuation."""
    if not isinstance(value, str) or not _PHONE.fullmatch(value.strip()):
        return False
    return 7 <= sum(character.isdigit() for character in value) <= 15


def is_valid_email(value) -> bool:
    """True if ``value`` is a well-formed email address (after :func:`normalize_email`)."""
    if not isinstance(value, str):
        return False
    return _VALID_EMAIL.fullmatch(normalize_email(value)) is not None


def _present_strings(series: pd.Series) -> pd.Series:
    if not is_text_dtype(series.dtype):
        return series.iloc[0:0]
    present = series.dropna()
    if isinstance(present.dtype, pd.StringDtype):
        return present[present.str.strip() != ""]
    return present[present.map(lambda value: isinstance(value, str) and bool(value.strip()))]


def is_email_column(series: pd.Series) -> bool:
    """True if ``series`` holds email addresses, judged from its values alone."""
    values = _present_strings(series)
    if values.empty:
        return False
    if values.str.contains("@", regex=False).mean() < _AT_SIGN_SHARE:
        return False
    if len(values) > _SAMPLE:
        values = values.sample(_SAMPLE, random_state=0)
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
    Values that look like phone numbers are always kept: they are contact
    details in the wrong place, not broken addresses.
    """
    if invalid not in ("missing", "keep"):
        raise ValueError("invalid must be 'missing' or 'keep'.")

    def fix(value):
        if not isinstance(value, str):
            return value
        address = normalize_email(value)
        if _VALID_EMAIL.fullmatch(address):
            return address
        if invalid == "keep" or looks_like_phone(value):
            return value
        # NaN, as Series.mask gives object columns; string dtypes convert it to their NA.
        return np.nan

    # pandas 3 infers "str" for a map that returns strings; keep the column's dtype.
    return series.map(fix).astype(series.dtype)
