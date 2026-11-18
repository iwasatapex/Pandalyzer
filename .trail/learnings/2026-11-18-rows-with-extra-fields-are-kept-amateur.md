---
title: "Rows with extra fields are kept with an on_bad_lines callable, because pandas 3.1 skips them with on_bad_lines=\"warn\""
author: Amateur
date: 2026-11-18
branch: audit-universal-cleaning
cost: { minutes: 22, tokens: 195407 }
id: l-a0a8e11d
scope: repo
status: active
src: agent
touches:
  - src/autoclean/cleaner.py
---

## Decided
When a CSV has rows with more fields than its header, load_csv falls back to the python engine with an on_bad_lines callable that trims each row to the header's width, and issues one ParserWarning itself. From pandas 3.1, on_bad_lines="warn" skips the whole row (pandas 3.0 trimmed it), so rows silently disappeared; the callable keeps them on every version.
pandas 3.0 and 2.x never call the callable with index_col=False (they trim the row and warn themselves), so the fallback records pandas' ParserWarning and replaces it with ours instead of relying on the callable being called.
Constant fills check the value against the dtype first and switch the column to object when it doesn't fit. pandas 3.1 only warns (Pandas4Warning) when fillna gets an incompatible value such as a string for a timedelta or an int for a "str" column, and a later version will raise; the old code relied on the TypeError.

## Tried and ruled out
Passing low_memory=False unconditionally: pandas rejects it with engine="python", so every load_csv(..., engine="python") failed. It is only set for the C engine now.

## Watch out
The local environment (pandas 3.0.6 + pyarrow) hides all of this. Run the suite on pandas 2.x and 3.1+ without pyarrow too; CI's matrix in .github/workflows/tests.yml does.

In the code: connectors._read_ragged, connectors.load_csv (low_memory only for engine "c"); cleaner._fits_dtype, _fill_series_constant.
