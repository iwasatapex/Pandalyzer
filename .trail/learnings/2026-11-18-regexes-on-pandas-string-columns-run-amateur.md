---
title: "Regexes on pandas string columns run on RE2, which lacks backreferences, lookarounds, \\u escapes and Unicode \\s"
author: Amateur
date: 2026-11-18
branch: audit-universal-cleaning
cost: { minutes: 16, tokens: 271001 }
id: l-aac0c3f9
scope: repo
status: active
src: agent
touches:
  - src/autoclean/_utils.py
---

## Decided
Every regex passed to a pandas .str method avoids backreferences, lookarounds and "\uXXXX" escapes, and spells out non-ASCII whitespace as literal characters. With pyarrow installed, pandas 3 stores text columns as Arrow strings and runs .str.contains/.fullmatch/.replace through RE2, which raises ArrowInvalid on those constructs instead of falling back to Python's re.
Thousands separators are matched as one alternative per separator, to keep "1,234 567" from parsing; a backreference would be simpler but RE2 rejects it.

## Tried and ruled out
A backreference "([,' ])\d{3}(?:\1\d{3})*" for consistent separators, "(?![A-Za-z])" after currency codes, and "\s" for whitespace. The first two fail outright on Arrow strings. "\s" fails silently: RE2's "\s" is ASCII-only, so non-breaking and narrow no-break spaces were not trimmed. The tests passed on object columns, which use Python's re.

## Watch out
Test new patterns on "str" and "string[pyarrow]" columns as well as object columns; tests/test_messy_data.py parametrizes over these dtypes.
Series.map that returns strings infers the "str" dtype in pandas 3, so text helpers cast back to the input dtype or object columns silently change type.

In the code: _utils.HORIZONTAL_SPACE / LINE_BREAKS / INVISIBLE (literal characters); inference._grouped(), _US_SEPARATORS, _EU_SEPARATORS; cleaner._EDGE_SPACE, _INNER_SPACE, _map_strings.
