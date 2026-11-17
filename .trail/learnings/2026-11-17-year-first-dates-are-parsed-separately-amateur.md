---
title: "Year-first dates are parsed separately so dayfirst can't flip them"
author: Amateur
date: 2026-11-17
branch: main
cost: { minutes: 9, tokens: 120114 }
id: l-b505a2d8
scope: repo
status: active
src: agent
touches:
  - src/autoclean/inference.py
  - tests/csv_fixtures.py
---

## Decided
- With dayfirst=True, dateutil reads ISO "2021-03-05" as 3 May. Values starting with a 4-digit year are parsed with dayfirst=False; the rest use the caller's setting. Without this, European files that mix dd.mm.yyyy with ISO dates, and any file we save and reload, get most dates wrong.
- Integer columns that pandas loaded as float64 (because of blanks) are narrowed back to Int64. Without that, save → reload turns every integer column into floats.

## Watch out
- In pandas 3, Series.map that returns strings infers the "str" dtype. If you build text yourself, give it that same dtype, or a second auto_clean pass will change the column again.
- Compressed fixture bytes (.gz/.zip) differ by zlib version: compare decompressed contents, not raw bytes.

In the code:
- src/autoclean/inference.py: _parse_dates (year_first mask), _narrow_typed_column
- tests/csv_fixtures.py: CsvFixture.payload; regenerate with `python -m tests.csv_fixtures`
