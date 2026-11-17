---
title: "Date columns are detected by how values look, not by how many parse"
author: Amateur
date: 2026-11-17
branch: main
cost: { minutes: 1, tokens: 1575 }
id: l-82c03e0a
scope: repo
status: active
src: agent
touches:
  - src/autoclean/cleaner.py
---

## Decided
- Type inference converts a text column to datetime when at least 95% of values *look* like dates (by pattern). Values that look like dates but are impossible (2024-02-30, 0000-00-00) become missing. Requiring 95% to *parse* meant one impossible date in a small column blocked the whole conversion.
- Numbers and booleans still decide on parse success, because there "looks like" and "parses" are the same thing.
- auto_clean leaves missing values in place by default. Imputing invents data (e.g. filling coupon_code would hand out coupons nobody used); missing="auto" is opt-in.

## Tried and ruled out
- A loose date pattern (\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}) matched version numbers and IP addresses. It now needs a 4-digit year, a 2-digit year with / or -, or a month name.
- Series.replace(mapping) for merging case variants: it scans the column once per key and took about 12 s on 200k rows with ~1,500 variants. One dict-lookup pass with map + mask halved the total run time.

## Watch out
- Tiny test fixtures trip the heuristics: the 95% type threshold, and "low cardinality" (≤50 distinct values and ≤ half of non-missing values) for mode-filling. Use enough rows in tests.
- A row that is entirely empty is dropped by drop_empty before missing-value handling, so it never shows up in missing counts.
- A capture group in a regex passed to str.contains raises a UserWarning; use non-capturing alternatives.

In the code:
- src/autoclean/inference.py: _DATE_LIKE, _MIN_DATE_PARSE_RATE = 0.5, _as_datetime, threshold default 0.95
- src/autoclean/cleaner.py: unify_case_variants (map + mask), _is_low_cardinality, auto_clean(missing="keep")
