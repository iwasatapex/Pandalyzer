# Pandalyzer

A data profiling and cleaning toolkit built on pandas. It loads and cleans CSV files with any schema; you don't need to tell it the column names or types. The importable package is `autoclean`.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from autoclean import auto_clean, compare_data_quality, load_csv, profile_dataframe, save_quality_report

raw = load_csv("database/messy_retail_orders.csv")    # delimiter, encoding and compression detected
clean, log = auto_clean(raw, return_log=True)          # schema-agnostic cleaning

save_quality_report(
    {"profile": profile_dataframe(clean), "comparison": compare_data_quality(raw, clean), "log": log},
    "out/report.json",
)
```

## Loading: `load_csv`

| Handles | How |
| --- | --- |
| Delimiters | `,` `;` tab `\|`, detected from the first lines (quotes respected) |
| Encodings | UTF-8 (± BOM), UTF-16, Windows-1252, Latin-1, with a fallback if a bad byte appears late in the file |
| Compression | `.gz`, `.bz2`, `.xz`, single-file `.zip` |
| Malformed rows | Rows with extra fields are kept (the extras dropped, with a `ParserWarning`); columns never shift |

Pass `sep=`, `encoding=` or any `pandas.read_csv` option to override detection. Values load unchanged; cleaning is a separate step.

## Cleaning: `auto_clean`

Runs these steps in order. Each one is a keyword you can switch off.

1. **Column names**: trim (`"strip"`, the default) or `"snake"` case, and make them unique.
2. **Whitespace**: trim spaces, tabs and non-breaking spaces around text values.
3. **Placeholders → missing**: `"N/A"`, `"null"`, `"-"`, `"unknown"`, `"TBD"`, `"#N/A"`, `""`… (case-insensitive; override with `missing_markers=`).
4. **Case variants**: `"Cash"`, `" CASH "` and `"cash"` become the most common spelling.
5. **Type inference** for text columns:
   - integers (`Int64`) and floats, including `"₹1,234.50"`, `"15%"` and `"(42)"`, plus European decimal commas (`"3.450,75"`)
   - booleans: `Yes/No`, `Y/N`, `True/False`
   - datetimes in mixed formats (`dayfirst=True` for DD/MM; year-first ISO dates are never flipped)

   Leading-zero codes (`"00123"`) and very long digit strings stay text. Values that don't parse become missing, and the log counts them.

   Typed columns are only narrowed: an integer column that pandas loaded as `float64` because it had blanks becomes `Int64`, and integers too big for int64 (long IDs) become text.
6. **Empty rows** and empty `Unnamed: N` columns (from trailing delimiters) are dropped.
7. **Duplicate rows** are removed. This runs after the fixes above, so near-duplicates are caught too.
8. **Missing values**:
   - `missing="keep"` (default) leaves them in place.
   - `missing="auto"` fills numbers with the median and low-cardinality text, category and boolean columns with the most common value. Free text, IDs and dates are left missing.
   - Any of `"drop_rows"`, `"fill_mean"`, `"fill_median"`, or `"fill_constant"` with `fill_value=`.

Columns that already have a real dtype (int, float, bool, datetime, category, nullable types) are never converted. Every function accepts every pandas dtype, including `str`, `string`, `object`, `Int64`, `Float64`, `boolean`, tz-aware datetimes, timedeltas, categories, and cells that hold lists.

Each step is also available on its own: `clean_column_names`, `strip_whitespace`, `standardize_missing_values`, `unify_case_variants`, `infer_column_types`, `drop_empty_rows`, `remove_duplicate_rows`, `handle_missing_values`, `fill_missing_auto`.

## What it does not do

- **Synonyms and typos**: `"COD"` vs `"Cash on Delivery"`, or `"Pendng"`. Map these yourself with `df.replace({...})`.
- **Number words**: `"five"` and `"fifty"` are not converted to numbers.
- **Out-of-range values**: values like `-3` or `999` are kept. Use `validate_numeric_ranges` to flag them.
- **Header-less files**: they need `header=None`.

## Other modules

| Module | Purpose |
| --- | --- |
| `profiler` | Per-column profile, plus before/after quality comparison. |
| `validators` | Missing-value thresholds, required columns, dtypes, and numeric ranges. |
| `connectors` | `save_csv`, plus `save_quality_report`, which writes strict JSON (NaN, NaT and inf become `null`). |

## Tests

```bash
pytest
```

`tests/data/` holds 10 generated CSV files (100 rows each), each with a different schema and format: semicolon/Windows-1252 with decimal commas, tab-separated, pipe-separated UTF-16, Latin-1 with multi-line quoted fields, gzip, zip, ragged rows, a BOM with duplicate headers, and already-clean data. [tests/test_csv_fixtures.py](tests/test_csv_fixtures.py) runs the whole project on each file: load, profile, clean, validate, fill, save, reload and report. It checks the results against the ground truth that [tests/csv_fixtures.py](tests/csv_fixtures.py) records when it generates the files. After editing that generator, regenerate the files with `python -m tests.csv_fixtures`.
