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
| Report preambles | Title and note lines above the header (`"Account statement"`, `"Generated: Jan 5, 2024"`, blank lines) are skipped and kept in `df.attrs["preamble"]`. A header that is narrower than the data rows is kept. |
| Excel headers | A header cell with a line break inside it (`"Hours⏎Worked"`) stays one column |

Pass `sep=` (including regex separators such as `r"\s+"` with `engine="python"`), `encoding=`, `skiprows=`, `header=` or any `pandas.read_csv` option to override detection. Values load unchanged; cleaning is a separate step.

## Cleaning: `auto_clean`

Runs these steps in order. Each one is a keyword you can switch off.

1. **Column names**: trim and collapse inner spaces and line breaks (`"strip"`, the default) or `"snake"` case, and make them unique.
2. **Text tidying** (`strip_text`):
   - trim spaces, tabs, non-breaking spaces and zero-width characters around text values
   - collapse repeated spaces inside them (`"John   Smith"` → `"John Smith"`); line breaks are kept
   - unwrap Excel's `="00123"`
3. **Placeholders → missing**: `"N/A"`, `"null"`, `"-"`, `"—"`, `"unknown"`, `"TBD"`, `"(blank)"`, `"<NA>"`, `"#N/A"`, `""`… (case-insensitive; override with `missing_markers=`).
4. **Export leftovers** (`drop_artifacts`):
   - rows that repeat the header (from concatenated exports, even when re-cased). Frames loaded with `header=None` are left alone.
   - trailing summary rows labelled `Total`, `Subtotal`, `Grand total` or `Sum`. A longer label (`"Total debits"`) counts only when the row leaves empty a column that the other rows fill, so a last record such as a store called `"Total Gym"` is kept.
5. **Emails**: columns of email addresses are detected from their values.
   - Addresses are trimmed and lower-cased, and `mailto:` and display names (`"Ann <ann@x.com>"`) are removed.
   - International addresses (`josé@example.com`, `user@例え.jp`, punycode endings) are valid.
   - Malformed addresses (`"@example.com"`, `"bob@"`, `"a@@b.com"`, `"john smith@x.com"`) become missing by default, and the log counts them. They are never rewritten into a different address.
   - Phone numbers in the same column are kept.

   Pass `emails="keep"` to leave malformed addresses in place, or `emails=None` to skip this step.
6. **Case variants**: `"Cash"`, `" CASH "` and `"cash"` become the most common spelling.
7. **Type inference** for text columns:
   - integers (`Int64`) and floats, including `"₹1,234.50"`, `"USD 1,234.50"`, `"15 EUR"`, `"15%"` and `"(42)"`
   - thousands separators written with a comma, space, apostrophe (Swiss `"1'234.50"`) or narrow space
   - European decimal commas (`"3.450,75"`, French `"1 234,56 €"`)
   - booleans: `Yes/No`, `Y/N`, `True/False`, `on/off`
   - datetimes in mixed formats (`dayfirst=True` for DD/MM; year-first ISO dates are never flipped)

   Leading-zero codes (`"00123"`) and very long digit strings stay text. Values that don't parse become missing, and the log counts them.

   Typed columns are only narrowed:
   - an integer column that pandas loaded as `float64` because it had blanks becomes `Int64`
   - a `True/False` column that pandas loaded as `object` because it had blanks becomes `boolean`
   - integers too big for int64 (long IDs) become text
8. **Empty rows** and empty `Unnamed: N` columns (from trailing delimiters) are dropped.
9. **Duplicate rows** are removed. This runs after the fixes above, so near-duplicates are caught too.
10. **Missing values**:
   - `missing="keep"` (default) leaves them in place.
   - `missing="auto"` fills numbers with the median and low-cardinality text, category and boolean columns with the most common value. Free text, IDs and dates are left missing.
   - Any of `"drop_rows"`, `"fill_mean"`, `"fill_median"`, or `"fill_constant"` with `fill_value=`.

Columns that already have a real dtype (int, float, bool, datetime, category, nullable types) are never converted. Category columns of text still get steps 2, 3 and 6 (whitespace, placeholders, case variants) and stay categories, with their category order kept. Every function accepts every pandas dtype, including `str`, `string`, `object`, `Int64`, `Float64`, `boolean`, tz-aware datetimes, timedeltas, categories, and cells that hold lists.

Each step is also available on its own: `clean_column_names`, `strip_whitespace`, `unwrap_excel_text`, `standardize_missing_values`, `drop_repeated_headers`, `drop_total_rows`, `normalize_emails`, `unify_case_variants`, `infer_column_types`, `drop_empty_rows`, `remove_duplicate_rows`, `handle_missing_values`, `fill_missing_auto`.

## What it does not do

- **Synonyms and typos**: `"COD"` vs `"Cash on Delivery"`, or `"Pendng"`. Map these yourself with `df.replace({...})`.
- **Number words**: `"five"` and `"fifty"` are not converted to numbers.
- **Out-of-range values**: values like `-3` or `999` are kept. Use `validate_numeric_ranges` to flag them.
- **Header-less files**: they need `header=None`.
- **Preambles without a title or blank line**: a lone note line such as `"Prepared by Ann, Finance"` directly above the header, with no numbers in it, looks like a header. Pass `skiprows=`.
- **Short dotted dates**: `"05.03.24"` stays text, because it looks like a version number. Four-digit years (`"05.03.2024"`) are converted.
- **Phone numbers**: they are kept as text (also inside email columns), but their formats are not unified.

## Other modules

| Module | Purpose |
| --- | --- |
| `profiler` | Per-column profile, plus before/after quality comparison. |
| `validators` | Missing-value thresholds, required columns, dtypes, numeric ranges, and `validate_emails`, which lists the rows with malformed addresses for review. All of them handle repeated column names. |
| `connectors` | `save_csv`, plus `save_quality_report`, which writes strict JSON (NaN, NaT and inf become `null`). |

## Project layout

| Path | Contents |
| --- | --- |
| `src/autoclean/` | The package: `connectors`, `cleaner`, `inference`, `profiler`, `validators` |
| `tests/` | The pytest suite, plus the fixture generator and its files in `tests/data/` |
| `scripts/` | `stress_test.py` writes a 100,000-row messy CSV whose right answers it knows, then scores `auto_clean` on it: `python scripts/stress_test.py`. The CSV goes to `database/`, and git ignores it. |
| `database/` | `messy_retail_orders.csv`, a 10,000-row messy sample dataset |
| `notebooks/` | Walkthroughs of each module on the sample dataset. They write to `manual_test_outputs/`, which git ignores. |
| `.trail/learnings/` | Notes from past work: what was decided, what was ruled out, and why |

## Tests and linting

GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs ruff and the tests on every push. It covers Python 3.10–3.14 on pandas 2.2, 2.3 and 3.x, with and without pyarrow. Text columns behave differently with pyarrow: Arrow strings run regexes on RE2.

```bash
pytest
ruff check src tests
ruff format src tests
```

`tests/data/` holds 15 generated CSV files (100 rows each), each with its own schema, format and messy data:

- semicolon/Windows-1252 with decimal commas
- tab-separated
- pipe-separated UTF-16
- Latin-1 with multi-line quoted fields
- gzip and zip archives
- ragged rows
- a BOM with duplicate headers
- already-clean data
- CRM contacts with messy emails
- a bank statement with a title above the header and `Total` rows
- concatenated exports with repeated headers
- Swiss and French price formats
- an Excel export with `="00123"` codes and CRLF line endings

[tests/test_csv_fixtures.py](tests/test_csv_fixtures.py) runs the whole project on each file: load, profile, clean, validate, fill, save, reload and report. [tests/test_messy_data.py](tests/test_messy_data.py) unit-tests each kind of messy data on its own. It checks the results against the ground truth that [tests/csv_fixtures.py](tests/csv_fixtures.py) records when it generates the files. After editing that generator, regenerate the files with `python -m tests.csv_fixtures`.

## License

MIT. See [LICENSE](LICENSE).
