#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/run/media/amateur/MATRIX/Pandalyzer/AutoClean"
PYTHON="${PYTHON:-python}"

if [[ ! -d "$PROJECT/src/autoclean" || ! -d "$PROJECT/tests" ]]; then
  echo "ERROR: Pandalyzer project not found at: $PROJECT" >&2
  echo "Edit PROJECT near the top of this script if the project is elsewhere." >&2
  exit 1
fi
cd "$PROJECT"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="../AutoClean_backup_${STAMP}"
mkdir -p "$BACKUP"
cp -a src tests pyproject.toml README.md requirements.txt "$BACKUP/" 2>/dev/null || true
echo "Backup created at: $BACKUP"

"$PYTHON" - <<'PY'
from pathlib import Path
root = Path.cwd()
src = root / "src" / "autoclean"
tests = root / "tests"

def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"SKIP (already present): {path.relative_to(root)}")
        return
    with path.open("a", encoding="utf-8") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write("\n" + block.strip() + "\n")
    print(f"UPDATED: {path.relative_to(root)}")

append_once(src / "validators.py", "# --- Pandalyzer extension: schema and range validation ---", r'''
# --- Pandalyzer extension: schema and range validation ---
def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Return required column names that are absent from the DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(required_columns, list):
        raise TypeError("required_columns must be a list of column names.")
    return [column for column in required_columns if column not in df.columns]


def validate_column_types(df: pd.DataFrame, expected_types: dict[str, object]) -> dict[str, str]:
    """Return missing columns or columns whose dtype differs from the expected dtype."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(expected_types, dict):
        raise TypeError("expected_types must be a dictionary.")
    problems = {}
    for column, expected in expected_types.items():
        if column not in df.columns:
            problems[column] = "missing column"
            continue
        try:
            expected_dtype = pd.api.types.pandas_dtype(expected)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid expected dtype for column {column!r}: {expected!r}") from exc
        if not pd.api.types.is_dtype_equal(df[column].dtype, expected_dtype):
            problems[column] = f"expected {expected_dtype}, found {df[column].dtype}"
    return problems


def validate_numeric_ranges(df: pd.DataFrame, ranges: dict[str, tuple[float | None, float | None]]) -> dict[str, list]:
    """Return index labels of non-missing numeric values outside inclusive bounds."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(ranges, dict):
        raise TypeError("ranges must be a dictionary.")
    problems = {}
    for column, bounds in ranges.items():
        if column not in df.columns:
            raise KeyError(f"Column not found: {column}")
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError("Each range must be a (minimum, maximum) tuple.")
        minimum, maximum = bounds
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"Minimum exceeds maximum for column {column!r}.")
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column {column!r} must have a numeric dtype.")
        values = df[column]
        present = values.notna()
        mask = pd.Series(False, index=df.index)
        if minimum is not None:
            mask |= present & (values < minimum)
        if maximum is not None:
            mask |= present & (values > maximum)
        if mask.any():
            problems[column] = df.index[mask].tolist()
    return problems
''')

append_once(src / "profiler.py", "# --- Pandalyzer extension: quality comparison ---", r'''
# --- Pandalyzer extension: quality comparison ---
def compare_data_quality(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Compare basic data-quality metrics before and after cleaning."""
    if not isinstance(before, pd.DataFrame) or not isinstance(after, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")
    def metrics(df: pd.DataFrame) -> dict:
        missing = int(df.isna().sum().sum())
        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "missing_cell_count": missing,
            "missing_percent": round(missing / df.size * 100, 2) if df.size else 0.0,
            "duplicate_row_count": int(df.duplicated().sum()),
        }
    old, new = metrics(before), metrics(after)
    return {"before": old, "after": new, "change": {key: new[key] - old[key] for key in old}}
''')

append_once(src / "connectors.py", "# --- Pandalyzer extension: exports ---", r'''
# --- Pandalyzer extension: exports ---
import json

def save_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Save a DataFrame to CSV, creating parent directories, and return the path."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=index)
    return destination


def save_quality_report(report: dict, path: str | Path) -> Path:
    """Save a quality report as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def safe(value):
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="index")
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if value is pd.NA:
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        return value
    with destination.open("w", encoding="utf-8") as file:
        json.dump(safe(report), file, indent=2, ensure_ascii=False, default=str)
    return destination
''')

append_once(src / "cleaner.py", "# --- Pandalyzer extension: end-to-end cleaning helper ---", r'''
# --- Pandalyzer extension: end-to-end cleaning helper ---
def clean_dataframe(df: pd.DataFrame, *, remove_duplicates: bool = True, missing_strategy: str = "drop_rows", fill_value=None) -> pd.DataFrame:
    """Apply the existing duplicate and missing-value cleaners in sequence."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    result = df.copy()
    if remove_duplicates:
        result = remove_duplicate_rows(result)
    return handle_missing_values(result, strategy=missing_strategy, value=fill_value)
''')

append_once(tests / "test_validators.py", "# --- Pandalyzer extension tests: schema and range validation ---", r'''
# --- Pandalyzer extension tests: schema and range validation ---
import pandas as pd
import pytest
from autoclean.validators import validate_required_columns, validate_column_types, validate_numeric_ranges

def test_validate_required_columns_returns_missing_names():
    assert validate_required_columns(pd.DataFrame({"id": [1]}), ["id", "status"]) == ["status"]

def test_validate_required_columns_rejects_non_dataframe():
    with pytest.raises(TypeError): validate_required_columns([], ["id"])

def test_validate_column_types_reports_mismatch_and_missing_column():
    df = pd.DataFrame({"amount": [2.5]})
    result = validate_column_types(df, {"amount": "object", "status": "string"})
    assert result["status"] == "missing column"
    assert result["amount"] == "expected object, found float64"

def test_validate_column_types_accepts_matching_dtype():
    assert validate_column_types(pd.DataFrame({"x": pd.Series([1], dtype="int64")}), {"x": "int64"}) == {}

def test_validate_numeric_ranges_returns_out_of_range_index_labels():
    df = pd.DataFrame({"amount": [-1, 10, 120]}, index=[10, 20, 30])
    assert validate_numeric_ranges(df, {"amount": (0, 100)}) == {"amount": [10, 30]}

def test_validate_numeric_ranges_ignores_missing_values():
    assert validate_numeric_ranges(pd.DataFrame({"amount": [None, 10.0]}), {"amount": (0, 100)}) == {}

def test_validate_numeric_ranges_rejects_non_numeric_column():
    with pytest.raises(TypeError): validate_numeric_ranges(pd.DataFrame({"x": ["bad"]}), {"x": (0, 10)})

def test_validate_numeric_ranges_rejects_invalid_bounds():
    with pytest.raises(ValueError): validate_numeric_ranges(pd.DataFrame({"x": [1]}), {"x": (10, 0)})
''')

append_once(tests / "test_profiler.py", "# --- Pandalyzer extension tests: quality comparison ---", r'''
# --- Pandalyzer extension tests: quality comparison ---
import pandas as pd
import pytest
from autoclean.profiler import compare_data_quality

def test_compare_data_quality_reports_changes():
    result = compare_data_quality(pd.DataFrame({"x": [1, None, 1]}), pd.DataFrame({"x": [1, 2]}))
    assert result["before"]["row_count"] == 3
    assert result["after"]["row_count"] == 2
    assert result["change"]["missing_cell_count"] == -1

def test_compare_data_quality_handles_empty_frames():
    assert compare_data_quality(pd.DataFrame(), pd.DataFrame())["before"]["missing_percent"] == 0.0

def test_compare_data_quality_rejects_non_dataframes():
    with pytest.raises(TypeError): compare_data_quality([], pd.DataFrame())
''')

append_once(tests / "test_connectors.py", "# --- Pandalyzer extension tests: exports ---", r'''
# --- Pandalyzer extension tests: exports ---
import json
import pandas as pd
import pytest
from autoclean.connectors import save_csv, save_quality_report

def test_save_csv_writes_csv_and_returns_path(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    path = save_csv(df, tmp_path / "nested" / "clean.csv")
    assert path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(path), df)

def test_save_csv_rejects_non_dataframe(tmp_path):
    with pytest.raises(TypeError): save_csv([1], tmp_path / "bad.csv")

def test_save_quality_report_writes_json(tmp_path):
    path = save_quality_report({"rows": 3, "issues": ["missing amount"]}, tmp_path / "report.json")
    assert json.loads(path.read_text(encoding="utf-8"))["rows"] == 3

def test_save_quality_report_converts_dataframe(tmp_path):
    path = save_quality_report({"columns": pd.DataFrame({"missing_count": [2]}, index=["amount"])}, tmp_path / "report.json")
    assert json.loads(path.read_text(encoding="utf-8"))["columns"]["amount"]["missing_count"] == 2
''')

append_once(tests / "test_cleaner.py", "# --- Pandalyzer extension tests: end-to-end cleaning helper ---", r'''
# --- Pandalyzer extension tests: end-to-end cleaning helper ---
import pandas as pd
import pytest
from autoclean.cleaner import clean_dataframe

def test_clean_dataframe_removes_duplicates_then_drops_missing_rows():
    df = pd.DataFrame({"id": [1, 1, 2], "amount": [10.0, 10.0, None]})
    result = clean_dataframe(df)
    assert result.to_dict(orient="records") == [{"id": 1, "amount": 10.0}]
    assert len(df) == 3

def test_clean_dataframe_can_fill_constant_without_deduplication():
    df = pd.DataFrame({"id": [1, 1], "status": ["ok", None]})
    result = clean_dataframe(df, remove_duplicates=False, missing_strategy="fill_constant", fill_value="unknown")
    assert result["status"].tolist() == ["ok", "unknown"]

def test_clean_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError): clean_dataframe([])
''')

print("Source and test updates complete.")
PY

echo
echo "Running syntax compilation..."
"$PYTHON" -m compileall -q src/autoclean tests
echo
echo "Running tests with coverage..."
"$PYTHON" -m pytest tests/ --cov=autoclean --cov-report=term-missing
echo
echo "Done. Existing architecture preserved; only existing modules and tests were extended."
echo "If a test fails, restore from backup: $BACKUP"
