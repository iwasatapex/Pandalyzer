import bz2
import codecs
import csv
import gzip
import json
import lzma
import math
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_DELIMITERS = (",", ";", "\t", "|")
_SAMPLE_BYTES = 256 * 1024
_SNIFF_LINES = 50
_COMPRESSED_OPENERS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}


def _read_sample(path: Path, size: int = _SAMPLE_BYTES) -> bytes:
    """Read the first bytes of a (possibly compressed) file."""
    suffix = path.suffix.lower()
    if suffix in _COMPRESSED_OPENERS:
        with _COMPRESSED_OPENERS[suffix](path, "rb") as file:
            return file.read(size)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"Zip archive must contain exactly one file: {path}")
            with archive.open(members[0]) as file:
                return file.read(size)
    with path.open("rb") as file:
        return file.read(size)


def detect_encoding(sample: bytes) -> str:
    """Guess a text encoding for ``sample``: BOM-marked UTF, UTF-8, then Windows-1252/Latin-1."""
    if sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            # final=False tolerates a multi-byte character cut off at the end of the sample.
            codecs.getincrementaldecoder(encoding)().decode(sample, final=False)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"  # Decodes any byte sequence.


def detect_delimiter(text: str, candidates: tuple[str, ...] = CANDIDATE_DELIMITERS) -> str:
    """Pick the delimiter that splits the first lines into the most consistent field counts.

    Quotes are respected, so commas inside quoted values don't count. Falls back
    to a comma when nothing splits the lines (e.g. a single-column file).
    """
    lines = text.splitlines()
    if len(lines) > 1 and not text.endswith(("\n", "\r")):
        lines = lines[:-1]  # Drop a line that may be truncated by the sample cut-off.
    lines = [line for line in lines[:_SNIFF_LINES] if line.strip()]
    best, best_score = ",", (0.0, 1)
    for delimiter in candidates:
        try:
            counts = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        except csv.Error:
            continue
        if not counts:
            continue
        typical = Counter(counts).most_common(1)[0][0]
        if typical < 2:
            continue
        consistency = sum(count == typical for count in counts) / len(counts)
        score = (consistency, typical)
        if score > best_score:
            best, best_score = delimiter, score
    return best


def load_csv(
    path: str | Path,
    *,
    sep: str | None = None,
    encoding: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a delimited text file into a DataFrame, detecting its format.

    - Delimiter: comma, semicolon, tab or pipe, detected unless ``sep`` is given.
    - Encoding: UTF-8 (with or without BOM), UTF-16, Windows-1252 or Latin-1,
      detected unless ``encoding`` is given.
    - Compression: ``.gz``, ``.bz2``, ``.xz`` and single-file ``.zip``.
    - Malformed rows: rows with extra fields are kept and the extra fields
      dropped, with a ``ParserWarning``, instead of failing the whole load or
      shifting columns.

    Values are loaded as-is; use :func:`autoclean.cleaner.auto_clean` to clean
    them. Extra keyword arguments are passed through to :func:`pandas.read_csv`.
    """
    path = Path(path)
    sample = _read_sample(path)
    if not sample.strip():
        raise pd.errors.EmptyDataError(f"CSV file is empty: {path}")

    if encoding is None:
        encoding = detect_encoding(sample)
    if sep is None and "delimiter" not in read_csv_kwargs:
        text = codecs.getincrementaldecoder(encoding)(errors="replace").decode(sample)
        sep = detect_delimiter(text)

    options = {"sep": sep, "encoding": encoding, "index_col": False, "low_memory": False}
    options.update(read_csv_kwargs)
    try:
        return pd.read_csv(path, **options)
    except UnicodeDecodeError:
        # The sample decoded cleanly but a later part of the file did not.
        options["encoding"] = "latin-1" if encoding == "cp1252" else "cp1252"
        try:
            return pd.read_csv(path, **options)
        except UnicodeDecodeError:
            options["encoding"] = "latin-1"
            return pd.read_csv(path, **options)
    except pd.errors.ParserError:
        if "on_bad_lines" in read_csv_kwargs or "engine" in read_csv_kwargs:
            raise
        options.pop("low_memory")
        options.update(engine="python", on_bad_lines="warn")
        return pd.read_csv(path, **options)


def save_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Save a DataFrame to CSV, creating parent directories, and return the path."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=index)
    return destination


def _to_json_safe(value):
    """Recursively convert pandas/numpy values into strict-JSON-compatible objects.

    Missing values (``None``, ``NaN``, ``pd.NA``, ``pd.NaT``) and infinities
    become ``None`` so the output is valid JSON for any parser.
    """
    if isinstance(value, pd.DataFrame):
        return _to_json_safe(value.to_dict(orient="index"))
    if isinstance(value, pd.Series):
        return _to_json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    elif isinstance(value, np.timedelta64):
        value = pd.Timedelta(value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_quality_report(report: dict, path: str | Path) -> Path:
    """Save a quality report as strict JSON and return the path."""
    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(_to_json_safe(report), file, indent=2, ensure_ascii=False, allow_nan=False)
    return destination
