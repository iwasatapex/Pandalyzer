import bz2
import codecs
import csv
import gzip
import io
import json
import lzma
import math
import re
import warnings
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_DELIMITERS = (",", ";", "\t", "|")
_MAX_PREAMBLE = 10  # Lines a report title block may span.
_BARE_NUMBER = re.compile(r"[\d.,:/+-]*\d[\d.,:/+-]*")  # "2024", "1.5", "05/01/2024"
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


def _records(text: str, delimiter: str) -> list[tuple[int, list[str]]]:
    """The first CSV records of ``text`` with the physical line each starts on.

    Parsing records (not lines) keeps a quoted header cell that contains a
    line break, such as Excel's ``"Hours\\nWorked"``, in one piece.
    """
    if not text.endswith(("\n", "\r")):
        text = text[: max(text.rfind("\n"), text.rfind("\r")) + 1]  # Drop a cut-off line.
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    records, start = [], 0
    try:
        for fields in reader:
            records.append((start, fields))
            start = reader.line_num
            if len(records) >= _SNIFF_LINES:
                break
    except csv.Error:
        pass
    return records


def _width(fields: list[str]) -> int:
    """Number of fields, not counting trailing empty ones (a stray final delimiter)."""
    width = len(fields)
    while width and not fields[width - 1].strip():
        width -= 1
    return width


def _looks_like_note(fields: list[str], typical: int) -> bool:
    """A title/notes line: much narrower than a row, or holding a bare number or date.

    Header names are labels; a field such as ``" 2024"`` (from ``"Generated: Jan
    5, 2024"``) marks a note even when the line happens to split into several fields.
    """
    if _width(fields) * 2 <= typical:
        return True
    return any(_BARE_NUMBER.fullmatch(field.strip()) for field in fields if field.strip())


def count_preamble_lines(text: str, delimiter: str) -> int:
    """Count the title/notes lines above the real header of a CSV export.

    Report exports often start with lines such as ``"Account statement"`` or
    ``"Generated: Jan 5, 2024"`` and a blank line. Two rules apply:

    1. A block of notes that starts with a title (a much narrower line) and
       ends in a blank line is preamble, even when a note such as
       ``"Generated: Jan 5, 2024"`` splits into as many fields as a row.
    2. After that, only lines above the first full-width record are candidates
       (trailing empty fields aside, so a stray final delimiter doesn't make a
       row wider). Everything up to a blank line among them is preamble, and so
       is any later line that looks like a note rather than a header. A
       narrower header above wider data rows is kept.

    Returns 0 when the first line already looks like the header.
    """
    records = _records(text, delimiter)
    widths = Counter(_width(fields) for _, fields in records if _width(fields))
    if not widths:
        return 0
    # Most common width; a tie goes to the wider one (2 notes + 2 rows of a tiny file).
    typical = max(widths, key=lambda width: (widths[width], width))
    if typical < 2:
        return 0
    header = _titled_block_end(records, typical)
    header += _notes_above_header(records[header:], typical)
    return records[header][0] if header < len(records) else 0


def _titled_block_end(records: list, typical: int) -> int:
    """Index just past a "title, notes…, blank line" block at the top, else 0."""
    end, titled = 0, False
    for index, (_, fields) in enumerate(records[:_MAX_PREAMBLE]):
        if not _width(fields):
            if titled:
                end = index + 1
            continue
        if _width(fields) * 2 <= typical:
            titled = True
        elif not _looks_like_note(fields, typical):
            break
    return end


def _notes_above_header(records: list, typical: int) -> int:
    """Rule 2 of :func:`count_preamble_lines`: the index of the header record."""
    first_full = next(
        (
            index
            for index, (_, fields) in enumerate(records)
            if _width(fields) and max(len(fields), _width(fields)) >= typical
        ),
        None,
    )
    if not first_full:  # None (no full-width record) or 0 (it is the first record).
        return 0
    leading = records[:first_full]
    blanks = [index for index, (_, fields) in enumerate(leading) if not _width(fields)]
    position = blanks[-1] + 1 if blanks else 0
    while position < len(leading) and _looks_like_note(leading[position][1], typical):
        position += 1
    return position


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
    - Preamble: title or note lines above the header (common in report
      exports) are skipped, unless ``skiprows`` or ``header`` is given. The
      skipped lines are kept in ``df.attrs["preamble"]``.

    Values are loaded as-is; use :func:`autoclean.cleaner.auto_clean` to clean
    them. Extra keyword arguments are passed through to :func:`pandas.read_csv`.
    """
    path = Path(path)
    sample = _read_sample(path)
    if not sample.strip():
        raise pd.errors.EmptyDataError(f"CSV file is empty: {path}")

    if encoding is None:
        encoding = detect_encoding(sample)
    text = codecs.getincrementaldecoder(encoding)(errors="replace").decode(sample)
    if sep is None and "delimiter" not in read_csv_kwargs:
        sep = detect_delimiter(text)

    options = {"sep": sep, "encoding": encoding, "index_col": False}
    if read_csv_kwargs.get("engine", "c") == "c":
        options["low_memory"] = False  # A C-engine option; pandas rejects it for "python".
    preamble = []
    delimiter = read_csv_kwargs.get("delimiter", sep)
    # csv can only sniff single-character delimiters; regex or multi-character
    # separators (sep=r"\s+", "::") are loaded without preamble detection.
    if (
        isinstance(delimiter, str)
        and len(delimiter) == 1
        and not {"skiprows", "header"} & read_csv_kwargs.keys()
    ):
        skip = count_preamble_lines(text, delimiter)
        if skip:
            preamble = re.split(r"\r\n|\r|\n", text)[:skip]  # Physical lines, as csv counts.
            options["skiprows"] = skip
    options.update(read_csv_kwargs)
    df = _read_csv(path, options, read_csv_kwargs, encoding)
    if preamble:
        df.attrs["preamble"] = preamble
    return df


def _read_csv(path: Path, options: dict, read_csv_kwargs: dict, encoding: str) -> pd.DataFrame:
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
        return _read_ragged(path, options)


def _read_ragged(path: Path, options: dict) -> pd.DataFrame:
    """Read a file whose rows have more fields than its header, keeping every row.

    The extra fields are dropped with one ``ParserWarning``. A callable is used
    because ``on_bad_lines="warn"`` skips the whole row from pandas 3.1 on.
    """
    width = len(pd.read_csv(path, **{**options, "nrows": 0}).columns)
    widened = []

    def keep_row(fields: list[str]) -> list[str]:
        widened.append(len(fields))
        return fields[:width]

    options = {key: value for key, value in options.items() if key != "low_memory"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = pd.read_csv(path, **options, engine="python", on_bad_lines=keep_row)
    # pandas < 3.1 trims the rows itself (with its own warning) instead of calling keep_row.
    trimmed_by_pandas = False
    for warning in caught:
        if issubclass(warning.category, pd.errors.ParserWarning):
            trimmed_by_pandas = True
        else:
            warnings.warn(warning.message, warning.category, stacklevel=4)
    if widened or trimmed_by_pandas:
        count = f"{len(widened)} rows" if widened else "Some rows"
        warnings.warn(
            f"{count} had more fields than the header ({width}); their extra fields were dropped.",
            pd.errors.ParserWarning,
            stacklevel=4,
        )
    return df


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
