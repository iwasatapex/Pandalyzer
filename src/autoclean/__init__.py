"""autoclean: load, profile, validate and clean tabular data with pandas."""

from autoclean.cleaner import (
    DEFAULT_MISSING_MARKERS,
    auto_clean,
    clean_column_names,
    clean_dataframe,
    drop_empty_rows,
    drop_repeated_headers,
    drop_total_rows,
    fill_missing_auto,
    handle_missing_values,
    normalize_emails,
    remove_duplicate_rows,
    standardize_missing_values,
    strip_whitespace,
    unify_case_variants,
    unwrap_excel_text,
)
from autoclean.connectors import count_preamble_lines, load_csv, save_csv, save_quality_report
from autoclean.emails import email_columns, is_valid_email, normalize_email
from autoclean.inference import infer_column_types, infer_series_type
from autoclean.profiler import compare_data_quality, profile_dataframe
from autoclean.validators import (
    validate_column_types,
    validate_emails,
    validate_missing_values,
    validate_numeric_ranges,
    validate_required_columns,
)

__all__ = [
    "DEFAULT_MISSING_MARKERS",
    "auto_clean",
    "clean_column_names",
    "clean_dataframe",
    "compare_data_quality",
    "count_preamble_lines",
    "drop_empty_rows",
    "drop_repeated_headers",
    "drop_total_rows",
    "email_columns",
    "fill_missing_auto",
    "handle_missing_values",
    "infer_column_types",
    "infer_series_type",
    "is_valid_email",
    "load_csv",
    "normalize_email",
    "normalize_emails",
    "profile_dataframe",
    "remove_duplicate_rows",
    "save_csv",
    "save_quality_report",
    "standardize_missing_values",
    "strip_whitespace",
    "unify_case_variants",
    "unwrap_excel_text",
    "validate_column_types",
    "validate_emails",
    "validate_missing_values",
    "validate_numeric_ranges",
    "validate_required_columns",
]
