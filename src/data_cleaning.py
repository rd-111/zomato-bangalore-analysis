"""Inspect and clean the Zomato Bangalore CSV without changing the source file.

Cleaning is deliberately conservative: rows are removed only when every source
field is an exact duplicate of an earlier row. Missing ratings remain missing,
and unusual but parseable values are flagged for review rather than silently
deleted.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DATA_DIR, RAW_DATA_PATH, require_dataset

RATING_COLUMN = "rate"
COST_COLUMN = "approx_cost(for two people)"
NUMERIC_COLUMNS = ("votes", COST_COLUMN)
BOOLEAN_COLUMNS = ("online_order", "book_table")
IMPORTANT_CATEGORICAL_COLUMNS = ("online_order", "book_table", "location", "cuisines", "rest_type")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the source CSV without changing its values or writing over it."""
    dataset_path = require_dataset(Path(path))
    try:
        return pd.read_csv(dataset_path, low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse CSV at {dataset_path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Could not read CSV at {dataset_path}: {exc}") from exc


def inspect_data(data: pd.DataFrame) -> dict[str, Any]:
    """Create the requested inspection outputs from an in-memory dataframe."""
    categorical = {}
    for column in IMPORTANT_CATEGORICAL_COLUMNS:
        if column in data.columns:
            categorical[column] = data[column].value_counts(dropna=False)
    numeric = data.select_dtypes(include=[np.number]).describe().transpose()
    return {
        "dimensions": data.shape,
        "columns": list(data.columns),
        "data_types": data.dtypes.astype(str),
        "first_five_rows": data.head(),
        "missing_values": data.isnull().sum().sort_values(ascending=False),
        "duplicate_rows": int(data.duplicated().sum()),
        "categorical_values": categorical,
        "numerical_summary": numeric,
    }


def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase, underscore-separated column names."""
    cleaned = data.copy()
    cleaned.columns = (
        cleaned.columns.astype(str).str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return cleaned


def parse_rating(values: pd.Series) -> pd.Series:
    """Parse a Zomato rating as float; NEW, '-', blanks and invalids become NaN."""
    text = values.astype("string").str.strip().str.upper()
    text = text.replace({"NEW": pd.NA, "-": pd.NA, "": pd.NA, "NAN": pd.NA})
    # Require a valid rating token; the optional '/5' suffix is discarded.
    extracted = text.str.extract(r"^([0-5](?:\.\d+)?)\s*(?:/\s*5)?\s*$", expand=False)
    rating = pd.to_numeric(extracted, errors="coerce")
    return rating.where(rating.between(0, 5))


def parse_numeric(values: pd.Series, *, remove_commas: bool = False) -> pd.Series:
    """Convert a series to numeric, mapping unparseable strings to missing."""
    text = values.astype("string").str.strip()
    if remove_commas:
        text = text.str.replace(",", "", regex=False)
    text = text.replace({"": pd.NA, "-": pd.NA, "NAN": pd.NA})
    return pd.to_numeric(text, errors="coerce")


def standardize_yes_no(values: pd.Series) -> pd.Series:
    """Map common yes/no spellings to title-case Yes/No; preserve unknowns as NA."""
    normalized = values.astype("string").str.strip().str.casefold()
    mapping = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
               "no": "No", "n": "No", "false": "No", "0": "No"}
    return normalized.map(mapping).astype("string")


def clean_text_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Trim outer whitespace and collapse repeated internal whitespace in text."""
    cleaned = data.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else value
        )
        cleaned[column] = cleaned[column].replace("", pd.NA)
    return cleaned


def _category_audit(series: pd.Series) -> dict[str, Any]:
    """Summarize blank-like values and whitespace-normalization collisions."""
    non_missing = series.dropna().astype("string")
    canonical = non_missing.str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    collisions = pd.DataFrame({"original": non_missing, "canonical": canonical})
    variant_groups = collisions.groupby("canonical")["original"].nunique()
    return {
        "distinct_non_missing": int(non_missing.nunique()),
        "blank_or_whitespace": int(series.astype("string").str.strip().eq("").sum()),
        "case_or_whitespace_variant_groups": int((variant_groups > 1).sum()),
        "examples": series.value_counts(dropna=False).head(10).to_dict(),
    }


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean an input dataframe and return (cleaned copy, auditable report).

    Only exact duplicate records are dropped. Rows with a missing rating or
    other incomplete descriptive fields remain available for analyses that do
    not require those fields. Invalid numeric inputs become NaN and are counted.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    original_rows = len(data)
    original_missing = data.isnull().sum()
    duplicates_removed = int(data.duplicated().sum())
    cleaned = data.drop_duplicates().copy()

    # Clean text before parsing so formatting differences are handled uniformly.
    cleaned = clean_text_columns(cleaned)
    before_rating = cleaned[RATING_COLUMN].copy() if RATING_COLUMN in cleaned else None
    if RATING_COLUMN in cleaned:
        cleaned[RATING_COLUMN] = parse_rating(cleaned[RATING_COLUMN])
    if COST_COLUMN in cleaned:
        cost_before = cleaned[COST_COLUMN].copy()
        cleaned[COST_COLUMN] = parse_numeric(cleaned[COST_COLUMN], remove_commas=True)
    else:
        cost_before = None
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns and column != COST_COLUMN:
            cleaned[column] = parse_numeric(cleaned[column])
    for column in BOOLEAN_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = standardize_yes_no(cleaned[column])

    # Invalid costs are non-empty source values that cannot be parsed, or parse
    # to non-positive amounts. High costs are flagged, never auto-dropped.
    invalid_cost_records = 0
    unrealistic_cost_records = 0
    if COST_COLUMN in cleaned.columns and cost_before is not None:
        original_cost_text = cost_before.astype("string").str.strip()
        parsed_cost = cleaned[COST_COLUMN]
        invalid_cost_records = int((original_cost_text.notna() & original_cost_text.ne("") & parsed_cost.isna()).sum())
        invalid_cost_records += int(parsed_cost.le(0).fillna(False).sum())
        unrealistic_cost_records = int(parsed_cost.gt(10000).fillna(False).sum())
        cleaned.loc[parsed_cost.le(0).fillna(False), COST_COLUMN] = np.nan

    invalid_rating_records = 0
    missing_ratings_after = 0
    if RATING_COLUMN in cleaned.columns and before_rating is not None:
        old_rating = before_rating.astype("string").str.strip()
        expected_missing = old_rating.isna() | old_rating.str.upper().isin(["", "NEW", "-"])
        invalid_rating_records = int((~expected_missing & cleaned[RATING_COLUMN].isna()).sum())
        missing_ratings_after = int(cleaned[RATING_COLUMN].isna().sum())

    categorical_audit = {
        col: _category_audit(cleaned[col])
        for col in ("location", "cuisines", "rest_type") if col in cleaned.columns
    }
    invalid_rating_mask = cleaned[RATING_COLUMN].notna() & ~cleaned[RATING_COLUMN].between(0, 5) if RATING_COLUMN in cleaned else pd.Series(dtype=bool)
    invalid_cost_mask = cleaned[COST_COLUMN].notna() & cleaned[COST_COLUMN].le(0) if COST_COLUMN in cleaned else pd.Series(dtype=bool)

    report: dict[str, Any] = {
        "original_rows": original_rows,
        "original_columns": int(data.shape[1]),
        "original_dimensions": [int(data.shape[0]), int(data.shape[1])],
        "duplicates_removed": duplicates_removed,
        "missing_ratings_after_cleaning": missing_ratings_after,
        "invalid_rating_records": invalid_rating_records,
        "invalid_cost_records": invalid_cost_records,
        "unrealistic_cost_records_over_10000": unrealistic_cost_records,
        "final_cleaned_rows": int(len(cleaned)),
        "rows_removed_total": int(original_rows - len(cleaned)),
        "missing_before": {str(k): int(v) for k, v in original_missing.items()},
        "missing_after": {str(k): int(v) for k, v in cleaned.isnull().sum().items()},
        "data_types_after": {str(k): str(v) for k, v in cleaned.dtypes.items()},
        "categorical_audit": categorical_audit,
        "invalid_rating_rows_remaining": int(invalid_rating_mask.sum()),
        "invalid_cost_rows_remaining": int(invalid_cost_mask.sum()),
        "cleaning_decisions": [
            "Removed exact duplicate rows only; repeated restaurant names are retained because they may be separate chain outlets.",
            "Converted NEW, '-', blank, and unparsable ratings to missing; no rating imputation or rating-based row deletion was performed.",
            "Converted cost commas and numeric strings to numbers; invalid or non-positive costs are represented as missing, while costs above 10000 are flagged and retained for review.",
            "Converted invalid numeric vote/cost strings to missing rather than dropping restaurant rows.",
            "Standardized recognized online_order and book_table yes/no values; unknown labels become missing.",
            "Trimmed and collapsed whitespace in text; location, cuisine, and restaurant-type labels are not merged across spelling/case variants.",
            "Kept rows with missing optional fields because those fields are not required for every later question.",
        ],
    }
    return cleaned, report


def save_cleaned_data(data: pd.DataFrame, path: Path | None = None) -> Path:
    """Save cleaned output outside the raw-data directory."""
    destination = Path(path) if path is not None else PROCESSED_DATA_DIR / "zomato_cleaned.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return destination


def save_cleaning_report(report: dict[str, Any], path: Path | None = None) -> Path:
    """Save the cleaning report as readable JSON."""
    destination = Path(path) if path is not None else PROCESSED_DATA_DIR / "cleaning_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return destination


def run_cleaning_pipeline(path: Path = RAW_DATA_PATH, *, save: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, inspect, clean, and optionally save data and its audit report."""
    raw_data = load_raw_data(path)
    cleaned, report = clean_data(raw_data)
    if save:
        save_cleaned_data(cleaned)
        save_cleaning_report(report)
    return cleaned, report
