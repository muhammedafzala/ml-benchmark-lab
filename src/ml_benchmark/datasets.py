"""Load CSV datasets and summarize data quality without changing the rows."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetReport:
    rows: int
    columns: int
    duplicate_rows: int
    columns_report: pd.DataFrame


def load_csv(path: str | Path, *, required_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Read a CSV and check its shape and required headers."""
    frame = pd.read_csv(path)
    check_dataset_columns(frame, required_columns)
    return frame


def check_dataset_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Require a nonempty table with unique names and the requested columns."""
    if not frame.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")


def validate_dataset(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] = (),
    rules: Mapping[str, Callable[[pd.Series], pd.Series]] | None = None,
) -> DatasetReport:
    """Count missing values, repeated rows, and values failing column rules.

    A rule returns True for a valid value. Missing values are counted
    separately. Duplicate counts exclude the first copy of each row.
    """
    rules = {} if rules is None else rules
    check_dataset_columns(frame, [*required_columns, *rules])

    columns_report = pd.DataFrame(
        {"dtype": frame.dtypes.astype(str), "missing": frame.isna().sum()}
    )
    columns_report["invalid"] = pd.Series(pd.NA, index=frame.columns, dtype="Int64")
    for column, rule in rules.items():
        values = frame[column].dropna()
        valid = rule(values)
        if (
            not isinstance(valid, pd.Series)
            or not valid.index.equals(values.index)
            or not pd.api.types.is_bool_dtype(valid)
        ):
            raise ValueError(f"Rule for {column!r} must return a boolean Series on the same index.")
        columns_report.loc[column, "invalid"] = int((~valid.fillna(False)).sum())

    return DatasetReport(
        rows=len(frame),
        columns=len(frame.columns),
        duplicate_rows=int(frame.duplicated().sum()),
        columns_report=columns_report,
    )
