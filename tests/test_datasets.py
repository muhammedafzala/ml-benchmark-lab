from pathlib import Path

import pandas as pd
import pytest

from ml_benchmark.datasets import load_csv, validate_dataset


def test_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "cars.csv"
    path.write_text("name,price\nCar A,100\nCar B,200\n", encoding="utf-8")

    dataset = load_csv(path, required_columns=["name", "price"])

    assert list(dataset.columns) == ["name", "price"]
    assert dataset["price"].tolist() == [100, 200]


def test_load_csv_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "cars.csv"
    path.write_text("name\nCar A\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required columns: price"):
        load_csv(path, required_columns=["price"])


def test_load_csv_rejects_header_only_file(tmp_path: Path) -> None:
    path = tmp_path / "cars.csv"
    path.write_text("name,price\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset contains no rows"):
        load_csv(path)


def test_validation_reports_issues_without_changing_rows() -> None:
    dataset = pd.DataFrame({"name": ["A", "A", "B", "C"], "price": [100, 100, None, -1]})

    report = validate_dataset(dataset, rules={"price": lambda values: values.gt(0)})

    assert (report.rows, report.columns, report.duplicate_rows) == (4, 2, 1)
    assert report.columns_report.loc["price"].to_dict() == {
        "dtype": "float64",
        "missing": 1,
        "invalid": 1,
    }
    assert pd.isna(report.columns_report.loc["name", "invalid"])
    assert len(dataset) == 4


def test_validation_checks_columns_used_by_rules() -> None:
    dataset = pd.DataFrame({"name": ["A"]})

    with pytest.raises(ValueError, match="Missing required columns: price"):
        validate_dataset(dataset, rules={"price": lambda values: values.gt(0)})
