"""Smoke test for the installed package's command-line interface."""

from pathlib import Path
import subprocess
import sys


def test_cli_reads_used_car_config() -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "used_car_prices.toml"
    result = subprocess.run(
        [sys.executable, "-m", "ml_benchmark", str(config)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "Experiment: Used car price prediction"


def test_cli_rejects_string_feature_list(tmp_path: Path) -> None:
    config = tmp_path / "invalid.toml"
    config.write_text(
        'name = "Example"\n[dataset]\nnumeric_features = "distance"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "ml_benchmark", str(config), "--prepare-data"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "dataset.numeric_features must be a list" in result.stderr
