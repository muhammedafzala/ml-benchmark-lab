"""Command-line entry point for ML Benchmark Lab."""

from __future__ import annotations

import argparse
from hashlib import file_digest
from importlib.metadata import version
from os import environ
from pathlib import Path
import platform
import tomllib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame
    from sklearn.compose import ColumnTransformer

    from ml_benchmark.preprocessing import DatasetSplit


def load_experiment(
    config: dict, config_path: Path
) -> tuple[DataFrame, DatasetSplit, ColumnTransformer, Path]:
    """Build the split and unfitted preprocessing from the experiment settings."""
    from ml_benchmark.datasets import load_csv
    from ml_benchmark.preprocessing import build_preprocessor, split_dataset

    dataset = config["dataset"]
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a TOML table.")
    numeric = dataset.get("numeric_features", [])
    categorical = dataset.get("categorical_features", [])
    for key, features in (("numeric_features", numeric), ("categorical_features", categorical)):
        if not isinstance(features, list) or not all(
            isinstance(name, str) and name.strip() for name in features
        ):
            raise ValueError(f"dataset.{key} must be a list of non-empty column names.")
    target = dataset["target"]
    split_settings = config["split"]
    if not isinstance(split_settings, dict):
        raise ValueError("split must be a TOML table.")
    data_path = config_path.parent / dataset["path"]
    frame = load_csv(data_path, required_columns=[*numeric, *categorical, target])
    split = split_dataset(
        frame, features=[*numeric, *categorical], target=target,
        task=config.get("task", "regression"), **split_settings,
    )
    preprocessor = build_preprocessor(numeric_features=numeric, categorical_features=categorical)
    return frame, split, preprocessor, data_path


def prepare_data(config: dict, config_path: Path) -> None:
    """Summarize the split and check preprocessing on training rows only."""
    frame, split, preprocessor, _ = load_experiment(config, config_path)
    transformed = preprocessor.fit_transform(split.X_train)
    print(f"Raw dataset: {len(frame):,} rows × {len(frame.columns)} columns")
    print(
        f"After exact repeat removal: {len(frame) - split.duplicates_removed:,} rows "
        f"× {len(frame.columns)} columns"
    )
    print(f"Training: {len(split.X_train):,} rows × {split.X_train.shape[1]} input columns")
    print(f"Reserved test: {len(split.X_test):,} rows × {split.X_test.shape[1]} input columns")
    print(f"Transformed training features: {transformed.shape[1]}")


def benchmark_experiment(config: dict, config_path: Path) -> None:
    """Run the configured benchmark and save its reports."""
    from ml_benchmark.benchmark import run_benchmark
    from ml_benchmark.models import create_classifiers, create_regressors
    from ml_benchmark.reporting import save_results

    frame, split, preprocessor, data_path = load_experiment(config, config_path)
    settings = config["benchmark"]
    if not isinstance(settings, dict):
        raise ValueError("benchmark must be a TOML table.")
    seed = config["split"]["random_state"]
    task = config.get("task", "regression")
    is_classification = task == "classification"
    positive_label = settings["positive_label"] if is_classification else None
    positive_name = settings.get("positive_name", "Positive")
    negative_name = settings.get("negative_name", "Negative")
    target_label = "" if is_classification else settings["target_label"]
    models = create_classifiers(seed) if is_classification else create_regressors(seed)
    output = config_path.parent / settings["output"]
    with data_path.open("rb") as dataset_file:
        checksum = file_digest(dataset_file, "sha256").hexdigest()
    result = run_benchmark(
        split, preprocessor, models, n_splits=settings["cv_folds"], random_state=seed,
        task=task, positive_label=positive_label,
    )
    metadata = {
        "experiment": config["name"],
        "task": task,
        "dataset": {
            **{key: value for key, value in config["dataset"].items() if key != "path"},
            "sha256": checksum,
        },
        "rows": {
            "raw": len(frame),
            "exact_repeats_removed": split.duplicates_removed,
            "train": len(split.X_train),
            "test": len(split.X_test),
        },
        "split": {**config["split"], "stratified": is_classification},
        "cross_validation": {
            "method": "StratifiedKFold" if is_classification else "KFold",
            "n_splits": settings["cv_folds"],
            "shuffle": True,
            "random_state": seed,
            "ranking": (
                "Highest mean validation positive-class F1, with alphabetical tie-breaks"
                if is_classification else "Lowest mean validation MAE, with alphabetical tie-breaks"
            ),
            "standard_deviation": "Sample standard deviation across folds (ddof=1)",
        },
        "models": {
            name: {"estimator": type(model).__name__, "parameters": model.get_params(deep=False)}
            for name, model in models.items()
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "omp_num_threads": environ.get("OMP_NUM_THREADS"),
            "packages": {
                name: version(name)
                for name in (
                    "ml-benchmark-lab", "pandas", "scikit-learn", "matplotlib",
                    "numpy", "scipy", "joblib", "threadpoolctl",
                )
            },
        },
        "timing": (
            "One sequential fit on all training rows and one batch predict on all test rows per "
            "pipeline using perf_counter wall time. Both include preprocessing and exclude cross-"
            "validation, metric computation, data loading, and report writing. No repeated trials."
        ),
    }
    if is_classification:
        metadata["classification"] = {
            "positive_label": positive_label,
            "positive_name": positive_name,
            "negative_name": negative_name,
            "class_counts": {
                "raw": frame[config["dataset"]["target"]].value_counts().to_dict(),
                "train": split.y_train.value_counts().to_dict(),
                "test": split.y_test.value_counts().to_dict(),
            },
            "roc_auc_input": "predict_proba column for the configured positive class",
        }
        metadata["timing"] += " The separately computed ROC probabilities are excluded."
    else:
        metadata["target_unit"] = target_label
    save_results(
        result, output, metadata=metadata, target_label=target_label,
        positive_name=positive_name, negative_name=negative_name,
    )
    print(result.leaderboard.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    metric = "positive-class F1" if is_classification else "MAE"
    print(f"Selected by cross-validation {metric}: {result.selected_model}")
    print(f"Saved benchmark reports to {output.resolve()}")


def main() -> None:
    """Read an experiment configuration and print its name."""
    parser = argparse.ArgumentParser(description="Read an ML Benchmark Lab experiment config.")
    parser.add_argument("config", type=Path, help="Path to a TOML experiment config")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--prepare-data", action="store_true", help="Check the split and training preprocessing"
    )
    actions.add_argument("--benchmark", action="store_true", help="Run the configured benchmark")
    args = parser.parse_args()

    try:
        with args.config.open("rb") as config_file:
            config = tomllib.load(config_file)
    except (OSError, ValueError) as error:
        parser.error(f"Could not read {args.config}: {error}")

    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        parser.error("The experiment config must include a non-empty name.")

    print(f"Experiment: {name}")
    try:
        if args.prepare_data:
            prepare_data(config, args.config)
        if args.benchmark:
            benchmark_experiment(config, args.config)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(f"Could not run the experiment: {error}")


if __name__ == "__main__":
    main()
