"""Split tabular data and build preprocessing that can be fitted per fold."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml_benchmark.datasets import check_dataset_columns


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    duplicates_removed: int


def split_dataset(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    test_size: float,
    random_state: int,
    task: str = "regression",
) -> DatasetSplit:
    """Remove exact full-row repeats, then split the features and target together."""
    if not features or len(set(features)) != len(features):
        raise ValueError("Provide at least one feature and list each feature once.")
    if target in features:
        raise ValueError("The target column cannot also be an input feature.")
    if task not in ("regression", "classification"):
        raise ValueError(f"Unknown benchmark task: {task!r}.")

    check_dataset_columns(frame, [*features, target])
    unique = frame.drop_duplicates()
    if unique[target].isna().any():
        raise ValueError(f"Target column {target!r} contains missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        unique.loc[:, list(features)],
        unique[target],
        test_size=test_size,
        random_state=random_state,
        stratify=unique[target] if task == "classification" else None,
    )
    return DatasetSplit(X_train, X_test, y_train, y_test, len(frame) - len(unique))


def _category_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Treat numeric category codes as labels and preserve missing entries."""
    return frame.astype("string")


def build_preprocessor(
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Return an unfitted transformer to place before a model in a pipeline."""
    features = [*numeric_features, *categorical_features]
    if not features or len(set(features)) != len(features):
        raise ValueError("Provide at least one feature and list each feature once.")

    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("labels", FunctionTransformer(_category_strings, feature_names_out="one-to-one")),
            (
                "impute",
                SimpleImputer(
                    missing_values=pd.NA,
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_features)),
            ("categorical", categorical, list(categorical_features)),
        ]
    )
