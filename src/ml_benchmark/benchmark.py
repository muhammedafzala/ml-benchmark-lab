"""Compare supervised pipelines using shared folds and a reserved holdout."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import mean, stdev
from time import perf_counter

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ml_benchmark.preprocessing import DatasetSplit


@dataclass(frozen=True)
class BenchmarkResult:
    leaderboard: pd.DataFrame
    fold_scores: pd.DataFrame
    selected_model: str
    selected_predictions: pd.DataFrame
    positive_label: int | str | None = None
    positive_scores: dict[str, pd.Series] = field(default_factory=dict)


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Return errors in the target's units and the unitless R² score."""
    return {
        "test_mae": float(mean_absolute_error(actual, predicted)),
        "test_rmse": float(root_mean_squared_error(actual, predicted)),
        "test_r2": float(r2_score(actual, predicted)),
    }


def classification_metrics(actual, predicted, scores, *, positive_label) -> dict[str, float]:
    """Score binary labels and rank probabilities for the designated positive class."""
    return {
        "test_accuracy": float(accuracy_score(actual, predicted)),
        "test_positive_precision": float(precision_score(
            actual, predicted, pos_label=positive_label, zero_division=0
        )),
        "test_positive_recall": float(recall_score(
            actual, predicted, pos_label=positive_label, zero_division=0
        )),
        "test_positive_f1": float(f1_score(
            actual, predicted, pos_label=positive_label, zero_division=0
        )),
        "test_roc_auc": float(roc_auc_score(pd.Series(actual).eq(positive_label), scores)),
    }


def run_benchmark(
    split: DatasetSplit,
    preprocessor: ColumnTransformer,
    models: Mapping[str, RegressorMixin | ClassifierMixin],
    *,
    n_splits: int,
    random_state: int,
    task: str = "regression",
    positive_label: int | str | None = None,
) -> BenchmarkResult:
    """Rank on training folds, then score each refitted pipeline on the holdout."""
    if not models:
        raise ValueError("Provide at least one model to benchmark.")
    if task not in ("regression", "classification"):
        raise ValueError(f"Unknown benchmark task: {task!r}.")
    is_classification = task == "classification"
    if is_classification:
        counts = split.y_train.value_counts()
        if len(counts) != 2 or positive_label not in counts.index:
            raise ValueError("Classification requires two classes and an explicit positive label.")
        if counts.min() < n_splits or set(split.y_test.unique()) != set(counts.index):
            raise ValueError("Both classes must be present in each training fold and the test set.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scorer = make_scorer(f1_score, pos_label=positive_label, zero_division=0)
        metric, direction = "f1", -1
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scorer = "neg_mean_absolute_error"
        metric, direction = "mae", 1
    if not is_classification and len(split.y_test) < 2:
        raise ValueError("The test split needs at least two rows to calculate R².")

    folds = list(splitter.split(split.X_train, split.y_train))
    pipelines = {
        name: Pipeline([("preprocess", clone(preprocessor)), ("model", clone(model))])
        for name, model in models.items()
    }
    cv_summary = {}
    fold_scores = []
    mean_column, std_column = f"cv_{metric}_mean", f"cv_{metric}_std"
    for name, pipeline in pipelines.items():
        # sklearn negates error scores; convert MAE back to its usual positive units.
        scores = (-direction * cross_val_score(
            pipeline, split.X_train, split.y_train, cv=folds,
            scoring=scorer, n_jobs=1, error_score="raise",
        )).tolist()
        cv_summary[name] = {mean_column: mean(scores), std_column: stdev(scores)}
        fold_scores.extend(
            {"model": name, "fold": fold, metric: score}
            for fold, score in enumerate(scores, start=1)
        )

    ranking = sorted(pipelines, key=lambda name: (direction * cv_summary[name][mean_column], name))
    selected_model = ranking[0]
    leaderboard = []
    selected_predictions = pd.DataFrame({"actual": split.y_test.to_numpy()})
    positive_scores = {}
    for rank, name in enumerate(ranking, start=1):
        pipeline = pipelines[name]
        start = perf_counter()
        pipeline.fit(split.X_train, split.y_train)
        fit_seconds = perf_counter() - start

        start = perf_counter()
        predictions = pipeline.predict(split.X_test)
        predict_seconds = perf_counter() - start

        if is_classification:
            # Column zero is the positive score only when classes_[0] is the positive label.
            column = pipeline.classes_.tolist().index(positive_label)
            scores = pipeline.predict_proba(split.X_test)[:, column]
            positive_scores[name] = pd.Series(scores)
            metrics = classification_metrics(split.y_test, predictions, scores, positive_label=positive_label)
        else:
            metrics = regression_metrics(split.y_test, predictions)
        leaderboard.append({
            "rank": rank, "model": name, **cv_summary[name],
            **metrics,
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        })
        if name == selected_model:
            selected_predictions["predicted"] = predictions
            if not is_classification:
                selected_predictions["residual"] = selected_predictions["actual"] - predictions

    return BenchmarkResult(
        pd.DataFrame(leaderboard), pd.DataFrame(fold_scores), selected_model, selected_predictions,
        positive_label if is_classification else None, positive_scores,
    )
