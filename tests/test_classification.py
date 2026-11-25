import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from ml_benchmark import benchmark
from ml_benchmark.__main__ import load_experiment
from ml_benchmark.models import create_classifiers
from ml_benchmark.preprocessing import build_preprocessor
from ml_benchmark.reporting import save_results


def test_binary_metrics_use_label_zero_and_probabilities() -> None:
    result = benchmark.classification_metrics(
        [0, 0, 0, 1, 1], [0, 1, 1, 0, 1], [0.9, 0.4, 0.3, 0.6, 0.1],
        positive_label=0,
    )
    assert result == pytest.approx({
        "test_accuracy": 2 / 5,
        "test_positive_precision": 1 / 2,
        "test_positive_recall": 1 / 3,
        "test_positive_f1": 2 / 5,
        "test_roc_auc": 4 / 6,
    })


def test_classification_config_stratifies_and_saves_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "breast_cancer.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    data = load_breast_cancer(as_frame=True)
    assert data.target_names.tolist() == ["malignant", "benign"]
    frame = data.frame
    frame.to_csv(tmp_path / "fixture.csv", index=False)
    config["dataset"]["path"] = "fixture.csv"
    _, split, preprocessor, _ = load_experiment(config, tmp_path / config_path.name)
    _, same_split, _, _ = load_experiment(config, tmp_path / config_path.name)
    pd.testing.assert_series_equal(split.y_train, same_split.y_train)
    pd.testing.assert_series_equal(split.y_test, same_split.y_test)
    assert split.y_train.value_counts().to_dict() == {0: 170, 1: 285}
    assert split.y_test.value_counts().to_dict() == {0: 42, 1: 72}
    assert split.y_train.index.intersection(split.y_test.index).empty

    folds_seen = []
    cross_val_score = benchmark.cross_val_score

    def record_folds(estimator, X, y, *, cv, **kwargs):
        folds_seen.append(cv)
        for train, validation in cv:
            assert y.iloc[validation].value_counts().to_dict() == {0: 34, 1: 57}
            assert y.iloc[train].value_counts().to_dict() == {0: 136, 1: 228}
        return cross_val_score(estimator, X, y, cv=cv, **kwargs)

    monkeypatch.setattr(benchmark, "cross_val_score", record_folds)
    models = create_classifiers(42)
    result = benchmark.run_benchmark(
        split, preprocessor, models,
        n_splits=5, random_state=42, task="classification", positive_label=0,
    )
    assert len(folds_seen) == len(models) == 6
    assert all(folds is folds_seen[0] for folds in folds_seen)
    assert result.leaderboard["cv_f1_mean"].is_monotonic_decreasing
    assert result.fold_scores.groupby("model")["fold"].nunique().eq(5).all()
    assert result.positive_label == 0
    leaderboard = result.leaderboard.set_index("model")
    assert leaderboard.loc["Dummy (most frequent)", "test_positive_recall"] == 0
    assert leaderboard.loc["Dummy (most frequent)", "test_roc_auc"] == 0.5

    manual = Pipeline([
        ("preprocess", build_preprocessor(
            numeric_features=config["dataset"]["numeric_features"], categorical_features=[]
        )),
        ("model", create_classifiers(42)["Logistic Regression"]),
    ]).fit(split.X_train, split.y_train)
    malignant_probability = manual.predict_proba(split.X_test)[:, 0]
    assert result.positive_scores["Logistic Regression"].tolist() == pytest.approx(malignant_probability)
    assert leaderboard.loc["Logistic Regression", "test_roc_auc"] == pytest.approx(
        roc_auc_score(split.y_test.eq(0), malignant_probability)
    )

    output = tmp_path / "reports"
    save_results(
        result, output, metadata={"task": "classification"}, positive_name="Malignant", negative_name="Benign"
    )
    pd.testing.assert_frame_equal(pd.read_csv(output / "leaderboard.csv"), result.leaderboard)
    assert len(pd.read_csv(output / "cv_fold_scores.csv")) == 30
    assert json.loads((output / "run_metadata.json").read_text())["selected_model"] == result.selected_model
    for filename in ("cv-f1", "test-metrics", "confusion-matrix", "roc-curves", "timing"):
        assert (output / f"{filename}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
