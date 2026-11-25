import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from ml_benchmark.__main__ import load_experiment
from ml_benchmark.benchmark import regression_metrics, run_benchmark
from ml_benchmark.models import create_regressors
from ml_benchmark.preprocessing import build_preprocessor, split_dataset
from ml_benchmark.reporting import save_results


def test_create_regressors() -> None:
    models = create_regressors(42)
    assert [type(model) for model in models.values()] == [
        DummyRegressor, Ridge, DecisionTreeRegressor, RandomForestRegressor,
        KNeighborsRegressor, GradientBoostingRegressor,
    ]
    assert models["Dummy (median)"].strategy == "median"
    for name in ("Decision Tree", "Random Forest", "Gradient Boosting"):
        assert models[name].random_state == 42
    assert create_regressors(42)["Ridge"] is not models["Ridge"]


def test_regression_metrics() -> None:
    metrics = regression_metrics([0, 2, 4], [0, 2, 1])
    assert metrics == pytest.approx({"test_mae": 1, "test_rmse": 3 ** 0.5, "test_r2": -0.125})


@pytest.mark.parametrize("config_name", ["used_car_prices", "california_housing"])
def test_regression_configs_use_same_runner(config_name: str, tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / f"{config_name}.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    numeric = dataset["numeric_features"]
    categorical = dataset.get("categorical_features", [])
    target = dataset["target"]
    frame = pd.DataFrame({
        **{feature: range(30) for feature in numeric},
        **{feature: ["a", "b"] * 15 for feature in categorical},
        target: [value * 2 for value in range(30)],
    })
    frame.to_csv(tmp_path / "fixture.csv", index=False)
    dataset["path"] = "fixture.csv"

    _, split, preprocessor, _ = load_experiment(config, tmp_path / config_path.name)
    seed = config["split"]["random_state"]
    result = run_benchmark(
        split, preprocessor, create_regressors(seed),
        n_splits=config["benchmark"]["cv_folds"], random_state=seed,
    )

    assert list(split.X_train.columns) == numeric + categorical
    assert split.y_train.name == target
    assert (len(split.X_train), len(split.X_test)) == (24, 6)
    assert len(result.leaderboard) == 6
    assert len(result.fold_scores) == 30
    assert result.leaderboard["test_mae"].notna().all()


def test_benchmark_ranks_by_training_folds_and_saves_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({
        "distance": range(60),
        "fuel": ["petrol", "diesel"] * 30,
        "price": [100 + 10 * distance for distance in range(60)],
    })
    split = split_dataset(
        frame, features=["distance", "fuel"], target="price", test_size=0.2, random_state=42
    )
    preprocessor = build_preprocessor(numeric_features=["distance"], categorical_features=["fuel"])
    fitted_indices = []
    fit_transform = ColumnTransformer.fit_transform

    def record_fit(transformer, X, y=None):
        fitted_indices.append(tuple(X.index))
        return fit_transform(transformer, X, y)

    monkeypatch.setattr(ColumnTransformer, "fit_transform", record_fit)
    models = create_regressors(42)
    result = run_benchmark(split, preprocessor, models, n_splits=5, random_state=42)

    assert result.selected_model == "Ridge"
    assert result.leaderboard["cv_mae_mean"].is_monotonic_increasing
    assert result.leaderboard["rank"].tolist() == list(range(1, 7))
    assert result.fold_scores.groupby("model")["fold"].nunique().eq(5).all()
    assert not hasattr(preprocessor, "transformers_")
    folds = list(KFold(5, shuffle=True, random_state=42).split(split.X_train))
    expected_indices = [tuple(split.X_train.iloc[train].index) for train, _ in folds]
    assert fitted_indices == (
        expected_indices * len(models) + [tuple(split.X_train.index)] * len(models)
    )
    baseline_scores = []
    for train_indices, validation_indices in folds:
        median = split.y_train.iloc[train_indices].median()
        baseline_scores.append(
            mean_absolute_error(split.y_train.iloc[validation_indices], [median] * len(validation_indices))
        )
    baseline = result.leaderboard.set_index("model").loc["Dummy (median)"]
    assert baseline["cv_mae_mean"] == pytest.approx(pd.Series(baseline_scores).mean())
    assert baseline["cv_mae_std"] == pytest.approx(pd.Series(baseline_scores).std())
    assert baseline["test_mae"] == pytest.approx((split.y_test - split.y_train.median()).abs().mean())
    assert result.selected_predictions["actual"].tolist() == split.y_test.tolist()
    pd.testing.assert_series_equal(
        result.selected_predictions["residual"],
        result.selected_predictions["actual"] - result.selected_predictions["predicted"],
        check_names=False,
    )
    assert result.leaderboard[["fit_seconds", "predict_seconds"]].ge(0).all().all()

    save_results(result, tmp_path, metadata={"seed": 42}, target_label="price units")
    saved = pd.read_csv(tmp_path / "leaderboard.csv")
    pd.testing.assert_frame_equal(saved, result.leaderboard)
    assert len(pd.read_csv(tmp_path / "cv_fold_scores.csv")) == 30
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata == {"seed": 42, "selected_model": "Ridge"}
    for filename in ("cv-mae", "test-errors", "test-r2", "timing", "actual-vs-predicted", "residuals"):
        assert (tmp_path / f"{filename}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
