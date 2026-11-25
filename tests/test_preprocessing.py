import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_benchmark.preprocessing import build_preprocessor, split_dataset


def test_split_removes_exact_repeats_and_is_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "year": [2010, 2010, 2010, 2011, 2012, 2013],
            "name": ["Car A", "Car A", "Car B", "Car C", "Car D", "Car E"],
            "price": [100, 100, 100, 200, 300, 400],
        }
    )
    settings = dict(features=["year"], target="price", test_size=0.20, random_state=42)
    first = split_dataset(frame, **settings)
    second = split_dataset(frame, **settings)

    assert first.duplicates_removed == 1
    assert first.X_train.shape == (4, 1)
    assert first.X_test.shape == (1, 1)
    assert set(first.X_train.index) | set(first.X_test.index) == {0, 2, 3, 4, 5}
    assert first.X_train.index.intersection(first.X_test.index).empty
    for features, target in [(first.X_train, first.y_train), (first.X_test, first.y_test)]:
        assert list(features.columns) == ["year"]
        pd.testing.assert_series_equal(target, frame.loc[features.index, "price"])
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    assert len(frame) == 6


def test_split_rejects_missing_target_values() -> None:
    frame = pd.DataFrame({"year": [2010, 2011], "price": [100, None]})
    with pytest.raises(ValueError, match="Target column .* contains missing values"):
        split_dataset(frame, features=["year"], target="price", test_size=0.20, random_state=42)


def test_split_rejects_target_as_feature() -> None:
    frame = pd.DataFrame({"price": [100, 200]})
    with pytest.raises(ValueError, match="cannot also be an input feature"):
        split_dataset(frame, features=["price"], target="price", test_size=0.20, random_state=42)


def test_pipeline_imputes_scales_and_ignores_unseen_categories() -> None:
    training = pd.DataFrame({"distance": [10, 30, None], "fuel": ["Petrol", "Diesel", None]})
    held_out = pd.DataFrame({"distance": [None, 1000], "fuel": [pd.NA, "Electric"]})
    preprocessor = build_preprocessor(numeric_features=["distance"], categorical_features=["fuel"])
    pipeline = Pipeline([("preprocess", preprocessor)])
    pipeline.fit(training)
    names = preprocessor.get_feature_names_out()
    result = pipeline.transform(held_out)
    encoded = pd.DataFrame(result.toarray() if hasattr(result, "toarray") else result, columns=names)

    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(20)
    assert scaler.var_[0] == pytest.approx(200 / 3)
    assert encoded.loc[0, "numeric__distance"] == pytest.approx(0)
    assert encoded.loc[0, "categorical__fuel___missing__"] == 1
    assert encoded.filter(like="categorical__").iloc[1].eq(0).all()
    assert not encoded.isna().any().any()


def test_numeric_category_codes_can_include_missing_values() -> None:
    training = pd.DataFrame({"code": [1.0, 2.0, None]})
    transformer = build_preprocessor(numeric_features=[], categorical_features=["code"])

    transformed = transformer.fit_transform(training)

    assert transformed.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert transformer.get_feature_names_out().tolist() == [
        "categorical__code_1.0", "categorical__code_2.0", "categorical__code___missing__",
    ]
