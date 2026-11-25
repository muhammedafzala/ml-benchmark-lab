"""Model settings for laptop-sized benchmarks."""

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def create_regressors(random_state: int) -> dict[str, RegressorMixin]:
    """Create fresh estimators with fixed settings and bounded parallelism."""
    return {
        "Dummy (median)": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=1.0, solver="lsqr"),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_leaf=2,
            random_state=random_state, n_jobs=1,
        ),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            min_samples_leaf=5, random_state=random_state,
        ),
    }


def create_classifiers(random_state: int) -> dict[str, ClassifierMixin]:
    """Create fresh classifiers and a majority-class baseline."""
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        # This solver avoids the obsolete L-BFGS options in scikit-learn 1.6.
        "Logistic Regression": LogisticRegression(solver="newton-cg", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=2,
            random_state=random_state, n_jobs=1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            min_samples_leaf=5, random_state=random_state,
        ),
    }
