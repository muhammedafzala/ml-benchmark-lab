"""Write benchmark tables, run details, and charts."""

from collections.abc import Mapping
import json
from pathlib import Path

from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

from ml_benchmark.benchmark import BenchmarkResult


def save_results(
    result: BenchmarkResult,
    output: Path,
    *,
    metadata: Mapping,
    target_label: str = "",
    positive_name: str = "Positive",
    negative_name: str = "Negative",
) -> None:
    """Save comparison charts and diagnostics in the run's output directory."""
    output.mkdir(parents=True, exist_ok=True)
    leaderboard = result.leaderboard
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    result.fold_scores.to_csv(output / "cv_fold_scores.csv", index=False)
    (output / "run_metadata.json").write_text(
        json.dumps({**metadata, "selected_model": result.selected_model}, indent=2) + "\n",
        encoding="utf-8",
    )
    names = leaderboard["model"].tolist()
    positions = list(range(len(names)))
    save_timing_chart(leaderboard, names, positions, output)

    if result.positive_label is not None:
        save_classification_charts(result, output, positive_name, negative_name)
        return

    fig, ax = comparison_chart(
        names,
        "Training cross-validation MAE",
        f"Mean MAE ({target_label})  •  Error bars show one fold standard deviation",
    )
    ax.barh(
        positions, leaderboard["cv_mae_mean"], xerr=leaderboard["cv_mae_std"],
        capsize=3, color="#246b83",
    )
    fig.savefig(output / "cv-mae.png", dpi=160)

    fig, ax = comparison_chart(names, "Reserved test errors", f"Error ({target_label})")
    ax.barh([p - 0.18 for p in positions], leaderboard["test_mae"], 0.36, label="MAE", color="#246b83")
    ax.barh([p + 0.18 for p in positions], leaderboard["test_rmse"], 0.36, label="RMSE", color="#b45e39")
    ax.legend(frameon=False)
    fig.savefig(output / "test-errors.png", dpi=160)

    fig, ax = comparison_chart(names, "Reserved test R²", "R² score")
    ax.barh(positions, leaderboard["test_r2"], color="#246b83")
    ax.axvline(0, color="#495766", linewidth=1)
    fig.savefig(output / "test-r2.png", dpi=160)

    predictions = result.selected_predictions
    fig = Figure(figsize=(6.5, 5), layout="constrained")
    ax = fig.subplots()
    ax.scatter(predictions["actual"], predictions["predicted"], s=14, alpha=0.45, color="#246b83")
    low = predictions[["actual", "predicted"]].min().min()
    high = predictions[["actual", "predicted"]].max().max()
    ax.plot([low, high], [low, high], "--", color="#b45e39", label="Perfect predictions")
    ax.set(
        title=f"Actual vs predicted • {result.selected_model}",
        xlabel=f"Actual ({target_label})", ylabel=f"Predicted ({target_label})",
    )
    ax.legend(frameon=False)
    fig.savefig(output / "actual-vs-predicted.png", dpi=160)

    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.scatter(predictions["predicted"], predictions["residual"], s=14, alpha=0.45, color="#246b83")
    ax.axhline(0, linestyle="--", color="#b45e39")
    ax.set(
        title=f"Reserved test residuals • {result.selected_model}",
        xlabel=f"Predicted ({target_label})", ylabel=f"Actual minus predicted ({target_label})",
    )
    fig.savefig(output / "residuals.png", dpi=160)


def comparison_chart(names, title, axis_label):
    fig = Figure(figsize=(9, 4.3), layout="constrained")
    ax = fig.subplots()
    ax.set(title=title, xlabel=axis_label, yticks=range(len(names)), yticklabels=names)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.2)
    return fig, ax


def save_timing_chart(leaderboard, names, positions, output):
    fig = Figure(figsize=(10, 4.3), layout="constrained")
    fit_ax, predict_ax = fig.subplots(1, 2)
    for ax, column, title in [
        (fit_ax, "fit_seconds", "Fit on all training rows"),
        (predict_ax, "predict_seconds", "Predict all test rows"),
    ]:
        ax.barh(positions, leaderboard[column] * 1000, color="#246b83")
        ax.set(title=title, xlabel="Wall time (ms)", yticks=positions, yticklabels=names)
        ax.invert_yaxis()
    fig.savefig(output / "timing.png", dpi=160)


def save_classification_charts(result, output, positive_name, negative_name):
    leaderboard = result.leaderboard
    names = leaderboard["model"].tolist()
    positions = range(len(names))
    fig, ax = comparison_chart(
        names, f"Training cross-validation F1 • {positive_name}",
        "Mean F1  •  Error bars show one fold standard deviation",
    )
    ax.barh(positions, leaderboard["cv_f1_mean"], xerr=leaderboard["cv_f1_std"],
            capsize=3, color="#246b83")
    ax.set_xlim(0, max(1, (leaderboard["cv_f1_mean"] + leaderboard["cv_f1_std"]).max()))
    fig.savefig(output / "cv-f1.png", dpi=160)

    fig = Figure(figsize=(12, 4.2), layout="constrained")
    fig.suptitle(f"Reserved test metrics • Positive class: {positive_name}")
    axes = fig.subplots(1, 5, sharey=True)
    for ax, (column, title) in zip(axes, [
        ("test_accuracy", "Accuracy"), ("test_positive_precision", "Precision"),
        ("test_positive_recall", "Recall"), ("test_positive_f1", "F1"),
        ("test_roc_auc", "ROC-AUC"),
    ], strict=True):
        ax.barh(positions, leaderboard[column], color="#246b83")
        ax.set(title=title, xlim=(0, 1), xticks=(0, 0.5, 1))
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set(yticks=positions, yticklabels=names)
    axes[0].invert_yaxis()
    fig.savefig(output / "test-metrics.png", dpi=160)

    labels = result.selected_predictions
    actual_positive = labels["actual"].eq(result.positive_label)
    predicted_positive = labels["predicted"].eq(result.positive_label)
    matrix = confusion_matrix(actual_positive, predicted_positive, labels=[False, True])
    fig = Figure(figsize=(6, 4.5), layout="constrained")
    ax = fig.subplots()
    ConfusionMatrixDisplay(matrix, display_labels=[negative_name, positive_name]).plot(
        ax=ax, cmap="Blues", colorbar=False,
    )
    ax.set_title(f"Reserved test confusion matrix • {result.selected_model}")
    fig.savefig(output / "confusion-matrix.png", dpi=160)

    fig = Figure(figsize=(7, 5), layout="constrained")
    ax = fig.subplots()
    for name in names:
        false_positive, true_positive, _ = roc_curve(actual_positive, result.positive_scores[name])
        auc = leaderboard.loc[leaderboard["model"].eq(name), "test_roc_auc"].item()
        ax.plot(false_positive, true_positive, label=f"{name} ({auc:.3f})")
    ax.set(
        title=f"Reserved test ROC curves • {positive_name}",
        xlabel="False positive rate", ylabel="True positive rate",
    )
    ax.legend(title="Model (AUC)", loc="lower right", frameon=False)
    fig.savefig(output / "roc-curves.png", dpi=160)
