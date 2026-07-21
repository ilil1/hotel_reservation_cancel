"""모델 성능 계산, 모델 선택, 그래프와 결과 파일 저장을 담당한다."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/.matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def calculate_metrics(y_true, probability, threshold: float) -> dict[str, float | int]:
    """예측 확률과 임계값으로 주요 분류 성능을 계산한다."""
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 6),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 6),
        "rows": int(len(y_true)),
        "cancellation_rate": round(float(y_true.mean()), 6),
    }


def select_best_model(fitted_models: dict[str, Pipeline], x_valid, y_valid):
    """기본 임계값 0.5의 검증 F1과 Accuracy로 최종 모델을 선택한다."""
    comparisons = []
    for name, model in fitted_models.items():
        probability = model.predict_proba(x_valid)[:, 1]
        comparisons.append(
            {"model": name, **calculate_metrics(y_valid, probability, 0.5)}
        )

    table = pd.DataFrame(comparisons).sort_values(
        ["f1", "accuracy"], ascending=False
    )
    winner_name = str(table.iloc[0]["model"])
    return winner_name, fitted_models[winner_name], table


def save_plots(y_valid, valid_probability, threshold, y_test, test_probability, output: Path):
    """테스트 혼동행렬을 PNG로 저장한다."""
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_probability >= threshold, cmap="Blues"
    )
    plt.title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close()


def save_feature_importance(model: Pipeline, output: Path):
    """트리 중요도 또는 회귀계수 절댓값을 동일한 CSV 형식으로 저장한다."""
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importance = (
        estimator.feature_importances_
        if hasattr(estimator, "feature_importances_")
        else np.abs(estimator.coef_[0])
    )
    pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(output / "feature_importance.csv", index=False)


def save_model_comparison_plot(comparison: pd.DataFrame, output: Path) -> None:
    """후보 모델의 검증 성능을 하나의 막대그래프로 비교한다."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    plot_data = comparison.set_index("model")[metrics_to_plot].transpose()
    ax = plot_data.plot(kind="bar", figsize=(12, 6), width=0.75)
    ax.set_title("Validation performance by model")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "model_comparison.png", dpi=160)
    plt.close()


def save_classification_details(y_test, test_probability, threshold: float, output: Path) -> None:
    """테스트 Classification Report와 혼동행렬 원본 수치를 저장한다."""
    predicted = (test_probability >= threshold).astype(int)
    report = classification_report(
        y_test,
        predicted,
        labels=[0, 1],
        target_names=["Not canceled", "Canceled"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output / "classification_report.csv")

    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_not_canceled", "actual_canceled"],
        columns=["predicted_not_canceled", "predicted_canceled"],
    ).to_csv(output / "confusion_matrix.csv")


def save_results(
    model: Pipeline,
    model_name: str,
    comparison: pd.DataFrame,
    x,
    y_valid,
    valid_probability,
    y_test,
    test_probability,
    threshold: float,
    leakage_columns: set[str],
    output: Path,
) -> dict:
    """모델, 성능 지표, 메타데이터, 중요도와 그래프를 저장한다."""
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "selected_model": model_name,
        "selection_rule": "highest validation F1, then Accuracy",
        "validation": calculate_metrics(y_valid, valid_probability, threshold),
        "test": calculate_metrics(y_test, test_probability, threshold),
    }
    metadata = {
        "threshold": threshold,
        "selected_model": model_name,
        "feature_columns": x.columns.tolist(),
        "leakage_columns_removed": sorted(leakage_columns),
        "data_rows": len(x),
        "date_min": str(x["arrival_date"].min().date()),
        "date_max": str(x["arrival_date"].max().date()),
    }

    joblib.dump(model, output / "model.joblib")
    comparison.to_csv(output / "model_comparison.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_feature_importance(model, output)
    save_model_comparison_plot(comparison, output)
    save_classification_details(y_test, test_probability, threshold, output)
    save_plots(
        y_valid,
        valid_probability,
        threshold,
        y_test,
        test_probability,
        output,
    )
    return report
