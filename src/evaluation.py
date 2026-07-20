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
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def calculate_metrics(y_true, probability, threshold: float) -> dict[str, float | int]:
    """예측 확률과 임계값으로 주요 분류 성능을 계산한다."""
    predicted = (probability >= threshold).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y_true, probability)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 6),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 6),
        "threshold": round(float(threshold), 6),
        "rows": int(len(y_true)),
        "cancellation_rate": round(float(y_true.mean()), 6),
    }


def find_best_threshold(y_true, probability) -> float:
    """검증 데이터에서 F1이 가장 높은 취소 판단 임계값을 찾는다."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5

    denominator = np.maximum(precision[:-1] + recall[:-1], 1e-12)
    f1_scores = 2 * precision[:-1] * recall[:-1] / denominator
    return float(thresholds[int(np.nanargmax(f1_scores))])


def select_best_model(fitted_models: dict[str, Pipeline], x_valid, y_valid):
    """검증 PR-AUC와 F1을 기준으로 최종 모델을 선택한다."""
    comparisons = []
    for name, model in fitted_models.items():
        probability = model.predict_proba(x_valid)[:, 1]
        threshold = find_best_threshold(y_valid, probability)
        comparisons.append(
            {"model": name, **calculate_metrics(y_valid, probability, threshold)}
        )

    table = pd.DataFrame(comparisons).sort_values(
        ["pr_auc", "f1"], ascending=False
    )
    winner_name = str(table.iloc[0]["model"])
    return winner_name, fitted_models[winner_name], table


def save_plots(y_valid, valid_probability, threshold, y_test, test_probability, output: Path):
    """검증 PR 곡선과 테스트 혼동행렬을 PNG로 저장한다."""
    precision, recall, _ = precision_recall_curve(y_valid, valid_probability)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Validation precision-recall curve (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(output / "precision_recall_curve.png", dpi=160)
    plt.close()

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
        "selection_rule": "highest validation PR-AUC, then F1",
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
    save_plots(
        y_valid,
        valid_probability,
        threshold,
        y_test,
        test_probability,
        output,
    )
    return report

