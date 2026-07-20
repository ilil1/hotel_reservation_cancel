"""Lisbon City Hotel 예약 취소 모델을 학습하고 평가 결과를 저장한다.

처리 순서:
1. 전체 호텔 데이터에서 City Hotel 예약만 추출한다.
2. 예측 시점에 알 수 없는 결과성 변수를 제거한다.
3. 도착일 순서로 학습/검증/테스트 데이터를 나눈다.
4. Logistic Regression과 Random Forest를 비교한다.
5. 검증 데이터에서 임계값을 정하고 테스트 데이터로 최종 평가한다.
6. 모델, 성능 지표, 변수 중요도와 그래프를 파일로 저장한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path("work/.matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "is_canceled"
# 아래 변수는 취소 결과가 확정된 뒤 기록되거나 객실 배정 과정에서 갱신될 수 있다.
# 실제 예약 시점 예측에 사용하면 미래 정보가 섞이는 데이터 누수가 발생한다.
LEAKAGE_COLUMNS = {
    "reservation_status",
    "reservation_status_date",
    # Usually unknown at the moment the booking is first scored.
    "assigned_room_type",
}


def parse_args() -> argparse.Namespace:
    """명령행에서 데이터 경로, 결과 경로와 난수 시드를 읽는다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/hotel_bookings.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_city_hotel(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """원본 CSV를 검증하고 City Hotel의 설명 변수와 정답을 반환한다."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found: {path}. Download hotel_bookings.csv from Kaggle; see README.md."
        )
    frame = pd.read_csv(path)
    required = {"hotel", TARGET, "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    city = frame.loc[frame["hotel"].eq("City Hotel")].copy()
    if city.empty:
        raise ValueError("No rows where hotel == 'City Hotel'.")

    # 연/월/일 열을 날짜 하나로 합쳐 정확한 시간순 분할 기준을 만든다.
    city["arrival_date"] = pd.to_datetime(
        city["arrival_date_year"].astype(str)
        + "-"
        + city["arrival_date_month"].astype(str)
        + "-"
        + city["arrival_date_day_of_month"].astype(str),
        errors="raise",
    )
    city = city.sort_values("arrival_date", kind="stable").reset_index(drop=True)
    y = city.pop(TARGET).astype(int)
    drop = [c for c in LEAKAGE_COLUMNS | {"hotel"} if c in city.columns]
    return city.drop(columns=drop), y


def chronological_split(x: pd.DataFrame, y: pd.Series):
    """과거 70%, 중간 15%, 최신 15% 순서로 데이터를 분리한다.

    무작위 분할보다 실제 운영 상황(과거 데이터로 미래 예약 예측)에 가깝고,
    같은 시기의 패턴이 학습과 테스트에 섞이는 것을 줄인다.
    """
    n = len(x)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    if train_end < 20 or valid_end == train_end or valid_end == n:
        raise ValueError("At least 30 chronologically ordered City Hotel rows are required.")
    return (
        x.iloc[:train_end].copy(), y.iloc[:train_end].copy(),
        x.iloc[train_end:valid_end].copy(), y.iloc[train_end:valid_end].copy(),
        x.iloc[valid_end:].copy(), y.iloc[valid_end:].copy(),
    )


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """숫자형과 범주형 변수에 알맞은 전처리기를 구성한다."""
    categorical = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = x.select_dtypes(include=["number"]).columns.tolist()
    # Arrival date is used for splitting, not as a raw identifier-like feature.
    categorical = [c for c in categorical if c != "arrival_date"]
    numeric = [c for c in numeric if c != "arrival_date"]
    # 숫자 결측치는 중앙값으로 채우고 표준화한다.
    # 범주 결측치는 최빈값으로 채운 뒤 원-핫 인코딩한다.
    # 학습 중 5회 미만 등장한 희귀 범주는 묶어 과적합과 차원 증가를 줄인다.
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical),
        ],
        remainder="drop",
    )


def metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    """주어진 임계값으로 확률을 분류하고 핵심 성능 지표를 계산한다."""
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


def best_f1_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    """검증 데이터에서 F1 점수가 가장 높은 취소 판단 임계값을 찾는다."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(scores))])


def save_plots(y_valid, valid_probability, threshold, y_test, test_probability, output: Path) -> None:
    """검증 PR 곡선과 테스트 혼동행렬을 PNG 파일로 저장한다."""
    precision, recall, _ = precision_recall_curve(y_valid, valid_probability)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Validation precision-recall curve (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(output / "precision_recall_curve.png", dpi=160)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, test_probability >= threshold, cmap="Blues")
    plt.title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close()


def main() -> None:
    """전체 학습·모델 선택·테스트 평가·결과 저장 파이프라인을 실행한다."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, y = load_city_hotel(args.data)
    x_train, y_train, x_valid, y_valid, x_test, y_test = chronological_split(x, y)

    # 선형 기준 모델과 비선형 상호작용을 학습할 수 있는 트리 모델을 비교한다.
    # class_weight는 취소/비취소 비율 차이를 학습 과정에 반영한다.
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", C=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=args.random_state,
        ),
    }
    fitted: dict[str, Pipeline] = {}
    comparisons = []
    # 두 후보 모두 학습 세트로만 훈련하고 검증 세트에서 비교한다.
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", build_preprocessor(x_train)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_valid)[:, 1]
        threshold = best_f1_threshold(y_valid, probability)
        row = {"model": name, **metrics(y_valid, probability, threshold)}
        comparisons.append(row)
        fitted[name] = pipeline

    comparison = pd.DataFrame(comparisons).sort_values(["pr_auc", "f1"], ascending=False)
    comparison.to_csv(args.output / "model_comparison.csv", index=False)
    # 운영상 취소 고객 탐지가 중요하므로 불균형 분류에 적합한 PR-AUC를 우선한다.
    winner_name = str(comparison.iloc[0]["model"])
    winner = fitted[winner_name]
    valid_probability = winner.predict_proba(x_valid)[:, 1]
    threshold = best_f1_threshold(y_valid, valid_probability)
    # 테스트 세트는 모델과 임계값 선택이 모두 끝난 뒤 한 번만 평가한다.
    test_probability = winner.predict_proba(x_test)[:, 1]

    report = {
        "selected_model": winner_name,
        "selection_rule": "highest validation PR-AUC, then F1",
        "validation": metrics(y_valid, valid_probability, threshold),
        "test": metrics(y_test, test_probability, threshold),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 트리 중요도 또는 회귀계수 절댓값을 공통 형식으로 저장한다.
    preprocessor = winner.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out()
    estimator = winner.named_steps["model"]
    importance = estimator.feature_importances_ if hasattr(estimator, "feature_importances_") else np.abs(estimator.coef_[0])
    pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(args.output / "feature_importance.csv", index=False)

    joblib.dump(winner, args.output / "model.joblib")
    metadata = {
        "threshold": threshold,
        "selected_model": winner_name,
        "feature_columns": x.columns.tolist(),
        "leakage_columns_removed": sorted(LEAKAGE_COLUMNS),
        "data_rows": len(x),
        "date_min": str(x["arrival_date"].min().date()),
        "date_max": str(x["arrival_date"].max().date()),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_plots(y_valid, valid_probability, threshold, y_test, test_probability, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
