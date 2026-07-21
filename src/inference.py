"""저장된 모델을 불러와 신규 City Hotel 예약의 취소 가능성을 예측한다.

Streamlit의 단일 예약 및 CSV 일괄 예측 화면이 같은 전처리·추론 코드를
사용하도록 공통 기능을 이 파일에 모아 둔다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    # dashboard.py에서 패키지 형태(src.inference)로 불러올 때 사용한다.
    from .features import add_engineered_features
except ImportError:
    # src 폴더를 모듈 검색 경로에 둔 환경에서도 불러올 수 있게 한다.
    from features import add_engineered_features


def load_prediction_artifacts(model_dir: Path) -> tuple[Any, dict]:
    """저장된 최종 모델과 학습 당시 입력 변수 정보를 불러온다."""
    model = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def prepare_reservations(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """신규 예약을 학습 때와 동일한 형태로 변환하고 필요한 열을 검사한다."""
    prepared = frame.copy()

    # 전체 호텔 파일이 들어오면 현재 모델의 대상인 City Hotel만 남긴다.
    if "hotel" in prepared.columns:
        prepared = prepared.loc[prepared["hotel"].eq("City Hotel")].copy()
    if prepared.empty:
        raise ValueError("예측할 City Hotel 예약이 없습니다.")

    required_date_columns = {
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    }
    missing_date_columns = sorted(required_date_columns - set(prepared.columns))
    if missing_date_columns:
        raise ValueError(f"도착일을 만드는 데 필요한 열이 없습니다: {missing_date_columns}")

    # 연·월·일을 실제 날짜로 결합해 학습 파이프라인의 입력 형태를 재현한다.
    prepared["arrival_date"] = pd.to_datetime(
        prepared["arrival_date_year"].astype(str)
        + "-"
        + prepared["arrival_date_month"].astype(str)
        + "-"
        + prepared["arrival_date_day_of_month"].astype(str),
        errors="raise",
    )
    prepared = add_engineered_features(prepared)

    missing = sorted(set(feature_columns) - set(prepared.columns))
    if missing:
        raise ValueError(f"모델 예측에 필요한 열이 없습니다: {missing}")
    return prepared


def predict_reservations(
    frame: pd.DataFrame,
    model: Any,
    metadata: dict,
) -> pd.DataFrame:
    """예약별 취소 확률과 임계값에 따른 최종 예측 결과를 반환한다."""
    feature_columns = metadata["feature_columns"]
    prepared = prepare_reservations(frame, feature_columns)
    probability = model.predict_proba(prepared[feature_columns])[:, 1]

    result = prepared.copy()
    result["cancellation_probability"] = probability
    result["predicted_canceled"] = (
        probability >= float(metadata.get("threshold", 0.5))
    ).astype(int)
    return result
