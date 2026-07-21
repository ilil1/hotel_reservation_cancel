"""취소 예측 결과를 호텔 운영에 활용할 수 있는 요약 정보로 변환한다."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_risk_levels(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """취소 확률을 낮음·주의·높음의 세 단계로 구분한다."""
    result = predictions.copy()
    probability = result["cancellation_probability"].astype(float)
    caution_threshold = min(0.30, threshold)
    result["risk_level"] = np.select(
        [probability >= threshold, probability >= caution_threshold],
        ["높음", "주의"],
        default="낮음",
    )
    return result


def build_reminder_targets(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """취소 고위험 예약과 고객에게 보낼 수 있는 확인 메시지 초안을 만든다."""
    risk_data = add_risk_levels(predictions, threshold).reset_index(drop=True)
    if "reservation_id" not in risk_data.columns:
        risk_data["reservation_id"] = [f"RES-{index + 1:05d}" for index in risk_data.index]

    arrival = pd.to_datetime(risk_data["arrival_date"], errors="coerce")
    risk_data["arrival_date"] = arrival.dt.strftime("%Y-%m-%d")
    targets = risk_data.loc[risk_data["risk_level"].eq("높음")].copy()
    targets["reminder_message"] = targets.apply(
        lambda row: (
            f"안녕하세요. {row['arrival_date']} Lisbon City Hotel 예약 확인을 부탁드립니다. "
            "예약을 유지하실 예정인지 확인해 주세요."
        ),
        axis=1,
    )
    columns = [
        "reservation_id",
        "arrival_date",
        "cancellation_probability",
        "risk_level",
        "reminder_message",
    ]
    return targets[columns].sort_values("cancellation_probability", ascending=False)


def build_arrival_demand_summary(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """도착일별 예약량, 고위험 예약, 예상 취소량과 예상 유지량을 계산한다."""
    risk_data = add_risk_levels(predictions, threshold)
    risk_data["arrival_date"] = pd.to_datetime(
        risk_data["arrival_date"], errors="coerce"
    ).dt.date
    risk_data["high_risk"] = risk_data["risk_level"].eq("높음").astype(int)
    risk_data["expected_cancellation"] = risk_data["cancellation_probability"].astype(float)
    risk_data["expected_retained"] = 1 - risk_data["expected_cancellation"]

    summary = (
        risk_data.groupby("arrival_date", as_index=False)
        .agg(
            reservations=("cancellation_probability", "size"),
            high_risk_reservations=("high_risk", "sum"),
            expected_cancellations=("expected_cancellation", "sum"),
            expected_retained_reservations=("expected_retained", "sum"),
        )
        .sort_values("arrival_date")
    )
    summary[["expected_cancellations", "expected_retained_reservations"]] = summary[
        ["expected_cancellations", "expected_retained_reservations"]
    ].round(1)
    return summary
