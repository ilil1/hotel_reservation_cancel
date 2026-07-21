"""호텔 예약 원본 변수로부터 예측에 도움이 되는 파생 특성을 만든다."""

from __future__ import annotations

import numpy as np
import pandas as pd


# 파생 특성을 만드는 데 필요한 원본 열이다.
REQUIRED_SOURCE_COLUMNS = {
    "stays_in_week_nights",
    "stays_in_weekend_nights",
    "adults",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "total_of_special_requests",
    "booking_changes",
    "agent",
    "company",
}

ENGINEERED_FEATURES = {
    "total_nights": "평일과 주말을 합한 총 숙박일 수",
    "total_guests": "성인·어린이·영아를 합한 총 투숙객 수",
    "is_family": "어린이 또는 영아가 포함된 가족 예약 여부",
    "previous_bookings_total": "과거 취소와 정상 예약을 합한 전체 과거 예약 수",
    "previous_cancellation_rate": "과거 전체 예약 중 취소한 비율",
    "has_special_requests": "특별 요청이 하나 이상 있는지 여부",
    "has_booking_changes": "예약 내용을 한 번 이상 변경했는지 여부",
    "is_agent_booking": "여행사·대행사를 통한 예약 여부",
    "is_company_booking": "기업과 관련된 예약 여부",
}


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """정답 열을 사용하지 않고 예약 시점의 정보로 파생 특성을 추가한다."""
    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing source columns for feature engineering: {missing}")

    result = frame.copy()

    # 합계 계산에서만 결측치를 0으로 보고, 원본 열의 결측치는 파이프라인에서 처리한다.
    children = result.get("children", pd.Series(0, index=result.index)).fillna(0)
    babies = result.get("babies", pd.Series(0, index=result.index)).fillna(0)
    adults = result["adults"].fillna(0)
    previous_cancellations = result["previous_cancellations"].fillna(0)
    previous_completed = result["previous_bookings_not_canceled"].fillna(0)

    result["total_nights"] = (
        result["stays_in_week_nights"].fillna(0)
        + result["stays_in_weekend_nights"].fillna(0)
    )
    result["total_guests"] = adults + children + babies
    result["is_family"] = ((children + babies) > 0).astype(int)

    result["previous_bookings_total"] = previous_cancellations + previous_completed
    result["previous_cancellation_rate"] = np.divide(
        previous_cancellations,
        result["previous_bookings_total"],
        out=np.zeros(len(result), dtype=float),
        where=result["previous_bookings_total"].to_numpy() > 0,
    )
    result["has_special_requests"] = (
        result["total_of_special_requests"].fillna(0) > 0
    ).astype(int)
    result["has_booking_changes"] = (result["booking_changes"].fillna(0) > 0).astype(int)
    result["is_agent_booking"] = result["agent"].notna().astype(int)
    result["is_company_booking"] = result["company"].notna().astype(int)

    # agent/company는 연속형 수치가 아니라 익명 식별번호이므로 범주형으로 처리한다.
    result["agent"] = result["agent"].map(
        lambda value: "Missing" if pd.isna(value) else str(int(value))
    )
    result["company"] = result["company"].map(
        lambda value: "Missing" if pd.isna(value) else str(int(value))
    )

    return result
