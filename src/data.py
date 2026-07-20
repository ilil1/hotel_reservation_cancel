"""호텔 예약 데이터 불러오기와 시간순 분할을 담당한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "is_canceled"

# 예약 취소 결과가 확정된 뒤 생기거나 바뀔 수 있어 학습에서 제외한다.
LEAKAGE_COLUMNS = {
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
}


def load_city_hotel(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """원본 CSV를 검사하고 Lisbon City Hotel 데이터만 반환한다."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found: {path}. Download hotel_bookings.csv from Kaggle; see README.md."
        )

    frame = pd.read_csv(path)
    required = {
        "hotel",
        TARGET,
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    city = frame.loc[frame["hotel"].eq("City Hotel")].copy()
    if city.empty:
        raise ValueError("No rows where hotel == 'City Hotel'.")

    # 연/월/일을 실제 날짜로 합쳐 시간순 분할 기준으로 사용한다.
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
    columns_to_drop = [c for c in LEAKAGE_COLUMNS | {"hotel"} if c in city.columns]
    x = city.drop(columns=columns_to_drop)
    return x, y


def chronological_split(x: pd.DataFrame, y: pd.Series):
    """과거 70%, 중간 15%, 최신 15%로 학습/검증/테스트를 나눈다."""
    train_end = int(len(x) * 0.70)
    valid_end = int(len(x) * 0.85)

    if train_end < 20 or valid_end == train_end or valid_end == len(x):
        raise ValueError("At least 30 chronologically ordered City Hotel rows are required.")

    return (
        x.iloc[:train_end].copy(),
        y.iloc[:train_end].copy(),
        x.iloc[train_end:valid_end].copy(),
        y.iloc[train_end:valid_end].copy(),
        x.iloc[valid_end:].copy(),
        y.iloc[valid_end:].copy(),
    )

