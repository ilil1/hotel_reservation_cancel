"""City Hotel 데이터의 분포와 상관관계를 분석하고 시각화한다."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/.matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

TARGET = "is_canceled"

# Streamlit에서 이미지를 줄여 표시해도 글자와 선이 선명하도록
# 일반 화면용보다 높은 해상도로 PNG를 저장한다.
CHART_DPI = 260


def _save_figure(fig: plt.Figure, output_without_suffix: Path) -> None:
    """PNG와 확대해도 선명한 SVG 형식을 함께 저장한다."""
    fig.savefig(
        output_without_suffix.with_suffix(".png"),
        dpi=CHART_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        output_without_suffix.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
    )

NUMERIC_FEATURES = [
    "lead_time",
    "adr",
    "total_of_special_requests",
    "previous_cancellations",
    "stays_in_week_nights",
    "stays_in_weekend_nights",
]

CORRELATION_FEATURES = [
    TARGET,
    "lead_time",
    "adr",
    "stays_in_week_nights",
    "stays_in_weekend_nights",
    "adults",
    "children",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_FEATURES = [
    "deposit_type",
    "market_segment",
    "customer_type",
    "arrival_date_month",
    "country",
    "meal",
]


def _save_target_distribution(city: pd.DataFrame, output: Path) -> None:
    """취소와 비취소의 건수 및 비율을 저장하고 막대그래프로 표현한다."""
    counts = city[TARGET].value_counts().reindex([0, 1], fill_value=0)
    result = pd.DataFrame(
        {
            "target": [0, 1],
            "label": ["Not canceled", "Canceled"],
            "count": counts.values,
            "rate": counts.values / len(city),
        }
    )
    result.to_csv(output / "target_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(result["label"], result["count"], color=["#4C78A8", "#E45756"])
    # 막대 위의 건수·비율 표시가 제목과 겹치지 않도록 위쪽 여백을 둔다.
    ax.set_ylim(0, result["count"].max() * 1.22)
    ax.set_title("City Hotel cancellation target distribution", pad=18)
    ax.set_ylabel("Reservations")
    for bar, count, rate in zip(bars, result["count"], result["rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({rate:.1%})",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    _save_figure(fig, output / "target_distribution")
    plt.close(fig)


def _save_numeric_distributions(city: pd.DataFrame, output: Path) -> None:
    """주요 숫자형 특성의 히스토그램을 저장한다."""
    city[NUMERIC_FEATURES].describe().transpose().to_csv(
        output / "numeric_summary.csv"
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feature in zip(axes.flat, NUMERIC_FEATURES):
        values = city[feature].dropna()
        # ADR의 극단값처럼 일부 이상치가 전체 분포를 가리지 않도록 99% 지점까지만 표시한다.
        upper = values.quantile(0.99)
        visible = values[values <= upper]
        ax.hist(visible, bins=35, color="#4C78A8", edgecolor="white")
        ax.set_title(feature)
        ax.set_xlabel("Value")
        ax.set_ylabel("Reservations")
    fig.suptitle("Numeric feature distributions (up to 99th percentile)", fontsize=15)
    fig.tight_layout()
    _save_figure(fig, output / "numeric_distributions")
    plt.close(fig)


def _save_categorical_distributions(city: pd.DataFrame, output: Path) -> None:
    """주요 범주형 특성의 예약 건수를 막대그래프로 저장한다."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    for ax, feature in zip(axes.flat, CATEGORICAL_FEATURES):
        limit = 10 if feature == "country" else 12
        counts = city[feature].fillna("Missing").astype(str).value_counts().head(limit)
        ax.barh(counts.index[::-1], counts.values[::-1], color="#72B7B2")
        ax.set_title(feature)
        ax.set_xlabel("Reservations")
    fig.suptitle("Categorical feature distributions", fontsize=15)
    fig.tight_layout()
    _save_figure(fig, output / "categorical_distributions")
    plt.close(fig)


def _save_correlations(city: pd.DataFrame, output: Path) -> None:
    """숫자 변수 간 Pearson 상관계수와 타겟 상관계수를 저장한다."""
    available = [column for column in CORRELATION_FEATURES if column in city.columns]
    correlation = city[available].corr(numeric_only=True)
    correlation.to_csv(output / "correlation_matrix.csv")

    target_correlation = (
        correlation[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .rename("correlation_with_is_canceled")
    )
    target_correlation.to_csv(output / "target_numeric_correlations.csv")

    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_yticks(np.arange(len(correlation.index)))
    ax.set_xticklabels(correlation.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(correlation.index, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    ax.set_title("Numeric feature correlation matrix")
    fig.tight_layout()
    _save_figure(fig, output / "correlation_heatmap")
    plt.close(fig)


def _cancellation_rate(city: pd.DataFrame, feature: str) -> pd.DataFrame:
    """범주별 예약 수와 평균 취소율을 계산한다."""
    categories = city[feature].astype("object").where(city[feature].notna(), "Missing")
    grouped = (
        city.assign(**{feature: categories})
        .groupby(feature, dropna=False)[TARGET]
        .agg(count="size", cancellation_rate="mean")
        .reset_index()
    )
    grouped.insert(0, "feature", feature)
    grouped = grouped.rename(columns={feature: "category"})
    return grouped


def _save_target_relationships(city: pd.DataFrame, output: Path) -> None:
    """주요 변수의 범주 또는 구간별 취소율을 분석하고 시각화한다."""
    relationship_data = city.copy()
    relationship_data["lead_time_group"] = pd.cut(
        relationship_data["lead_time"],
        bins=[-1, 7, 30, 90, 180, float("inf")],
        labels=["0-7", "8-30", "31-90", "91-180", "181+"],
    )
    relationship_data["special_requests_group"] = (
        relationship_data["total_of_special_requests"].clip(upper=3).astype(str)
    ).replace({"3": "3+"})

    features = [
        "deposit_type",
        "market_segment",
        "customer_type",
        "arrival_date_month",
        "lead_time_group",
        "special_requests_group",
    ]
    all_rates = pd.concat(
        [_cancellation_rate(relationship_data, feature) for feature in features],
        ignore_index=True,
    )
    all_rates.to_csv(output / "target_relationships.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(17, 16))
    for ax, feature in zip(axes.flat, features):
        rates = all_rates.loc[all_rates["feature"].eq(feature)].copy()
        if feature == "market_segment":
            rates = rates.sort_values("count", ascending=False).head(8)
        ax.barh(
            rates["category"].astype(str)[::-1],
            rates["cancellation_rate"][::-1],
            color="#F58518",
        )
        ax.set_title(f"Cancellation rate by {feature}")
        ax.set_xlabel("Cancellation rate")
        ax.set_xlim(0, 1)
    fig.suptitle("Feature relationships with cancellation target", fontsize=15)
    fig.tight_layout()
    _save_figure(fig, output / "target_relationships")
    plt.close(fig)


def generate_eda(data_path: Path, output: Path) -> None:
    """City Hotel EDA 표와 시각화 파일을 모두 생성한다."""
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    city = frame.loc[frame["hotel"].eq("City Hotel")].copy()
    if city.empty:
        raise ValueError("No rows where hotel == 'City Hotel'.")

    _save_target_distribution(city, output)
    _save_numeric_distributions(city, output)
    _save_categorical_distributions(city, output)
    _save_correlations(city, output)
    _save_target_relationships(city, output)
    print(f"EDA 결과: {output}")
