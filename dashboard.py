"""Lisbon City Hotel 취소 예측 결과를 보여주는 Streamlit 웹 대시보드.

학습 과정에서 생성된 JSON, CSV, PNG 결과를 읽어 성능 지표,
모델 비교, 혼동행렬, PR 곡선과 변수 중요도를 웹 화면에 표시한다.
대시보드는 모델을 다시 학습하지 않으므로 빠르게 실행된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULT_DIR = Path("outputs/model")

# 원본 변수 이름을 대시보드에서 이해하기 쉬운 한국어와 설명으로 바꾼다.
FEATURE_INFO = {
    "lead_time": ("예약 리드타임", "예약일부터 도착 예정일까지 남은 일수"),
    "arrival_date_year": ("도착 연도", "고객의 도착 예정 연도"),
    "arrival_date_month": ("도착 월", "고객의 도착 예정 월"),
    "arrival_date_week_number": ("도착 주차", "연중 몇 번째 주에 도착하는지 나타내는 번호"),
    "arrival_date_day_of_month": ("도착 일자", "도착 예정일의 일(day) 부분"),
    "stays_in_weekend_nights": ("주말 숙박일 수", "토요일과 일요일에 숙박하는 일수"),
    "stays_in_week_nights": ("평일 숙박일 수", "월요일부터 금요일까지 숙박하는 일수"),
    "adults": ("성인 수", "예약에 포함된 성인 인원"),
    "children": ("어린이 수", "예약에 포함된 어린이 인원"),
    "babies": ("영아 수", "예약에 포함된 영아 인원"),
    "meal": ("식사 유형", "예약에 포함된 식사 패키지 유형"),
    "country": ("고객 국가", "고객의 출신 국가 코드"),
    "market_segment": ("시장 세그먼트", "온라인 여행사, 단체 등 예약 시장 구분"),
    "distribution_channel": ("판매 채널", "호텔에 예약이 전달된 경로"),
    "is_repeated_guest": ("재방문 고객 여부", "이전에 방문한 고객이면 1, 아니면 0"),
    "previous_cancellations": ("과거 취소 횟수", "해당 고객의 이전 예약 취소 횟수"),
    "previous_bookings_not_canceled": ("과거 정상 예약 수", "취소하지 않고 완료한 이전 예약 수"),
    "reserved_room_type": ("예약 객실 유형", "고객이 처음 예약한 익명화된 객실 등급"),
    "booking_changes": ("예약 변경 횟수", "예약 후 숙박 조건 등을 변경한 횟수"),
    "deposit_type": ("보증금 유형", "보증금 없음, 환불 불가, 환불 가능 등의 조건"),
    "agent": ("예약 대행사", "예약을 처리한 여행사의 익명 식별번호"),
    "company": ("회사", "예약과 관련된 회사의 익명 식별번호"),
    "days_in_waiting_list": ("대기 일수", "예약 확정 전 대기 명단에 있었던 일수"),
    "customer_type": ("고객 유형", "개별, 단체 등 예약 고객 유형"),
    "adr": ("평균 일일 객실요금", "Average Daily Rate: 하루 평균 객실요금"),
    "required_car_parking_spaces": ("요청 주차 공간 수", "고객이 요청한 주차 공간 개수"),
    "total_of_special_requests": ("특별 요청 수", "침대·고층 객실 등 고객이 남긴 특별 요청 개수"),
}

CATEGORY_LABELS = {
    "Non Refund": "환불 불가",
    "No Deposit": "보증금 없음",
    "Refundable": "환불 가능",
    "Online TA": "온라인 여행사",
    "Offline TA/TO": "오프라인 여행사·여행 운영사",
    "Groups": "단체",
    "Direct": "직접 예약",
    "Corporate": "기업",
    "Transient": "일반 개별 고객",
    "Transient-Party": "개별 단체 고객",
    "Contract": "계약 고객",
    "Group": "그룹 고객",
    "PRT": "포르투갈",
    "FRA": "프랑스",
    "GBR": "영국",
    "DEU": "독일",
    "ESP": "스페인",
    "ITA": "이탈리아",
    "BRA": "브라질",
    "NLD": "네덜란드",
}


def describe_feature(encoded_name: str) -> tuple[str, str, str]:
    """전처리된 변수명을 영문명, 한국어명, 설명으로 변환한다."""
    english_name = encoded_name.replace("num__", "", 1).replace("cat__", "", 1)

    # 범주형 변수는 deposit_type_Non Refund처럼 원본 변수와 범주가 합쳐져 있다.
    for base_name in sorted(FEATURE_INFO, key=len, reverse=True):
        if english_name == base_name or english_name.startswith(base_name + "_"):
            korean_name, description = FEATURE_INFO[base_name]
            category = english_name[len(base_name):].lstrip("_")
            if category:
                translated_category = CATEGORY_LABELS.get(category, category)
                korean_name = f"{korean_name}: {translated_category}"
                description = f"{description} — 값이 '{category}'인 경우"
            return english_name, korean_name, description

    return english_name, english_name, "변수 설명이 아직 등록되지 않았습니다."

# 브라우저 탭 제목, 아이콘과 넓은 화면 레이아웃을 지정한다.
st.set_page_config(page_title="Lisbon Hotel 취소 예측", page_icon="🏨", layout="wide")

# 지표 카드를 구분하기 위한 최소한의 화면 스타일이다.
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #f6f8fb; border: 1px solid #e6e9ef;
        padding: 16px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏨 Lisbon City Hotel 예약 취소 예측")
st.caption("2015-07-01 ~ 2017-08-31 · City Hotel 79,330건 · 시간순 검증")

# 결과 파일이 없을 때 빈 화면이나 추적 오류 대신 실행 방법을 안내한다.
required = ["metrics.json", "metadata.json", "model_comparison.csv", "feature_importance.csv"]
missing = [name for name in required if not (RESULT_DIR / name).exists()]
if missing:
    st.error("학습 결과가 없습니다. 먼저 `python src/train.py`를 실행해 주세요.")
    st.stop()

# 학습 결과 파일을 한 번만 읽어 이후 화면 구성에 재사용한다.
metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
metadata = json.loads((RESULT_DIR / "metadata.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(RESULT_DIR / "model_comparison.csv")
importance = pd.read_csv(RESULT_DIR / "feature_importance.csv")
test = metrics["test"]

# 최종 판단에는 모델 선택에 사용하지 않은 테스트 지표를 표시한다.
st.subheader("테스트 성능")
cols = st.columns(6)
values = [
    ("PR-AUC", test["pr_auc"]),
    ("ROC-AUC", test["roc_auc"]),
    ("Precision", test["precision"]),
    ("Recall", test["recall"]),
    ("F1", test["f1"]),
    ("판단 임계값", test["threshold"]),
]
for col, (label, value) in zip(cols, values):
    col.metric(label, f"{value:.3f}")

st.info(
    f"선정 모델은 {metrics['selected_model'].replace('_', ' ').title()}입니다. "
    f"테스트 예약 {test['rows']:,}건에서 실제 취소의 {test['recall']:.1%}를 탐지했고, "
    f"취소 예측 중 {test['precision']:.1%}가 실제 취소였습니다."
)

# 넓은 화면에서 두 진단 그래프를 나란히 배치한다.
left, right = st.columns(2)
with left:
    st.subheader("혼동행렬")
    matrix_path = RESULT_DIR / "confusion_matrix.png"
    if matrix_path.exists():
        st.image(str(matrix_path), width="stretch")
with right:
    st.subheader("Precision–Recall 곡선")
    curve_path = RESULT_DIR / "precision_recall_curve.png"
    if curve_path.exists():
        st.image(str(curve_path), width="stretch")

# 내부 영문 열 이름을 사용자가 읽기 쉬운 이름으로 바꾼다.
st.subheader("후보 모델 비교")
display_comparison = comparison.rename(
    columns={
        "model": "모델", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
        "precision": "Precision", "recall": "Recall", "f1": "F1",
        "threshold": "임계값", "rows": "검증 건수", "cancellation_rate": "취소율",
    }
)
st.dataframe(
    display_comparison.style.format({
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Precision": "{:.3f}",
        "Recall": "{:.3f}", "F1": "{:.3f}", "임계값": "{:.3f}", "취소율": "{:.1%}",
    }),
    width="stretch",
    hide_index=True,
)

# 사용자가 화면에 표시할 변수 개수를 조절할 수 있다.
st.subheader("주요 예측 변수")
top_n = st.slider("표시할 변수 수", min_value=5, max_value=30, value=15, step=5)
top = importance.head(top_n).copy()
descriptions = top["feature"].map(describe_feature)
top[["영문 변수", "한국어 변수", "설명"]] = pd.DataFrame(
    descriptions.tolist(), index=top.index
)
top["그래프 표시명"] = top.apply(
    lambda row: f"{row['한국어 변수']} ({row['영문 변수']})", axis=1
)
top["중요도 비율"] = top["importance"] * 100

st.bar_chart(
    top.set_index("그래프 표시명")["importance"],
    horizontal=True,
    height=max(500, top_n * 32),
)
st.caption("중요도는 예측에 기여한 크기이며, 취소 확률을 높이거나 낮추는 방향을 뜻하지는 않습니다.")

st.markdown("#### 예측 변수 설명")
st.dataframe(
    top[["영문 변수", "한국어 변수", "설명", "중요도 비율"]].style.format(
        {"중요도 비율": "{:.2f}%"}
    ),
    width="stretch",
    hide_index=True,
)
st.info(
    "예: '보증금 유형: 환불 불가'의 중요도가 높다는 것은 모델이 이 정보를 많이 "
    "사용했다는 뜻입니다. 환불 불가 조건이 취소를 늘리거나 줄인다는 방향까지 의미하지는 않습니다."
)

with st.expander("데이터와 모델 정보"):
    a, b, c, d = st.columns(4)
    a.metric("전체 City Hotel 예약", f"{metadata['data_rows']:,}건")
    b.metric("테스트 예약", f"{test['rows']:,}건")
    c.metric("테스트 취소율", f"{test['cancellation_rate']:.1%}")
    d.metric("입력 변수", f"{len(metadata['feature_columns'])}개")
    st.write("데이터 기간:", metadata["date_min"], "~", metadata["date_max"])
    st.write("제거한 누수 변수:", ", ".join(metadata["leakage_columns_removed"]))

st.divider()
st.caption("연구·의사결정 지원용 모델입니다. 고객 자동 거절이나 차별적 조치에 사용하지 마세요.")
