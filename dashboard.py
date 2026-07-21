"""Lisbon City Hotel 취소 예측 결과를 보여주는 Streamlit 웹 대시보드.

학습 과정에서 생성된 JSON, CSV, PNG 결과를 읽어 성능 지표,
모델 비교, 혼동행렬, 분류 보고서와 변수 중요도를 웹 화면에 표시한다.
대시보드는 모델을 다시 학습하지 않으므로 빠르게 실행된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULT_DIR = Path("outputs/model")
EDA_DIR = Path("outputs/eda")


def show_centered_image(path: Path, width: int) -> None:
    """그래프가 화면 전체를 과도하게 채우지 않도록 중앙에 제한된 크기로 표시한다."""
    if path.exists():
        with st.container(horizontal_alignment="center"):
            # SVG는 화면을 확대하거나 줄여도 글자와 선이 흐려지지 않는다.
            image = path.read_text(encoding="utf-8") if path.suffix == ".svg" else path
            st.image(image, width=width)

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

# 모델 결과를 보기 전에 City Hotel 원본 데이터의 분포와 관계를 먼저 확인한다.
st.subheader("탐색적 데이터 분석 (EDA)")
eda_required = [
    "target_distribution.svg",
    "numeric_distributions.svg",
    "categorical_distributions.svg",
    "correlation_heatmap.svg",
    "target_relationships.svg",
]
eda_missing = [name for name in eda_required if not (EDA_DIR / name).exists()]
if eda_missing:
    st.warning("EDA 결과가 없습니다. `python src/train.py`를 다시 실행해 주세요.")
else:
    target_tab, numeric_tab, category_tab, corr_tab, relation_tab = st.tabs(
        ["타겟 분포", "숫자 특성", "범주 특성", "상관관계", "특성과 취소율"]
    )
    with target_tab:
        show_centered_image(EDA_DIR / "target_distribution.svg", width=620)
        target_distribution = pd.read_csv(EDA_DIR / "target_distribution.csv")
        canceled = target_distribution.loc[target_distribution["target"].eq(1)].iloc[0]
        st.caption(
            f"City Hotel 전체 예약 중 취소는 {int(canceled['count']):,}건이며, "
            f"취소율은 {canceled['rate']:.1%}입니다."
        )
    with numeric_tab:
        show_centered_image(EDA_DIR / "numeric_distributions.svg", width=900)
        st.caption("극단값 때문에 전체 모양이 가려지지 않도록 각 숫자 변수의 99백분위수까지 표시합니다.")
    with category_tab:
        show_centered_image(EDA_DIR / "categorical_distributions.svg", width=900)
        st.caption("보증금, 예약 시장, 고객 유형, 도착 월, 주요 국가와 식사 유형의 예약 건수입니다.")
    with corr_tab:
        show_centered_image(EDA_DIR / "correlation_heatmap.svg", width=760)
        target_corr = pd.read_csv(EDA_DIR / "target_numeric_correlations.csv")
        target_corr.columns = ["숫자 변수", "취소 여부와 상관계수"]
        st.dataframe(
            target_corr.style.format({"취소 여부와 상관계수": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption("상관계수는 -1~1이며 절댓값이 클수록 선형 관계가 강합니다. 상관관계는 인과관계를 뜻하지 않습니다.")
    with relation_tab:
        show_centered_image(EDA_DIR / "target_relationships.svg", width=900)
        st.caption("보증금 유형, 예약 시장, 고객 유형, 도착 월, 리드타임 구간과 특별 요청 수별 취소율입니다.")

# 검증 데이터의 성능을 비교해 최종 모델을 선택한 과정을 먼저 보여준다.
st.subheader("후보 모델 비교")
display_comparison = comparison.rename(
    columns={
        "model": "모델",
        "accuracy": "Accuracy",
        "precision": "Precision", "recall": "Recall", "f1": "F1",
        "rows": "검증 건수", "cancellation_rate": "취소율",
    }
)
st.dataframe(
    display_comparison.style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}",
        "Recall": "{:.3f}", "F1": "{:.3f}", "취소율": "{:.1%}",
    }),
    width="stretch",
    hide_index=True,
)

comparison_plot = RESULT_DIR / "model_comparison.svg"
show_centered_image(comparison_plot, width=760)

# 최종 판단에는 모델 선택에 사용하지 않은 테스트 지표를 표시한다.
st.subheader("테스트 성능")
metric_items = [
    ("Accuracy", "accuracy", "전체 예약 중 취소·정상 여부를 정확히 맞힌 비율"),
    ("Precision", "precision", "취소라고 예측한 예약 중 실제로 취소된 비율"),
    ("Recall", "recall", "실제 취소 예약 중 모델이 취소로 찾아낸 비율"),
    ("F1", "f1", "Precision과 Recall의 균형을 나타내는 조화 평균"),
]
for column, (label, key, description) in zip(st.columns(4), metric_items):
    with column.container(border=True):
        st.metric(label, f"{test[key]:.3f}")
        st.caption(description)

st.info(
    f"**지표 해석:** 네 지표는 일반적으로 높을수록 좋지만, 하나만 보지 말고 함께 확인해야 합니다. "
    f"선정 모델은 **{metrics['selected_model'].replace('_', ' ').title()}**이며, "
    f"취소라고 예측한 결과의 **{test['precision']:.1%}**가 실제 취소였습니다. "
    f"반면 실제 취소 예약의 **{test['recall']:.1%}**를 찾았고 "
    f"약 **{1 - test['recall']:.1%}**는 놓쳤습니다. "
    "취소 예약을 놓치지 않는 것이 중요하다면 Recall을 중점적으로 봐야 합니다."
)

# 혼동행렬에서 정상·취소 예약별 탐지율을 계산해 테스트 성능을 요약한다.
test_matrix_path = RESULT_DIR / "confusion_matrix.csv"
if test_matrix_path.exists():
    test_matrix = pd.read_csv(test_matrix_path, index_col=0)
    test_tn = int(test_matrix.loc["actual_not_canceled", "predicted_not_canceled"])
    test_fp = int(test_matrix.loc["actual_not_canceled", "predicted_canceled"])
    test_fn = int(test_matrix.loc["actual_canceled", "predicted_not_canceled"])
    test_tp = int(test_matrix.loc["actual_canceled", "predicted_canceled"])
    normal_detection_rate = test_tn / (test_tn + test_fp)
    cancellation_detection_rate = test_tp / (test_tp + test_fn)
    st.warning(
        f"정상 예약은 **{normal_detection_rate:.1%}**를 정확히 찾았지만, "
        f"취소 예약은 **{cancellation_detection_rate:.1%}**만 찾았습니다. "
        f"따라서 모델은 정상 예약은 비교적 잘 구분하지만, "
        f"실제 취소 예약의 **{1 - cancellation_detection_rate:.1%}**를 놓치고 있습니다."
    )

st.subheader("혼동행렬")
matrix_path = RESULT_DIR / "confusion_matrix.svg"
matrix_csv_path = RESULT_DIR / "confusion_matrix.csv"

matrix_left, matrix_right = st.columns([1.15, 1], vertical_alignment="center")
with matrix_left:
    show_centered_image(matrix_path, width=520)

with matrix_right:
    if matrix_csv_path.exists():
        matrix = pd.read_csv(matrix_csv_path, index_col=0)
        true_negative = int(matrix.loc["actual_not_canceled", "predicted_not_canceled"])
        false_positive = int(matrix.loc["actual_not_canceled", "predicted_canceled"])
        false_negative = int(matrix.loc["actual_canceled", "predicted_not_canceled"])
        true_positive = int(matrix.loc["actual_canceled", "predicted_canceled"])

        with st.container(border=True):
            st.markdown("#### 결과 해석")
            st.markdown(
                f"- **{true_negative:,}건**: 정상 예약을 정상으로 정확히 예측\n"
                f"- **{true_positive:,}건**: 취소 예약을 취소로 정확히 예측\n"
                f"- **{false_positive:,}건**: 정상 예약을 취소로 잘못 예측\n"
                f"- **{false_negative:,}건**: 취소 예약을 정상으로 잘못 예측"
            )
            st.caption(
                f"즉, 실제 취소 {true_positive + false_negative:,}건 중 "
                f"{true_positive:,}건을 찾았고 {false_negative:,}건을 놓쳤습니다."
            )

st.subheader("분류 성능 상세 (Classification Report)")
report_path = RESULT_DIR / "classification_report.csv"
if report_path.exists() and matrix_csv_path.exists():
    report_table = pd.read_csv(report_path, index_col=0).reset_index().rename(
        columns={
            "index": "구분",
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "support": "건수",
        }
    )
    report_table["구분"] = report_table["구분"].replace(
        {"Not canceled": "정상 예약", "Canceled": "취소 예약"}
    )
    st.dataframe(
        report_table.style.format(
            {
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-Score": "{:.3f}",
                "건수": "{:.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    class_scores = report_table.loc[
        report_table["구분"].isin(["정상 예약", "취소 예약"]),
        ["구분", "Precision", "Recall", "F1-Score"],
    ].set_index("구분")
    st.bar_chart(class_scores, height=380)
    st.caption(
        "정상 예약과 취소 예약 각각의 Precision, Recall, F1-Score와 데이터 건수입니다."
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
    total_rows = int(metadata["data_rows"])
    train_rows = int(total_rows * 0.70)
    validation_end = int(total_rows * 0.85)
    validation_rows = validation_end - train_rows

    a, b, c, d = st.columns(4)
    a.metric("전체 City Hotel 예약", f"{total_rows:,}건")
    b.metric("테스트 예약", f"{test['rows']:,}건")
    c.metric("테스트 취소율", f"{test['cancellation_rate']:.1%}")
    d.metric("입력 변수", f"{len(metadata['feature_columns'])}개")

    st.markdown(
        f"- **전체 City Hotel 예약 {total_rows:,}건**: 원본 데이터에서 City Hotel에 해당하는 전체 예약입니다.\n"
        f"- **테스트 예약 {test['rows']:,}건**: 모델 학습에 사용하지 않고 최종 성능 평가용으로 따로 보관한 예약입니다.\n"
        f"- **테스트 취소율 {test['cancellation_rate']:.1%}**: 테스트 예약 중 실제로 취소된 예약의 비율입니다.\n"
        f"- **입력 변수 {len(metadata['feature_columns'])}개**: 리드타임, 객실요금, 특별 요청 수 등 모델이 취소 여부를 예측할 때 사용하는 입력 항목입니다."
    )

    st.markdown("#### 데이터 분할")
    st.code(
        f"City Hotel 전체 {total_rows:,}건\n"
        f"├─ 학습 데이터 {train_rows:,}건 (70%)  모델 학습\n"
        f"├─ 검증 데이터 {validation_rows:,}건 (15%)  후보 모델 비교\n"
        f"└─ 테스트 데이터 {test['rows']:,}건 (15%)  최종 성능 평가",
        language=None,
    )
    st.caption("도착일 기준 시간순으로 나누어 과거 데이터로 학습하고 이후 데이터로 평가합니다.")
    st.write("데이터 기간:", metadata["date_min"], "~", metadata["date_max"])
    st.write("제거한 누수 변수:", ", ".join(metadata["leakage_columns_removed"]))
    st.markdown(
        "- **`assigned_room_type`**: 예약 후 최종적으로 배정된 객실 유형\n"
        "- **`reservation_status`**: Canceled, Check-Out 등 예약의 최종 상태\n"
        "- **`reservation_status_date`**: 최종 예약 상태가 확정·변경된 날짜"
    )
    st.info(
        "이 변수들은 예약 시점에 알 수 없거나 취소 결과를 직접 알려주는 사후 정보입니다. "
        "모델이 이를 학습하면 실제 예측 능력 대신 정답을 미리 보게 되어 성능이 부풀려집니다. "
        "따라서 데이터 누수를 방지하기 위해 전처리 단계에서 제거했습니다."
    )

st.divider()
st.caption("연구·의사결정 지원용 모델입니다. 고객 자동 거절이나 차별적 조치에 사용하지 마세요.")
