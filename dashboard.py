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


def show_model_comparison(comparison_data: pd.DataFrame, plot_path: Path) -> None:
    """두 후보 모델의 검증 성능 표와 그래프를 같은 형식으로 표시한다."""
    display_comparison = comparison_data.rename(
        columns={
            "model": "모델",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "rows": "검증 건수",
            "cancellation_rate": "취소율",
        }
    )
    st.dataframe(
        display_comparison.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "취소율": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    show_centered_image(plot_path, width=760)


def show_test_evaluation(
    result_row: pd.Series,
    selected_model: str,
    matrix_csv_path: Path,
    matrix_plot_path: Path,
    report_path: Path,
) -> None:
    """한 단계에서 선택된 모델의 테스트 성능과 상세 평가를 표시한다."""
    st.markdown("#### 선택 모델의 최종 테스트 성능")
    st.caption("후보 모델 선택에 사용하지 않은 테스트 데이터 11,900건의 결과입니다.")
    metric_items = [
        ("Accuracy", "전체 예약 중 취소·정상 여부를 정확히 맞힌 비율"),
        ("Precision", "취소라고 예측한 예약 중 실제로 취소된 비율"),
        ("Recall", "실제 취소 예약 중 모델이 취소로 찾아낸 비율"),
        ("F1", "Precision과 Recall의 균형을 나타내는 조화 평균"),
    ]
    for column, (label, description) in zip(st.columns(4), metric_items):
        with column.container(border=True):
            st.metric(label, f"{result_row[label]:.1%}")
            st.caption(description)

    st.info(
        f"선정 모델은 **{selected_model}** 입니다. 취소라고 예측한 결과의 "
        f"**{result_row['Precision']:.1%}** 가 실제 취소였으며, 실제 취소 예약의 "
        f"**{result_row['Recall']:.1%}** 를 찾고 **{1 - result_row['Recall']:.1%}** 를 놓쳤습니다."
    )

    if not matrix_csv_path.exists():
        return

    matrix = pd.read_csv(matrix_csv_path, index_col=0)
    true_negative = int(matrix.loc["actual_not_canceled", "predicted_not_canceled"])
    false_positive = int(matrix.loc["actual_not_canceled", "predicted_canceled"])
    false_negative = int(matrix.loc["actual_canceled", "predicted_not_canceled"])
    true_positive = int(matrix.loc["actual_canceled", "predicted_canceled"])
    actual_normal = true_negative + false_positive
    actual_canceled = true_positive + false_negative
    predicted_canceled = true_positive + false_positive
    total_predictions = actual_normal + actual_canceled
    correct_predictions = true_negative + true_positive
    overall_accuracy = correct_predictions / total_predictions
    normal_detection_rate = true_negative / (true_negative + false_positive)
    cancellation_detection_rate = true_positive / (true_positive + false_negative)
    false_alarm_rate = false_positive / actual_normal
    missed_cancellation_rate = false_negative / actual_canceled
    cancellation_precision = true_positive / predicted_canceled

    st.warning(
        f"정상 예약은 **{normal_detection_rate:.1%}** 를 정확히 찾았고, "
        f"취소 예약은 **{cancellation_detection_rate:.1%}** 를 찾았습니다. "
        f"실제 취소 예약의 **{1 - cancellation_detection_rate:.1%}** 를 놓쳤습니다."
    )

    st.markdown("#### 혼동행렬")
    matrix_left, matrix_right = st.columns([1.15, 1], vertical_alignment="center")
    with matrix_left:
        show_centered_image(matrix_plot_path, width=520)
    with matrix_right:
        with st.container(border=True):
            st.markdown("##### 결과 해석")
            st.markdown(
                f"- **전체 결과:** 테스트 예약 {total_predictions:,}건 중 "
                f"{correct_predictions:,}건을 맞혀 정확도는 **{overall_accuracy:.1%}** 입니다.\n"
                f"- **정상 예약 탐지:** 실제 정상 예약 {actual_normal:,}건 중 "
                f"{true_negative:,}건을 정상으로 맞혔습니다. 정상 예약 탐지율은 "
                f"**{normal_detection_rate:.1%}** 입니다.\n"
                f"- **정상 예약 오탐:** 정상 예약 {false_positive:,}건을 취소 위험으로 잘못 판단했습니다. "
                f"정상 예약 중 오탐 비율은 **{false_alarm_rate:.1%}** 입니다.\n"
                f"- **취소 예약 탐지:** 실제 취소 예약 {actual_canceled:,}건 중 "
                f"{true_positive:,}건을 찾아 취소 Recall은 **{cancellation_detection_rate:.1%}** 입니다.\n"
                f"- **취소 예약 미탐:** 취소 예약 {false_negative:,}건을 정상으로 잘못 판단했습니다. "
                f"실제 취소 중 놓친 비율은 **{missed_cancellation_rate:.1%}** 입니다.\n"
                f"- **취소 예측 신뢰도:** 취소라고 예측한 {predicted_canceled:,}건 중 "
                f"실제 취소는 {true_positive:,}건으로, 취소 Precision은 "
                f"**{cancellation_precision:.1%}** 입니다."
            )
            if cancellation_detection_rate < normal_detection_rate:
                st.caption(
                    "이 모델은 정상 예약을 구분하는 능력보다 실제 취소 예약을 찾아내는 능력이 낮습니다. "
                    "호텔이 취소 누락을 줄이려면 Recall을 높이는 방향의 모델 선택이나 임계값 조정이 필요합니다."
                )
            else:
                st.caption(
                    "이 모델은 정상 예약보다 실제 취소 예약을 더 높은 비율로 찾아냅니다. "
                    "다만 정상 예약을 취소 위험으로 잘못 분류하는 오탐 건수도 함께 확인해야 합니다."
                )

    if not report_path.exists():
        return

    st.markdown("#### 분류 성능 상세 (Classification Report)")
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
    report_table["건수"] = report_table.apply(
        lambda row: "-" if row["구분"] == "accuracy" else f"{int(row['건수']):,}",
        axis=1,
    )
    st.dataframe(
        report_table.style.format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    class_scores = report_table.loc[
        report_table["구분"].isin(["정상 예약", "취소 예약"]),
        ["구분", "Precision", "Recall", "F1-Score"],
    ].set_index("구분")
    st.markdown("#### 정상·취소 예약 성능 비교")
    score_columns = st.columns(2)
    for column, class_name in zip(score_columns, ["정상 예약", "취소 예약"]):
        scores = class_scores.loc[class_name]
        with column.container(border=True):
            st.markdown(f"##### {class_name}")
            st.progress(float(scores["Precision"]), text=f"Precision — {scores['Precision']:.1%}")
            st.progress(float(scores["Recall"]), text=f"Recall — {scores['Recall']:.1%}")
            st.progress(float(scores["F1-Score"]), text=f"F1-Score — {scores['F1-Score']:.1%}")


def show_feature_importance(
    importance_data: pd.DataFrame,
    stage_key: str,
) -> None:
    """한 단계에서 선택된 모델의 변수 중요도와 변수 설명을 표시한다."""
    st.markdown("#### Feature Importance (변수 중요도)")
    top_n = st.slider(
        "표시할 변수 수",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
        key=f"feature_count_{stage_key}",
    )
    top = importance_data.head(top_n).copy()
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
    st.caption(
        "중요도는 예측에 기여한 크기이며, 취소 확률을 높이거나 낮추는 방향을 뜻하지는 않습니다."
    )
    st.markdown("#### 예측 변수 설명")
    st.dataframe(
        top[["영문 변수", "한국어 변수", "설명", "중요도 비율"]].style.format(
            {"중요도 비율": "{:.2f}%"}
        ),
        width="stretch",
        hide_index=True,
    )


def show_data_model_info(
    metadata_data: dict,
    result_row: pd.Series,
    stage_label: str,
) -> None:
    """한 단계의 데이터 분할과 모델 입력 변수 정보를 표시한다."""
    with st.expander(f"데이터와 모델 정보 — {stage_label}"):
        total_rows = int(metadata_data["data_rows"])
        train_rows = int(total_rows * 0.70)
        validation_end = int(total_rows * 0.85)
        validation_rows = validation_end - train_rows
        model_feature_count = len(
            [
                column
                for column in metadata_data["feature_columns"]
                if column != "arrival_date"
            ]
        )

        a, b, c, d = st.columns(4)
        a.metric("전체 City Hotel 예약", f"{total_rows:,}건")
        b.metric("테스트 예약", f"{int(result_row['테스트 건수']):,}건")
        c.metric("테스트 취소율", f"{result_row['취소율']:.1%}")
        d.metric("모델 입력 변수", f"{model_feature_count}개")

        st.markdown("#### 데이터 분할")
        st.code(
            f"City Hotel 전체 {total_rows:,}건\n"
            f"├─ 학습 데이터 {train_rows:,}건 (70%)  모델 학습\n"
            f"├─ 검증 데이터 {validation_rows:,}건 (15%)  후보 모델 비교\n"
            f"└─ 테스트 데이터 {int(result_row['테스트 건수']):,}건 (15%)  최종 성능 평가",
            language=None,
        )
        st.caption("도착일 기준 시간순으로 나누어 과거 데이터로 학습하고 이후 데이터로 평가합니다.")
        st.write("선택 모델:", result_row["선택 모델"])
        st.write("데이터 기간:", metadata_data["date_min"], "~", metadata_data["date_max"])
        st.write("제거한 누수 변수:", ", ".join(metadata_data["leakage_columns_removed"]))
        st.info(
            "누수 변수는 예약 시점에 알 수 없거나 취소 결과를 직접 알려주는 사후 정보이므로 "
            "모델 성능이 부풀려지는 것을 막기 위해 제거했습니다."
        )


def show_parameter_settings(title: str, parameters: dict) -> None:
    """하이퍼파라미터 이름과 설정값을 읽기 쉬운 표로 표시한다."""
    parameter_labels = {
        "model__C": "규제 강도 역수 (C)",
        "C": "규제 강도 역수 (C)",
        "model__class_weight": "클래스 가중치",
        "class_weight": "클래스 가중치",
        "model__penalty": "규제 방식",
        "penalty": "규제 방식",
        "model__solver": "최적화 알고리즘",
        "solver": "최적화 알고리즘",
        "model__n_estimators": "트리 개수",
        "n_estimators": "트리 개수",
        "model__max_depth": "최대 트리 깊이",
        "max_depth": "최대 트리 깊이",
        "model__min_samples_leaf": "리프 최소 데이터 수",
        "min_samples_leaf": "리프 최소 데이터 수",
        "model__max_features": "분할 시 최대 변수 수",
        "max_features": "분할 시 최대 변수 수",
    }
    rows = []
    for name, value in parameters.items():
        display_value = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        rows.append(
            {
                "하이퍼파라미터": parameter_labels.get(name, name),
                "영문명": name.replace("model__", ""),
                "설정값": display_value,
            }
        )
    st.markdown(f"#### {title}")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def show_tuning_method_result(
    method_title: str,
    result_row: pd.Series,
    baseline_row: pd.Series,
    method_metadata: dict,
    importance_data: pd.DataFrame,
    result_suffix: str,
    stage_key: str,
) -> None:
    """하나의 튜닝 방법에 대한 설정, 성능과 상세 평가를 같은 형식으로 표시한다."""
    st.markdown(f"#### {method_title}로 찾은 최적 설정")
    st.caption(
        "Feature Engineering 적용 후 모델을 기준으로 학습 데이터 내부에서 "
        "TimeSeriesSplit 3분할을 사용해 시간 순서를 유지했습니다."
    )
    model_column, feature_column, row_column, rate_column = st.columns([1.6, 1, 1, 1])
    model_column.metric("모델", "Logistic Regression", border=True)
    feature_column.metric("입력 변수", "36개", border=True)
    row_column.metric(
        "테스트 예약", f"{int(result_row['테스트 건수']):,}건", border=True
    )
    rate_column.metric("테스트 취소율", f"{result_row['취소율']:.1%}", border=True)

    method_column, cv_column, iteration_column = st.columns(3)
    method_column.metric("탐색 방법", method_title, border=True)
    cv_column.metric("교차검증", "시간순 3분할", border=True)
    iteration_column.metric("탐색 횟수", f"{method_metadata['trial_count']}개", border=True)
    selection_rule_korean = (
        "교차검증 평균 Recall 80% 이상인 설정 중 평균 F1이 가장 높은 설정을 선택했습니다."
        if method_metadata["best_cv_metrics"]["recall"] >= 0.80
        else "교차검증 평균 Recall 80%를 충족한 설정이 없어 평균 Recall이 가장 높은 설정을 선택했습니다."
    )
    st.info(f"선택 기준: {selection_rule_korean}")
    show_parameter_settings("탐색한 파라미터 범위", method_metadata["parameter_space"])
    show_parameter_settings("선택된 최적 하이퍼파라미터", method_metadata["best_params"])

    cv_metrics = method_metadata["best_cv_metrics"]
    st.markdown("#### 시간순 교차검증 평균 성능")
    cv_columns = st.columns(4)
    for column, metric_name in zip(cv_columns, ["accuracy", "precision", "recall", "f1"]):
        column.metric(metric_name.title(), f"{cv_metrics[metric_name]:.1%}", border=True)

    show_test_evaluation(
        result_row,
        "Logistic Regression",
        RESULT_DIR / f"confusion_matrix_{result_suffix}.csv",
        RESULT_DIR / f"confusion_matrix_{result_suffix}.svg",
        RESULT_DIR / f"classification_report_{result_suffix}.csv",
    )

    st.markdown("#### Feature Engineering 적용 후 대비 테스트 성능 변화")
    for metric_name in ["Accuracy", "Precision", "Recall", "F1"]:
        delta = float(result_row[metric_name] - baseline_row[metric_name])
        direction = "▲" if delta >= 0 else "▼"
        color = "green" if delta >= 0 else "red"
        with st.container(border=True):
            label_column, score_column, change_column = st.columns(
                [1.5, 3.5, 1.2], vertical_alignment="center"
            )
            label_column.markdown(f"**{metric_name}**")
            score_column.progress(
                float(result_row[metric_name]),
                text=f"튜닝 후 {result_row[metric_name]:.1%}",
            )
            change_column.markdown(
                f"**튜닝 전 대비**  \n:{color}[{direction} {abs(delta) * 100:.1f}%p]"
            )

    show_feature_importance(importance_data, stage_key)
    method_info_path = RESULT_DIR / f"metadata_{result_suffix}.json"
    method_model_info = json.loads(method_info_path.read_text(encoding="utf-8"))
    show_data_model_info(method_model_info, result_row, method_title)

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
    "total_nights": ("총 숙박일 수", "평일과 주말을 합한 전체 숙박일 수"),
    "total_guests": ("총 투숙객 수", "성인, 어린이, 영아를 합한 전체 투숙객 수"),
    "is_family": ("가족 예약 여부", "어린이 또는 영아가 포함된 예약인지 여부"),
    "previous_bookings_total": ("과거 전체 예약 수", "과거 취소와 정상 예약을 합한 개수"),
    "previous_cancellation_rate": ("과거 취소율", "과거 전체 예약 중 취소한 비율"),
    "has_special_requests": ("특별 요청 여부", "특별 요청이 하나 이상 있는지 여부"),
    "has_booking_changes": ("예약 변경 여부", "예약 내용을 한 번 이상 변경했는지 여부"),
    "is_agent_booking": ("대행사 예약 여부", "여행사·대행사를 통한 예약인지 여부"),
    "is_company_booking": ("기업 예약 여부", "기업과 관련된 예약인지 여부"),
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
required = [
    "model_comparison_before_feature_engineering.csv",
    "model_comparison_after_feature_engineering.csv",
    "confusion_matrix_before_feature_engineering.csv",
    "confusion_matrix_after_feature_engineering.csv",
    "classification_report_before_feature_engineering.csv",
    "classification_report_after_feature_engineering.csv",
    "feature_importance_before_feature_engineering.csv",
    "feature_importance_after_feature_engineering.csv",
    "metadata_before_feature_engineering.json",
    "metadata_after_feature_engineering.json",
    "feature_engineering_comparison.csv",
    "tuning_comparison.csv",
    "tuning_metadata.json",
    "confusion_matrix_after_tuning.csv",
    "classification_report_after_tuning.csv",
    "feature_importance_after_tuning.csv",
    "metadata_after_tuning.json",
    "tuning_methods_comparison.csv",
    "tuning_metadata_grid_search.json",
    "tuning_metadata_optuna.json",
    "confusion_matrix_grid_search.csv",
    "confusion_matrix_optuna.csv",
    "classification_report_grid_search.csv",
    "classification_report_optuna.csv",
    "feature_importance_grid_search.csv",
    "feature_importance_optuna.csv",
    "metadata_grid_search.json",
    "metadata_optuna.json",
]
missing = [name for name in required if not (RESULT_DIR / name).exists()]
if missing:
    st.error("학습 결과가 없습니다. 먼저 `python src/train.py`를 실행해 주세요.")
    st.stop()

# 학습 결과 파일을 한 번만 읽어 이후 화면 구성에 재사용한다.
before_model_comparison = pd.read_csv(
    RESULT_DIR / "model_comparison_before_feature_engineering.csv"
)
after_model_comparison = pd.read_csv(
    RESULT_DIR / "model_comparison_after_feature_engineering.csv"
)
before_importance = pd.read_csv(
    RESULT_DIR / "feature_importance_before_feature_engineering.csv"
)
after_importance = pd.read_csv(
    RESULT_DIR / "feature_importance_after_feature_engineering.csv"
)
before_metadata = json.loads(
    (RESULT_DIR / "metadata_before_feature_engineering.json").read_text(encoding="utf-8")
)
after_metadata = json.loads(
    (RESULT_DIR / "metadata_after_feature_engineering.json").read_text(encoding="utf-8")
)
feature_comparison = pd.read_csv(RESULT_DIR / "feature_engineering_comparison.csv")
tuning_comparison = pd.read_csv(RESULT_DIR / "tuning_comparison.csv")
tuning_metadata = json.loads(
    (RESULT_DIR / "tuning_metadata.json").read_text(encoding="utf-8")
)
after_tuning_importance = pd.read_csv(
    RESULT_DIR / "feature_importance_after_tuning.csv"
)
all_tuning_comparison = pd.read_csv(RESULT_DIR / "tuning_methods_comparison.csv")
grid_tuning_metadata = json.loads(
    (RESULT_DIR / "tuning_metadata_grid_search.json").read_text(encoding="utf-8")
)
optuna_tuning_metadata = json.loads(
    (RESULT_DIR / "tuning_metadata_optuna.json").read_text(encoding="utf-8")
)
grid_tuning_importance = pd.read_csv(RESULT_DIR / "feature_importance_grid_search.csv")
optuna_tuning_importance = pd.read_csv(RESULT_DIR / "feature_importance_optuna.csv")

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

st.subheader("Feature Engineering 전·후 비교")
stage_labels = {
    "before_feature_engineering": "적용 전",
    "after_feature_engineering": "적용 후",
}
model_labels = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}
feature_display = feature_comparison.copy()
feature_display["단계"] = feature_display["stage"].map(stage_labels)
feature_display["선택 모델"] = feature_display["selected_model"].map(model_labels)
feature_display = feature_display.rename(
    columns={
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "rows": "테스트 건수",
        "cancellation_rate": "취소율",
    }
)

before_row = feature_display.loc[feature_display["단계"].eq("적용 전")].iloc[0]
after_row = feature_display.loc[feature_display["단계"].eq("적용 후")].iloc[0]

tuning_stage_labels = {"before_tuning": "튜닝 전", "after_tuning": "튜닝 후"}
tuning_model_labels = {
    "logistic_regression": "Logistic Regression",
    "logistic_regression_tuned": "Logistic Regression",
    "random_forest": "Random Forest",
    "random_forest_tuned": "Random Forest",
}
tuning_display = tuning_comparison.copy()
tuning_display["단계"] = tuning_display["stage"].map(tuning_stage_labels)
tuning_display["선택 모델"] = tuning_display["selected_model"].map(tuning_model_labels)
tuning_display = tuning_display.rename(
    columns={
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "rows": "테스트 건수",
        "cancellation_rate": "취소율",
    }
)
before_tuning_row = tuning_display.loc[tuning_display["단계"].eq("튜닝 전")].iloc[0]
after_tuning_row = tuning_display.loc[tuning_display["단계"].eq("튜닝 후")].iloc[0]
all_tuning_display = all_tuning_comparison.rename(
    columns={
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "rows": "테스트 건수",
        "cancellation_rate": "취소율",
    }
)
all_tuning_display["선택 모델"] = "Logistic Regression"
grid_tuning_row = all_tuning_display.loc[
    all_tuning_display["method"].eq("grid_search")
].iloc[0]
optuna_tuning_row = all_tuning_display.loc[
    all_tuning_display["method"].eq("optuna")
].iloc[0]

before_tab, after_tab, randomized_tab, grid_tab, optuna_tab = st.tabs(
    [
        "Feature Engineering 적용 전",
        "Feature Engineering 적용 후 (튜닝 전)",
        "RandomizedSearchCV",
        "GridSearchCV",
        "Optuna",
    ]
)

with before_tab:
    st.markdown("#### 원본 변수만 사용한 결과")
    st.caption("원본 예약 데이터의 입력 변수 27개를 사용하고 파생 변수는 추가하지 않았습니다.")

    model_column, feature_column, row_column, rate_column = st.columns(
        [1.6, 1, 1, 1]
    )
    model_column.metric("선택 모델", before_row["선택 모델"], border=True)
    feature_column.metric("입력 변수", "27개", border=True)
    row_column.metric("테스트 예약", f"{int(before_row['테스트 건수']):,}건", border=True)
    rate_column.metric("테스트 취소율", f"{before_row['취소율']:.1%}", border=True)

    st.markdown("#### 후보 모델 비교")
    st.caption("원본 변수만 사용했을 때 검증 데이터에서 두 모델의 성능을 비교한 결과입니다.")
    show_model_comparison(
        before_model_comparison,
        RESULT_DIR / "model_comparison_before_feature_engineering.svg",
    )
    show_test_evaluation(
        before_row,
        before_row["선택 모델"],
        RESULT_DIR / "confusion_matrix_before_feature_engineering.csv",
        RESULT_DIR / "confusion_matrix_before_feature_engineering.svg",
        RESULT_DIR / "classification_report_before_feature_engineering.csv",
    )
    show_feature_importance(before_importance, "before")
    show_data_model_info(before_metadata, before_row, "Feature Engineering 적용 전")

with after_tab:
    st.markdown("#### 파생 변수 9개를 추가한 결과 (튜닝 전)")
    st.caption(
        "원본 입력 변수 27개에 파생 변수 9개를 추가한 36개 입력 변수의 결과이며, "
        "하이퍼파라미터 튜닝 전 기준 모델입니다."
    )

    model_column, feature_column, row_column, rate_column = st.columns(
        [1.6, 1, 1, 1]
    )
    model_column.metric("선택 모델", after_row["선택 모델"], border=True)
    feature_column.metric("입력 변수", "36개", border=True)
    row_column.metric("테스트 예약", f"{int(after_row['테스트 건수']):,}건", border=True)
    rate_column.metric("테스트 취소율", f"{after_row['취소율']:.1%}", border=True)

    st.markdown("**추가한 파생 변수**")
    st.markdown(
        "- 총 숙박일(`total_nights`), 총 투숙객 수(`total_guests`), 가족 예약 여부(`is_family`)\n"
        "- 과거 예약 수(`previous_bookings_total`), 과거 취소율(`previous_cancellation_rate`)\n"
        "- 특별 요청 여부(`has_special_requests`), 예약 변경 여부(`has_booking_changes`)\n"
        "- 여행사 예약 여부(`is_agent_booking`), 기업 예약 여부(`is_company_booking`)"
    )
    show_parameter_settings("튜닝 전 하이퍼파라미터", tuning_metadata["baseline_params"])
    st.markdown("#### 후보 모델 비교")
    st.caption("파생 변수 9개를 추가했을 때 검증 데이터에서 두 모델의 성능을 비교한 결과입니다.")
    show_model_comparison(
        after_model_comparison,
        RESULT_DIR / "model_comparison_after_feature_engineering.svg",
    )
    show_test_evaluation(
        after_row,
        after_row["선택 모델"],
        RESULT_DIR / "confusion_matrix_after_feature_engineering.csv",
        RESULT_DIR / "confusion_matrix_after_feature_engineering.svg",
        RESULT_DIR / "classification_report_after_feature_engineering.csv",
    )

    st.markdown("#### 성능 변화")
    st.caption("Feature Engineering 적용 전과 비교한 적용 후의 변화를 보여줍니다.")
    metric_descriptions = {
        "Accuracy": "전체 정답률",
        "Precision": "취소 예측의 정확성",
        "Recall": "실제 취소 탐지율",
        "F1": "Precision·Recall 균형",
    }
    for metric_name in ["Accuracy", "Precision", "Recall", "F1"]:
        delta = float(after_row[metric_name] - before_row[metric_name])
        direction = "▲" if delta >= 0 else "▼"
        color = "green" if delta >= 0 else "red"
        with st.container(border=True):
            label_column, score_column, change_column = st.columns(
                [1.5, 3.5, 1.2], vertical_alignment="center"
            )
            label_column.markdown(
                f"**{metric_name}**  \n{metric_descriptions[metric_name]}"
            )
            score_column.progress(
                float(after_row[metric_name]),
                text=f"적용 후 {after_row[metric_name]:.1%}",
            )
            change_column.markdown(
                f"**적용 전 대비**  \n:{color}[{direction} {abs(delta) * 100:.1f}%p]"
            )

    st.info(
        f"Feature Engineering 후 Recall은 {before_row['Recall']:.1%}에서 {after_row['Recall']:.1%}로, "
        f"F1은 {before_row['F1']:.1%}에서 {after_row['F1']:.1%}로 높아졌습니다. "
        f"반면 Accuracy는 {before_row['Accuracy']:.1%}에서 {after_row['Accuracy']:.1%}로, "
        f"Precision은 {before_row['Precision']:.1%}에서 {after_row['Precision']:.1%}로 낮아졌습니다. "
        "즉, 실제 취소를 더 많이 찾는 대신 정상 예약을 취소로 판단하는 경우가 늘었습니다."
    )
    show_feature_importance(after_importance, "after")
    show_data_model_info(after_metadata, after_row, "Feature Engineering 적용 후")

with randomized_tab:
    show_tuning_method_result(
        "RandomizedSearchCV",
        after_tuning_row,
        before_tuning_row,
        tuning_metadata,
        after_tuning_importance,
        "after_tuning",
        "randomized_tuning",
    )

with grid_tab:
    show_tuning_method_result(
        "GridSearchCV",
        grid_tuning_row,
        before_tuning_row,
        grid_tuning_metadata,
        grid_tuning_importance,
        "grid_search",
        "grid_tuning",
    )

with optuna_tab:
    show_tuning_method_result(
        "Optuna",
        optuna_tuning_row,
        before_tuning_row,
        optuna_tuning_metadata,
        optuna_tuning_importance,
        "optuna",
        "optuna_tuning",
    )

st.divider()
st.caption("연구·의사결정 지원용 모델입니다. 고객 자동 거절이나 차별적 조치에 사용하지 마세요.")
