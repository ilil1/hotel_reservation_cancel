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
        st.image(str(matrix_path), use_container_width=True)
with right:
    st.subheader("Precision–Recall 곡선")
    curve_path = RESULT_DIR / "precision_recall_curve.png"
    if curve_path.exists():
        st.image(str(curve_path), use_container_width=True)

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
    use_container_width=True,
    hide_index=True,
)

# 사용자가 화면에 표시할 변수 개수를 조절할 수 있다.
st.subheader("주요 예측 변수")
top_n = st.slider("표시할 변수 수", min_value=5, max_value=30, value=15, step=5)
top = importance.head(top_n).copy()
top["feature"] = (
    top["feature"].str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)
st.bar_chart(top.set_index("feature")["importance"], horizontal=True, height=500)
st.caption("중요도는 예측에 기여한 크기이며, 취소 확률을 높이거나 낮추는 방향을 뜻하지는 않습니다.")

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
