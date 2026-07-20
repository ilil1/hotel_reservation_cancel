"""학습된 모델로 신규 Lisbon City Hotel 예약의 취소 확률을 계산한다.

입력 CSV를 읽어 학습 당시와 같은 변수를 구성하고, 각 예약에
취소 확률과 최종 취소 예측값(0/1)을 추가한 CSV를 만든다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    """모델과 신규 예약을 불러와 예측 결과 CSV를 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/model"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()

    # metadata에는 학습 때 사용한 열과 검증 데이터에서 정한 임계값이 들어 있다.
    model = joblib.load(args.model_dir / "model.joblib")
    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input)
    # 전체 호텔 파일을 입력해도 City Hotel 행만 예측 대상으로 남긴다.
    if "hotel" in frame.columns:
        frame = frame.loc[frame["hotel"].eq("City Hotel")].copy()
    required = [c for c in metadata["feature_columns"] if c != "arrival_date"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    # 학습 파이프라인의 입력 형태와 동일하게 도착일 열을 재구성한다.
    frame["arrival_date"] = pd.to_datetime(
        frame["arrival_date_year"].astype(str) + "-" + frame["arrival_date_month"].astype(str)
        + "-" + frame["arrival_date_day_of_month"].astype(str), errors="raise"
    )
    # predict_proba의 두 번째 열은 클래스 1, 즉 취소 확률이다.
    probability = model.predict_proba(frame[metadata["feature_columns"]])[:, 1]
    result = frame.copy()
    result["cancellation_probability"] = probability
    # 고정된 0.5가 아니라 검증 단계에서 최적화한 임계값을 사용한다.
    result["predicted_canceled"] = (probability >= metadata["threshold"]).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
