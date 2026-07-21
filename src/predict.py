"""학습된 모델로 신규 Lisbon City Hotel 예약의 취소 확률을 계산한다.

입력 CSV를 읽어 학습 당시와 같은 변수를 구성하고, 각 예약에
취소 확률과 최종 취소 예측값(0/1)을 추가한 CSV를 만든다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from inference import load_prediction_artifacts, predict_reservations


def main() -> None:
    """모델과 신규 예약을 불러와 예측 결과 CSV를 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/model"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()

    # 웹 예측 화면과 동일한 공통 추론 파이프라인을 사용한다.
    model, metadata = load_prediction_artifacts(args.model_dir)
    frame = pd.read_csv(args.input)
    result = predict_reservations(frame, model, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
