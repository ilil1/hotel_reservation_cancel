"""Lisbon City Hotel 예약 취소 모델 학습을 실행하는 진입점."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import LEAKAGE_COLUMNS, chronological_split, load_city_hotel
from evaluation import find_best_threshold, save_results, select_best_model
from models import fit_candidates


def parse_args() -> argparse.Namespace:
    """데이터 경로, 결과 경로와 난수 시드를 읽는다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/hotel_bookings.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """데이터 준비부터 최종 결과 저장까지 순서대로 실행한다."""
    args = parse_args()

    # 1. Lisbon City Hotel 데이터만 불러오고 시간순으로 나눈다.
    x, y = load_city_hotel(args.data)
    split = chronological_split(x, y)
    x_train, y_train, x_valid, y_valid, x_test, y_test = split

    # 2. 후보 모델을 학습하고 검증 성능이 가장 좋은 모델을 선택한다.
    fitted_models = fit_candidates(x_train, y_train, args.random_state)
    model_name, model, comparison = select_best_model(
        fitted_models, x_valid, y_valid
    )

    # 3. 검증 데이터로 판단 임계값을 정한 뒤 테스트 데이터를 평가한다.
    valid_probability = model.predict_proba(x_valid)[:, 1]
    threshold = find_best_threshold(y_valid, valid_probability)
    test_probability = model.predict_proba(x_test)[:, 1]

    # 4. 모델, 지표, 메타데이터, 중요도와 그래프를 저장한다.
    report = save_results(
        model=model,
        model_name=model_name,
        comparison=comparison,
        x=x,
        y_valid=y_valid,
        valid_probability=valid_probability,
        y_test=y_test,
        test_probability=test_probability,
        threshold=threshold,
        leakage_columns=LEAKAGE_COLUMNS,
        output=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

