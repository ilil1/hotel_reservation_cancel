"""Lisbon City Hotel 예약 취소 모델 학습을 실행하는 진입점."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import LEAKAGE_COLUMNS, chronological_split, load_city_hotel
from data_profile import create_data_profile
from eda import generate_eda
from evaluation import (
    calculate_metrics,
    save_all_tuning_comparison,
    save_feature_engineering_comparison,
    save_named_tuning_result,
    save_results,
    save_stage_model_comparisons,
    save_stage_model_details,
    save_stage_test_details,
    save_tuning_results,
    select_best_model,
)
from features import add_engineered_features
from models import fit_candidates
from tuning import (
    tune_with_grid_search,
    tune_with_optuna,
    tune_with_randomized_search,
)


def parse_args() -> argparse.Namespace:
    """데이터 경로, 결과 경로와 난수 시드를 읽는다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/hotel_bookings.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def train_variant(x, y, random_state: int) -> dict:
    """동일한 분할·모델 선택 기준으로 하나의 데이터 버전을 학습하고 평가한다."""
    split = chronological_split(x, y)
    x_train, y_train, x_valid, y_valid, x_test, y_test = split
    fitted_models = fit_candidates(x_train, y_train, random_state)
    model_name, model, comparison = select_best_model(
        fitted_models, x_valid, y_valid
    )
    threshold = 0.5
    valid_probability = model.predict_proba(x_valid)[:, 1]
    test_probability = model.predict_proba(x_test)[:, 1]
    report = {
        "selected_model": model_name,
        "selection_rule": "highest validation F1, then Accuracy",
        "validation": calculate_metrics(y_valid, valid_probability, threshold),
        "test": calculate_metrics(y_test, test_probability, threshold),
    }
    return {
        "model": model,
        "model_name": model_name,
        "comparison": comparison,
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "x_test": x_test,
        "valid_probability": valid_probability,
        "y_test": y_test,
        "test_probability": test_probability,
        "threshold": threshold,
        "report": report,
    }


def evaluate_tuned_model(method_key: str, tuning_info: dict, baseline: dict) -> dict:
    """튜닝 모델을 같은 검증·테스트 데이터로 평가해 공통 결과 형식으로 반환한다."""
    model = tuning_info["model"]
    threshold = 0.5
    valid_probability = model.predict_proba(baseline["x_valid"])[:, 1]
    test_probability = model.predict_proba(baseline["x_test"])[:, 1]
    model_name = f"{baseline['model_name']}_{method_key}"
    return {
        "model": model,
        "model_name": model_name,
        "y_valid": baseline["y_valid"],
        "valid_probability": valid_probability,
        "y_test": baseline["y_test"],
        "test_probability": test_probability,
        "threshold": threshold,
        "report": {
            "selected_model": model_name,
            "selection_rule": tuning_info["selection_rule"],
            "validation": calculate_metrics(
                baseline["y_valid"], valid_probability, threshold
            ),
            "test": calculate_metrics(
                baseline["y_test"], test_probability, threshold
            ),
        },
    }
def main() -> None:
    """데이터 준비부터 최종 결과 저장까지 순서대로 실행한다."""
    args = parse_args()

    # 1. head, info, describe, shape, 결측값과 중복값 보고서를 만든다.
    profile_output = args.output.parent / "data_profile"
    create_data_profile(args.data, profile_output)

    # 2. 특성·타겟 분포, 상관관계와 변수별 취소율을 분석한다.
    eda_output = args.output.parent / "eda"
    generate_eda(args.data, eda_output)

    # 3. Feature Engineering 전 원본 변수로 기준 모델을 학습·평가한다.
    x_before, y = load_city_hotel(args.data)
    before = train_variant(x_before, y, args.random_state)

    # 4. 파생 특성을 추가한 뒤 동일한 분할·기준으로 다시 학습·평가한다.
    x_after = add_engineered_features(x_before)
    after = train_variant(x_after, y, args.random_state)

    # 5. 동일한 학습 데이터와 시간순 교차검증으로 세 가지 튜닝 방법을 비교한다.
    tuning_infos = {
        "randomized_search": tune_with_randomized_search(
            after["model_name"], after["x_train"], after["y_train"], args.random_state
        ),
        "grid_search": tune_with_grid_search(
            after["model_name"], after["x_train"], after["y_train"], args.random_state
        ),
        "optuna": tune_with_optuna(
            after["model_name"], after["x_train"], after["y_train"], args.random_state
        ),
    }
    tuned_results = {
        method: evaluate_tuned_model(method, info, after)
        for method, info in tuning_infos.items()
    }
    final_method = max(
        tuned_results,
        key=lambda method: (
            tuned_results[method]["report"]["validation"]["f1"],
            tuned_results[method]["report"]["validation"]["accuracy"],
        ),
    )
    tuned = tuned_results[final_method]
    tuning_info = tuning_infos[final_method]

    # 6. 튜닝 후 모델을 신규 예약 추론에 사용할 최종 모델로 저장한다.
    report = save_results(
        model=tuned["model"],
        model_name=tuned["model_name"],
        comparison=after["comparison"],
        x=x_after,
        y_valid=tuned["y_valid"],
        valid_probability=tuned["valid_probability"],
        y_test=tuned["y_test"],
        test_probability=tuned["test_probability"],
        threshold=tuned["threshold"],
        leakage_columns=LEAKAGE_COLUMNS,
        output=args.output,
        selection_rule=tuning_info["selection_rule"],
    )
    save_feature_engineering_comparison(
        before["report"], after["report"], args.output
    )
    save_stage_model_comparisons(
        before["comparison"], after["comparison"], args.output
    )
    save_stage_test_details(before, after, args.output)
    save_stage_model_details(
        before,
        after,
        x_before,
        x_after,
        LEAKAGE_COLUMNS,
        args.output,
    )
    save_tuning_results(
        after,
        tuned_results["randomized_search"],
        tuning_infos["randomized_search"],
        x_after,
        LEAKAGE_COLUMNS,
        args.output,
    )
    save_named_tuning_result(
        "grid_search",
        tuned_results["grid_search"],
        tuning_infos["grid_search"],
        x_after,
        LEAKAGE_COLUMNS,
        args.output,
    )
    save_named_tuning_result(
        "optuna",
        tuned_results["optuna"],
        tuning_infos["optuna"],
        x_after,
        LEAKAGE_COLUMNS,
        args.output,
    )
    save_all_tuning_comparison(after, tuned_results, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
