"""시간순 교차검증으로 여러 방식의 하이퍼파라미터 튜닝을 수행한다."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from models import build_candidates, build_preprocessor


MIN_RECALL = 0.80
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def _build_pipeline(model_name: str, x_train: pd.DataFrame, random_state: int) -> Pipeline:
    """선택 모델과 전처리를 하나의 파이프라인으로 구성한다."""
    estimator = build_candidates(random_state)[model_name]
    return Pipeline(
        [("preprocess", build_preprocessor(x_train)), ("model", estimator)]
    )


def _parameter_space(model_name: str) -> dict:
    """RandomizedSearchCV와 GridSearchCV가 공통으로 사용할 탐색 범위를 반환한다."""
    if model_name == "logistic_regression":
        return {
            "model__C": [0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            "model__class_weight": [None, "balanced"],
        }
    return {
        "model__n_estimators": [200, 350, 500],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [2, 5, 10],
        "model__max_features": ["sqrt", 0.5],
        "model__class_weight": ["balanced", "balanced_subsample"],
    }


def _select_balanced_high_recall_model(cv_results: dict) -> int:
    """평균 Recall 80% 이상 후보 중 F1이 가장 높은 설정을 선택한다."""
    recall = np.asarray(cv_results["mean_test_recall"])
    f1 = np.asarray(cv_results["mean_test_f1"])
    eligible = np.flatnonzero(recall >= MIN_RECALL)
    if len(eligible):
        return int(eligible[np.argmax(f1[eligible])])
    return int(np.argmax(recall))


def _selection_rule(recall_values) -> str:
    """실제 후보 성능에 맞는 선택 규칙 설명을 반환한다."""
    reached = bool(np.max(np.asarray(recall_values)) >= MIN_RECALL)
    if reached:
        return "mean CV Recall >= 0.80, then highest mean CV F1"
    return "no candidate reached mean CV Recall 0.80; selected highest mean CV Recall"


def _baseline_params(model_name: str, random_state: int, parameter_space: dict) -> dict:
    """튜닝 전 모델에서 탐색 대상 파라미터의 초기값만 추출한다."""
    estimator = build_candidates(random_state)[model_name]
    names = [name.replace("model__", "") for name in parameter_space]
    return {name: estimator.get_params()[name] for name in names}


def _search_result(search, method: str, parameter_space: dict, model_name: str, random_state: int) -> dict:
    """scikit-learn 탐색 객체를 대시보드에서 공통으로 읽는 결과 형식으로 변환한다."""
    best_index = int(search.best_index_)
    cv_summary = {
        metric: round(float(search.cv_results_[f"mean_test_{metric}"][best_index]), 6)
        for metric in SCORING
    }
    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_metrics": cv_summary,
        "baseline_params": _baseline_params(model_name, random_state, parameter_space),
        "parameter_space": parameter_space,
        "search_method": method,
        "cv_method": "TimeSeriesSplit(n_splits=3)",
        "selection_rule": _selection_rule(search.cv_results_["mean_test_recall"]),
    }


def tune_with_randomized_search(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict:
    """RandomizedSearchCV로 8개 조합을 탐색한다."""
    parameter_space = _parameter_space(model_name)
    search = RandomizedSearchCV(
        estimator=_build_pipeline(model_name, x_train, random_state),
        param_distributions=parameter_space,
        n_iter=8,
        scoring=SCORING,
        refit=_select_balanced_high_recall_model,
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
        random_state=random_state,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    result = _search_result(
        search, "RandomizedSearchCV", parameter_space, model_name, random_state
    )
    result["trial_count"] = 8
    return result


def tune_with_grid_search(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict:
    """GridSearchCV로 지정한 모든 파라미터 조합을 탐색한다."""
    parameter_space = _parameter_space(model_name)
    search = GridSearchCV(
        estimator=_build_pipeline(model_name, x_train, random_state),
        param_grid=parameter_space,
        scoring=SCORING,
        refit=_select_balanced_high_recall_model,
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    result = _search_result(
        search, "GridSearchCV", parameter_space, model_name, random_state
    )
    result["trial_count"] = len(search.cv_results_["params"])
    return result


def tune_with_optuna(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict:
    """Optuna TPE sampler로 12회 탐색하고 최적 모델을 전체 학습 데이터에 적합한다."""
    cv = TimeSeriesSplit(n_splits=3)
    trial_metrics: dict[int, dict[str, float]] = {}

    def objective(trial: optuna.Trial) -> float:
        pipeline = _build_pipeline(model_name, x_train, random_state)
        if model_name == "logistic_regression":
            params = {
                "model__C": trial.suggest_float("C", 0.02, 5.0, log=True),
                "model__class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
            }
        else:
            params = {
                "model__n_estimators": trial.suggest_int("n_estimators", 200, 500, step=50),
                "model__max_depth": trial.suggest_int("max_depth", 8, 24, step=4),
                "model__min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 2, 10
                ),
                "model__max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", 0.5]
                ),
            }
        pipeline.set_params(**params)
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )
        metrics = {
            name: float(np.mean(scores[f"test_{name}"])) for name in SCORING
        }
        trial_metrics[trial.number] = metrics
        trial.set_user_attr("metrics", metrics)
        return 1.0 + metrics["f1"] if metrics["recall"] >= MIN_RECALL else metrics["recall"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=12, show_progress_bar=False)

    best_params = study.best_trial.params
    pipeline_params = {f"model__{name}": value for name, value in best_params.items()}
    best_model = _build_pipeline(model_name, x_train, random_state)
    best_model.set_params(**pipeline_params)
    best_model.fit(x_train, y_train)
    best_metrics = trial_metrics[study.best_trial.number]
    recall_values = [metrics["recall"] for metrics in trial_metrics.values()]
    parameter_space = (
        {"C": "0.02~5.0 (log scale)", "class_weight": [None, "balanced"]}
        if model_name == "logistic_regression"
        else {
            "n_estimators": "200~500",
            "max_depth": "8~24",
            "min_samples_leaf": "2~10",
            "max_features": ["sqrt", 0.5],
        }
    )
    return {
        "model": best_model,
        "best_params": best_params,
        "best_cv_metrics": {
            name: round(value, 6) for name, value in best_metrics.items()
        },
        "baseline_params": _baseline_params(
            model_name, random_state, _parameter_space(model_name)
        ),
        "parameter_space": parameter_space,
        "search_method": "Optuna (TPE)",
        "cv_method": "TimeSeriesSplit(n_splits=3)",
        "trial_count": 12,
        "selection_rule": _selection_rule(recall_values),
    }


# 기존 호출 코드와의 호환성을 유지한다.
tune_selected_model = tune_with_randomized_search
