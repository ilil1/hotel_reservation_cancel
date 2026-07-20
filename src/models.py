"""전처리 파이프라인과 후보 머신러닝 모델을 구성한다."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """숫자형과 범주형 변수에 맞는 결측치 처리와 변환을 구성한다."""
    categorical = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = x.select_dtypes(include=["number"]).columns.tolist()

    # 도착일은 데이터 분할에만 사용하고 모델 입력에서는 제외한다.
    categorical = [c for c in categorical if c != "arrival_date"]
    numeric = [c for c in numeric if c != "arrival_date"]

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def build_candidates(random_state: int) -> dict[str, object]:
    """성능을 비교할 기준 모델과 트리 모델을 만든다."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            C=0.5,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def fit_candidates(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict[str, Pipeline]:
    """모든 후보 모델을 같은 학습 데이터와 전처리 방식으로 훈련한다."""
    fitted = {}
    for name, estimator in build_candidates(random_state).items():
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
    return fitted

