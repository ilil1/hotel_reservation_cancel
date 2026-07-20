# Lisbon City Hotel 예약 취소 예측

포르투갈 Lisbon의 City Hotel 예약을 대상으로, 예약 시점에 취소 가능성을 예측하는 재현 가능한 머신러닝 프로젝트입니다.

## 핵심 설계

- 전체 119,390건 중 `City Hotel` 79,330건만 사용
- `is_canceled`를 예측
- 미래 정보가 과거 학습에 섞이지 않도록 도착일 기준 시간순 분할(학습 70% / 검증 15% / 테스트 15%)
- 예약 결과를 사후에 알려주는 `reservation_status`, `reservation_status_date` 제거
- Logistic Regression과 Random Forest 비교
- 검증 세트에서 F1 기준으로 의사결정 임계값 선택 후 테스트 세트는 한 번만 평가
- PR-AUC, ROC-AUC, Precision, Recall, F1, 혼동행렬과 변수 중요도 저장

## 데이터 준비

Kaggle의 [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)에서 `hotel_bookings.csv`를 내려받아 다음 위치에 둡니다.

```text
data/raw/hotel_bookings.csv
```

원본은 CC BY 4.0이며, Antonio, Almeida, Nunes (2019)의 *Hotel booking demand datasets*에서 유래했습니다. 저장소에는 원본 데이터를 재배포하지 않습니다.

## 실행

Python 3.10 이상을 권장합니다.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

## 웹 대시보드

학습 결과를 브라우저에서 확인할 수 있습니다.

```powershell
streamlit run dashboard.py
```

기본 주소는 `http://localhost:8501`입니다. 성능 요약, 모델 비교, 혼동행렬, PR 곡선, 주요 변수를 한 화면에서 확인할 수 있습니다.

빠른 동작 확인은 다음과 같습니다.

```powershell
python src/train.py --data data/sample/hotel_bookings_sample.csv --output work/sample_run
```

학습 완료 후 `outputs/model/`에 모델과 지표가 생성됩니다.

```text
model.joblib                 학습된 전처리+모델
metadata.json                임계값, 입력 변수, 데이터 기간
metrics.json                 검증/테스트 성능
model_comparison.csv         후보 모델 비교
feature_importance.csv       주요 예측 변수
confusion_matrix.png         테스트 혼동행렬
precision_recall_curve.png   검증 PR 곡선 및 선택 임계값
```

## 새 예약 예측

입력 CSV는 학습 데이터와 같은 설명 변수 열을 가져야 하며 `is_canceled`는 없어도 됩니다.

```powershell
python src/predict.py --input new_bookings.csv --output predictions.csv
```

출력에는 `cancellation_probability`와 `predicted_canceled`가 추가됩니다.

## 운영 시 유의점

이 모델은 연구용 익명 데이터의 과거 패턴을 학습합니다. 실제 운영 전에는 최근 Lisbon 호텔 데이터로 재학습하고, 잘못된 개입 비용을 반영해 임계값을 다시 정해야 합니다. 모델 점수는 자동 취소나 고객 차별이 아니라 보증금 안내, 리마인더 등 저위험 개입의 우선순위로 사용하는 것이 적절합니다.
