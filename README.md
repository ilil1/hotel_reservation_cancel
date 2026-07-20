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

## 전체 실행 흐름

```text
data/raw/hotel_bookings.csv
            ↓
        src/train.py
            ↓
    src/data.py       데이터 준비
    src/models.py     모델 학습
    src/evaluation.py 모델 평가·저장
            ↓
       outputs/model/
          ↙       ↘
 dashboard.py     src/predict.py
 결과 웹 표시      신규 예약 예측
```

`train.py`는 전체 작업 순서만 관리합니다. 복잡한 세부 코드는 기능별 파일로 나누어 두었습니다.

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `src/train.py` | 아래 모듈을 순서대로 호출하는 학습 시작 파일 |
| `src/data.py` | CSV 로딩, City Hotel 추출, 누수 변수 제거, 시간순 데이터 분할 |
| `src/models.py` | 결측치 처리, 원-핫 인코딩, Logistic Regression·Random Forest 학습 |
| `src/evaluation.py` | 지표 계산, 임계값 결정, 최종 모델 선택, 결과 파일 저장 |
| `src/predict.py` | 학습된 모델을 실제 신규 예약 CSV에 적용할 때 사용 |
| `dashboard.py` | 이미 저장된 평가 결과를 웹 화면으로 표시 |
| `requirements.txt` | 실행에 필요한 Python 패키지 목록 |
| `data/sample/` | 전체 데이터 없이 빠르게 코드 동작을 시험하는 가상 샘플 |
| `outputs/model/` | 학습된 모델, 성능 지표, 그래프 저장 위치 |

## 데이터 준비

Kaggle의 [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)에서 `hotel_bookings.csv`를 내려받아 다음 위치에 둡니다.

```text
data/raw/hotel_bookings.csv
```

원본은 CC BY 4.0이며, Antonio, Almeida, Nunes (2019)의 *Hotel booking demand datasets*에서 유래했습니다. 저장소에는 원본 데이터를 재배포하지 않습니다.

원본 CSV에는 두 호텔의 예약이 함께 들어 있습니다.

```text
hotel
├─ City Hotel       79,330건  (Lisbon)
└─ Resort Hotel     40,060건  (Algarve)
                    ────────
전체               119,390건
```

두 호텔은 `hotel` 열로 구분됩니다. 이 프로젝트는 아래 조건으로 Lisbon의 City Hotel만 선택하므로 Resort Hotel 40,060건은 모델 학습에서 제외됩니다.

```python
city = frame.loc[frame["hotel"].eq("City Hotel")].copy()
```

## 설치

Python 3.10 이상을 권장합니다.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 모델 학습

실제 Kaggle 데이터로 학습합니다.

```powershell
.venv\Scripts\python.exe src\train.py
```

내부적으로 다음 순서가 자동 실행됩니다.

1. 전체 데이터에서 City Hotel 79,330건 추출
2. 도착일 기준 학습 70%, 검증 15%, 테스트 15% 분할
3. 두 후보 모델 학습 및 검증 성능 비교
4. 검증 데이터에서 F1 임계값 결정
5. 테스트 데이터 최종 평가
6. 모델·지표·그래프를 `outputs/model/`에 저장

빠른 동작 확인은 소규모 샘플로 실행합니다. 이 결과는 실제 성능으로 해석하지 않습니다.

```powershell
.venv\Scripts\python.exe src\train.py `
  --data data\sample\hotel_bookings_sample.csv `
  --output work\sample_run
```

## 웹 대시보드

학습 결과를 브라우저에서 확인할 수 있습니다.

```powershell
.venv\Scripts\streamlit.exe run dashboard.py
```

기본 주소는 `http://localhost:8501`입니다. 성능 요약, 모델 비교, 혼동행렬, PR 곡선, 주요 변수를 한 화면에서 확인할 수 있습니다.

대시보드는 모델을 다시 학습하지 않습니다. `outputs/model/`에 이미 저장된 결과 파일을 읽어 화면에 표시하므로 빠르게 실행됩니다. 데이터나 모델을 변경했다면 `train.py`를 먼저 다시 실행해야 합니다.

## 주요 평가 지표

| 지표 | 의미 |
|---|---|
| PR-AUC | 여러 임계값에서 Precision과 Recall의 전반적인 균형 |
| ROC-AUC | 취소 예약에 정상 예약보다 높은 위험 점수를 부여하는 능력 |
| Precision | 취소라고 예측한 예약 중 실제 취소 비율 |
| Recall | 실제 취소 예약 중 모델이 찾아낸 비율 |
| F1 | Precision과 Recall의 조화 평균 |
| 판단 임계값 | 이 확률 이상이면 취소 위험으로 분류하는 기준 |

현재 테스트 결과는 PR-AUC 0.824, ROC-AUC 0.867, Precision 0.684, Recall 0.803, F1 0.739입니다.

## 학습 결과 파일

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

이 단계는 선택 사항입니다. 현재 접수 중인 신규 예약 CSV가 있을 때만 사용합니다. 입력 파일은 학습 데이터와 같은 설명 변수 열을 가져야 하며 `is_canceled`는 없어도 됩니다.

```powershell
.venv\Scripts\python.exe src\predict.py `
  --input new_bookings.csv `
  --output outputs\predictions.csv
```

출력에는 `cancellation_probability`와 `predicted_canceled`가 추가됩니다.

## 운영 시 유의점

이 모델은 연구용 익명 데이터의 과거 패턴을 학습합니다. 실제 운영 전에는 최근 Lisbon 호텔 데이터로 재학습하고, 잘못된 개입 비용을 반영해 임계값을 다시 정해야 합니다. 모델 점수는 자동 취소나 고객 차별이 아니라 보증금 안내, 리마인더 등 저위험 개입의 우선순위로 사용하는 것이 적절합니다.
