# Lisbon City Hotel 예약 취소 예측

포르투갈 Lisbon의 City Hotel 예약을 대상으로, 예약 시점에 취소 가능성을 예측하는 재현 가능한 머신러닝 프로젝트입니다.

## 이 모델로 해결하려는 문제

호텔 예약이 실제 숙박 전에 취소되면 빈 객실, 매출 손실, 인력 및 객실 운영계획의 불확실성이 발생할 수 있습니다. 이 프로젝트는 예약 당시 확인할 수 있는 리드타임, 객실요금, 보증금 유형, 특별 요청 수, 예약 경로 등의 정보를 사용해 **해당 예약이 향후 취소될 가능성**을 예측합니다.

호텔 운영자는 예측 결과를 다음과 같은 저위험 대응의 우선순위로 활용할 수 있습니다.

- 취소 위험이 높은 예약에 확인 또는 리마인더 메시지 발송
- 예상 취소량을 고려한 객실 재판매와 수요 관리
- 취소 위험을 반영한 객실 및 인력 운영계획 수립

예측 대상은 `is_canceled`입니다.

| 값 | 의미 |
|---:|---|
| 0 | 예약이 취소되지 않음 |
| 1 | 예약이 취소됨 |

## 문제 유형: 이진 분류

이 프로젝트는 지도학습 기반 **이진 분류(Binary Classification)** 문제입니다. 입력 변수로 각 예약의 취소 확률을 계산한 뒤, 판단 임계값 이상이면 취소 위험 예약인 `1`, 미만이면 정상 예약인 `0`으로 분류합니다.

- 분류: 정답이 `취소(1)` 또는 `정상(0)`이므로 해당

비교 모델로 Logistic Regression과 Random Forest Classifier를 사용하며, 검증 F1이 더 높은 모델을 최종 선택합니다. F1이 같으면 Accuracy를 비교합니다.

## 데이터 출처

- 데이터셋: [Kaggle Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- 원본 파일: `hotel_bookings.csv`
- 수집 기간: 2015년 7월 1일~2017년 8월 31일
- 라이선스: CC BY 4.0
- 원저자: Nuno Antonio, Ana de Almeida, Luís Nunes
- 원논문: *Hotel booking demand datasets*, Data in Brief, 2019

Kaggle에서 파일을 내려받아 `data/raw/hotel_bookings.csv`에 배치해야 합니다.

## 핵심 설계

- 전체 119,390건 중 `City Hotel` 79,330건만 사용
- `is_canceled`를 예측
- 미래 정보가 과거 학습에 섞이지 않도록 도착일 기준 시간순 분할(학습 70% / 검증 15% / 테스트 15%)
- 예약 후 정보인 `assigned_room_type`, `reservation_status`, `reservation_status_date` 제거
- Logistic Regression과 Random Forest 비교
- 기본 분류 기준 0.5를 사용하고, 검증 F1으로 모델을 선택한 후 테스트 세트는 한 번만 평가
- Accuracy, Precision, Recall, F1, 혼동행렬, Classification Report 저장
- 후보 모델 성능 비교 표·그래프와 변수 중요도 저장

`assigned_room_type`은 예약 후 최종 배정된 객실 유형이고, `reservation_status`와 `reservation_status_date`는 예약의 최종 상태와 해당 상태가 확정된 날짜입니다. 예약 시점에 알 수 없거나 취소 결과를 직접 알려주는 사후 정보이므로, 성능이 부풀려지는 데이터 누수를 방지하기 위해 전처리 단계에서 제거합니다.

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
| `src/data_profile.py` | `head`, `info`, `describe`, `shape`, 결측값과 중복값 점검 |
| `src/eda.py` | 특성·타겟 분포, 상관관계와 변수별 취소율 분석·시각화 |
| `src/models.py` | 결측치 처리, 원-핫 인코딩, Logistic Regression·Random Forest 학습 |
| `src/evaluation.py` | 지표 계산, 최종 모델 선택, 결과 파일·그래프 저장 |
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

원본은 CC BY 4.0이며, Antonio, Almeida, Nunes (2019)의 *Hotel booking demand datasets*에서 유래했습니다.

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

1. `head`, `info`, `describe`, `shape`, 결측값과 중복값 확인
2. 특성·타겟 분포와 변수 간·타겟 간 관계를 EDA로 분석
3. 전체 데이터에서 City Hotel 79,330건 추출
4. 도착일 기준 학습 70%, 검증 15%, 테스트 15% 분할
5. 두 후보 모델 학습 및 검증 성능 비교
6. 검증 F1과 Accuracy로 최종 모델 선택
7. 테스트 데이터 최종 평가
8. 모델·지표·그래프를 `outputs/model/`에 저장

## 데이터 기본 정보 확인

`train.py`를 실행하면 모델에 사용하는 City Hotel 데이터의 기본 정보가 `outputs/data_profile/`에 자동 저장됩니다.

| 결과 파일 | 내용 |
|---|---|
| `overview.json` | City Hotel의 `shape`, 전체 결측값 수, 중복 행 수 |
| `city_hotel_head.csv` | City Hotel 데이터의 상위 5행 (`head`) |
| `city_hotel_info.txt` | 열, 자료형, non-null 개수 (`info`) |
| `city_hotel_describe.csv` | 기술통계 (`describe`) |
| `city_hotel_missing_values.csv` | 열별 결측값 개수와 비율 |

중복 행은 개수만 확인하고 자동 삭제하지 않습니다. 익명화된 데이터에는 서로 다른 예약이 동일한 값 조합으로 보일 수 있으므로, 중복이라는 이유만으로 삭제하면 실제 예약을 잃을 수 있기 때문입니다.

## 탐색적 데이터 분석 (EDA)

City Hotel 데이터만 대상으로 다음 분석을 수행하며 결과는 `outputs/eda/`에 저장됩니다.

- 타겟 분포: 취소·정상 예약 건수와 비율
- 숫자 특성 분포: 리드타임, ADR, 특별 요청 수, 과거 취소 횟수, 숙박일 수
- 범주 특성 분포: 보증금, 시장 세그먼트, 고객 유형, 도착 월, 국가, 식사 유형
- 변수 간 상관관계: 주요 숫자 변수의 Pearson 상관계수와 히트맵
- 변수와 타겟 간 관계: 범주 및 구간별 취소율

```text
outputs/eda/
├─ target_distribution.csv / .png / .svg
├─ numeric_summary.csv
├─ numeric_distributions.png / .svg
├─ categorical_distributions.png / .svg
├─ correlation_matrix.csv
├─ correlation_heatmap.png / .svg
├─ target_numeric_correlations.csv
├─ target_relationships.csv
└─ target_relationships.png / .svg
```

상관관계는 변수 간 선형 관계를 나타낼 뿐 인과관계를 의미하지 않습니다. Random Forest 변수 중요도와도 다른 개념입니다.

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

기본 주소는 `http://localhost:8501`입니다. EDA, 성능 요약, 모델 비교, 혼동행렬, Classification Report, 주요 변수를 한 화면에서 확인할 수 있습니다.

대시보드는 모델을 다시 학습하지 않습니다. `outputs/model/`에 이미 저장된 결과 파일을 읽어 화면에 표시하므로 빠르게 실행됩니다. 데이터나 모델을 변경했다면 `train.py`를 먼저 다시 실행해야 합니다.

## 주요 평가 지표

| 지표 | 의미 |
|---|---|
| Accuracy | 전체 예약 중 취소·정상 여부를 정확히 맞힌 비율 |
| Precision | 취소라고 예측한 예약 중 실제 취소 비율 |
| Recall | 실제 취소 예약 중 모델이 찾아낸 비율 |
| F1 | Precision과 Recall의 조화 평균 |

Classification Report에서는 취소·정상 예약별 Precision, Recall, F1-Score와 건수를 함께 확인합니다.

## 학습 결과 파일

학습 완료 후 `outputs/model/`에 모델과 지표가 생성됩니다.

```text
model.joblib                 학습된 전처리+모델
metadata.json                모델 설정, 입력 변수, 데이터 기간
metrics.json                 검증/테스트 성능
model_comparison.csv         후보 모델 비교
model_comparison.png / .svg  후보 모델 평가 지표 비교 그래프
feature_importance.csv       주요 예측 변수
confusion_matrix.png / .svg  테스트 혼동행렬
confusion_matrix.csv         테스트 혼동행렬 원본 수치
classification_report.csv    클래스별 Precision·Recall·F1-Score
```

## 새 예약 예측

이 단계는 선택 사항입니다. 현재 접수 중인 신규 예약 CSV가 있을 때만 사용합니다. 입력 파일은 학습 데이터와 같은 설명 변수 열을 가져야 하며 `is_canceled`는 없어도 됩니다.

```powershell
.venv\Scripts\python.exe src\predict.py `
  --input new_bookings.csv `
  --output outputs\predictions.csv
```

출력에는 `cancellation_probability`와 `predicted_canceled`가 추가됩니다.
