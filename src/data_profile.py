"""학습 전 데이터의 구조, 요약 통계, 결측값과 중복값을 점검한다."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd


def _save_dataset_profile(frame: pd.DataFrame, name: str, output: Path) -> dict:
    """하나의 데이터프레임에 대한 기본 정보 파일을 저장한다."""
    frame.head().to_csv(output / f"{name}_head.csv", index=False)

    # info()는 반환값 대신 출력 스트림에 내용을 쓰므로 문자열로 받아 저장한다.
    info_buffer = io.StringIO()
    frame.info(buf=info_buffer, show_counts=True)
    (output / f"{name}_info.txt").write_text(
        info_buffer.getvalue(), encoding="utf-8"
    )

    frame.describe(include="all").transpose().to_csv(
        output / f"{name}_describe.csv"
    )

    missing = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": frame.isna().sum().values,
            "missing_rate": (frame.isna().mean().values * 100),
        }
    ).sort_values("missing_count", ascending=False)
    missing.to_csv(output / f"{name}_missing_values.csv", index=False)

    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "total_missing_values": int(frame.isna().sum().sum()),
    }


def create_data_profile(data_path: Path, output: Path) -> dict:
    """모델에 사용하는 City Hotel 데이터의 기본 정보 보고서를 생성한다."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    output.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(data_path)
    city = original.loc[original["hotel"].eq("City Hotel")].copy()

    overview = {
        "data_path": str(data_path),
        "city_hotel": _save_dataset_profile(city, "city_hotel", output),
    }
    (output / "overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[데이터 기본 정보]")
    print(
        f"City Hotel shape: {city.shape} | 중복 행: "
        f"{overview['city_hotel']['duplicate_rows']:,}개 | 결측값: "
        f"{overview['city_hotel']['total_missing_values']:,}개"
    )
    print(f"상세 보고서: {output}\n")
    return overview
