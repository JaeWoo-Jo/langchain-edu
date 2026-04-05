"""식품안전나라 I2790 API에서 영양성분 데이터를 수집한다.

사용법:
    uv run python download_nutrition.py
    uv run python download_nutrition.py --max-count 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from config import FOOD_SAFETY_API_KEY, NUTRITION_DIR

# 식품안전나라 식품영양성분 DB
API_BASE = "https://openapi.foodsafetykorea.go.kr/api"
SERVICE_ID = "I2790"
PAGE_SIZE = 100


def fetch_nutrition(api_key: str, start: int, end: int) -> list[dict]:
    """영양성분 API에서 start~end 범위의 데이터를 가져온다."""
    url = f"{API_BASE}/{api_key}/{SERVICE_ID}/json/{start}/{end}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    service_data = data.get(SERVICE_ID, {})
    result = service_data.get("row", [])
    return result


def get_total_count(api_key: str) -> int:
    """API에서 전체 식품 수를 조회한다."""
    url = f"{API_BASE}/{api_key}/{SERVICE_ID}/json/1/1"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return int(data.get(SERVICE_ID, {}).get("total_count", "0"))


def download_all_nutrition(api_key: str, max_count: int | None = None):
    """전체 영양성분 데이터를 수집하여 JSON 파일로 저장한다."""
    NUTRITION_DIR.mkdir(parents=True, exist_ok=True)

    total = get_total_count(api_key)
    if max_count:
        total = min(total, max_count)
    print(f"전체 식품 수: {total}")

    all_items: list[dict] = []
    for start in range(1, total + 1, PAGE_SIZE):
        end = min(start + PAGE_SIZE - 1, total)
        print(f"  수집 중: {start}~{end}")
        try:
            batch = fetch_nutrition(api_key, start, end)
            all_items.extend(batch)
        except Exception as e:
            print(f"  오류 발생 ({start}~{end}): {e}")
        time.sleep(0.5)

    output_path = NUTRITION_DIR / "nutrition.json"
    output_path.write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"저장 완료: {output_path} ({len(all_items)}건)")


def main():
    parser = argparse.ArgumentParser(description="식품안전나라 영양성분 데이터 수집")
    parser.add_argument("--max-count", type=int, default=None, help="최대 수집 건수")
    args = parser.parse_args()

    if not FOOD_SAFETY_API_KEY:
        print("오류: FOOD_SAFETY_API_KEY가 설정되지 않았습니다.")
        print("  .env 파일에 FOOD_SAFETY_API_KEY=your_key 를 추가하세요.")
        print("  공공데이터 포털(data.go.kr)에서 '식품영양성분 DB' API 키를 발급받으세요.")
        return

    download_all_nutrition(FOOD_SAFETY_API_KEY, args.max_count)


if __name__ == "__main__":
    main()
