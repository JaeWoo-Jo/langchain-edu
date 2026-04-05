"""공공데이터 포털 식품영양성분 DB API에서 영양성분 데이터를 수집한다.

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

from config import NUTRITION_DIR

# 공공데이터 포털 식품영양성분 DB (식품의약품안전처)
API_BASE = "https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02"
# 공공데이터 포털 인증키 (URL 인코딩된 상태로 사용)
DATA_GO_KR_KEY = "g6JAyYA1Rw3V5kKpRO52c4FjsbfFx8XedX8G+L2mMgunFRdMEB03lnF1mw3H71LupTA/RdjiYkqYbWB8Xr/APA=="
PAGE_SIZE = 500


def fetch_nutrition(api_key: str, page_no: int, num_of_rows: int) -> list[dict]:
    """영양성분 API에서 지정 페이지의 데이터를 가져온다."""
    resp = requests.get(
        API_BASE,
        params={
            "serviceKey": api_key,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            "type": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("body", {}).get("items", [])


def get_total_count(api_key: str) -> int:
    """API에서 전체 식품 수를 조회한다."""
    resp = requests.get(
        API_BASE,
        params={
            "serviceKey": api_key,
            "pageNo": 1,
            "numOfRows": 1,
            "type": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return int(data.get("body", {}).get("totalCount", 0))


def download_all_nutrition(api_key: str, max_count: int | None = None):
    """전체 영양성분 데이터를 수집하여 JSON 파일로 저장한다."""
    NUTRITION_DIR.mkdir(parents=True, exist_ok=True)

    total = get_total_count(api_key)
    if max_count:
        total = min(total, max_count)
    print(f"전체 식품 수: {total}")

    output_path = NUTRITION_DIR / "nutrition.json"
    # 기존 데이터 이어받기
    all_items: list[dict] = []
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            all_items = existing
            print(f"  기존 데이터 {len(all_items)}건 로드")

    page_no = len(all_items) // PAGE_SIZE + 1
    save_interval = 1000  # 1000건마다 중간 저장

    while len(all_items) < total:
        remaining = total - len(all_items)
        rows = min(PAGE_SIZE, remaining)
        print(f"  수집 중: 페이지 {page_no} ({len(all_items)}/{total})")
        try:
            batch = fetch_nutrition(api_key, page_no, rows)
            if not batch:
                break
            all_items.extend(batch)
        except Exception as e:
            print(f"  오류 발생 (페이지 {page_no}): {e}")
        page_no += 1
        time.sleep(0.1)

        # 중간 저장
        if len(all_items) % save_interval < PAGE_SIZE:
            output_path.write_text(
                json.dumps(all_items, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [중간 저장] {len(all_items)}건")

    output_path.write_text(
        json.dumps(all_items, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"저장 완료: {output_path} ({len(all_items)}건)")


def main():
    parser = argparse.ArgumentParser(description="공공데이터 포털 영양성분 데이터 수집")
    parser.add_argument("--max-count", type=int, default=None, help="최대 수집 건수")
    args = parser.parse_args()

    download_all_nutrition(DATA_GO_KR_KEY, args.max_count)


if __name__ == "__main__":
    main()
