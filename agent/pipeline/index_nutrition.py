"""영양성분 데이터를 전용 ES 인덱스에 적재하는 스크립트.

사용법:
    uv run python index_nutrition.py
    uv run python index_nutrition.py --recreate   # 인덱스 삭제 후 재생성
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from config import ES_URL, ES_USER, ES_PASSWORD, NUTRITION_DIR

NUTRITION_INDEX = "nutrition-info"

NUTRITION_MAPPING = {
    "mappings": {
        "properties": {
            "food_name": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
            "category": {"type": "keyword"},
            "serving_size": {"type": "keyword"},
            "calories": {"type": "float"},
            "protein": {"type": "float"},
            "fat": {"type": "float"},
            "carbs": {"type": "float"},
            "sugar": {"type": "float"},
            "fiber": {"type": "float"},
            "calcium": {"type": "float"},
            "sodium": {"type": "float"},
            "cholesterol": {"type": "float"},
            "source_name": {"type": "keyword"},
            "research_date": {"type": "keyword"},
        }
    },
}


def parse_float(value) -> float | None:
    """문자열 숫자를 float으로 변환. 빈 값이면 None."""
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return None


def transform_record(raw: dict) -> dict:
    """API 원본 레코드를 ES 문서로 변환."""
    return {
        "food_name": raw.get("FOOD_NM_KR", ""),
        "category": raw.get("FOOD_CAT1_NM") or "미분류",
        "serving_size": raw.get("SERVING_SIZE", ""),
        "calories": parse_float(raw.get("AMT_NUM1")),
        "protein": parse_float(raw.get("AMT_NUM3")),
        "fat": parse_float(raw.get("AMT_NUM4")),
        "carbs": parse_float(raw.get("AMT_NUM6")),
        "sugar": parse_float(raw.get("AMT_NUM7")),
        "fiber": parse_float(raw.get("AMT_NUM8")),
        "calcium": parse_float(raw.get("AMT_NUM9")),
        "sodium": parse_float(raw.get("AMT_NUM13")),
        "cholesterol": parse_float(raw.get("AMT_NUM14")),
        "source_name": raw.get("SUB_REF_NAME", ""),
        "research_date": raw.get("RESEARCH_YMD", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="영양성분 데이터를 ES에 적재")
    parser.add_argument("--recreate", action="store_true", help="인덱스 삭제 후 재생성")
    args = parser.parse_args()

    # ES 연결
    es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD), verify_certs=False)
    print(f"ES 연결: {es.info()['version']['number']}")

    # 인덱스 생성
    if es.indices.exists(index=NUTRITION_INDEX):
        if args.recreate:
            es.indices.delete(index=NUTRITION_INDEX)
            print(f"기존 인덱스 '{NUTRITION_INDEX}' 삭제")
        else:
            print(f"인덱스 '{NUTRITION_INDEX}' 이미 존재 (--recreate로 재생성 가능)")

    if not es.indices.exists(index=NUTRITION_INDEX):
        es.indices.create(index=NUTRITION_INDEX, body=NUTRITION_MAPPING)
        print(f"인덱스 '{NUTRITION_INDEX}' 생성 완료")

    # 데이터 로드
    nutrition_file = NUTRITION_DIR / "nutrition.json"
    if not nutrition_file.exists():
        print(f"파일 없음: {nutrition_file}")
        print("먼저 download_nutrition.py를 실행하세요.")
        return

    raw_data = json.loads(nutrition_file.read_text(encoding="utf-8"))
    print(f"로컬 데이터: {len(raw_data)}건")

    # 벌크 적재
    def gen_actions():
        for raw in raw_data:
            doc = transform_record(raw)
            if not doc["food_name"]:
                continue
            yield {
                "_index": NUTRITION_INDEX,
                "_source": doc,
            }

    success, errors = bulk(es, gen_actions(), chunk_size=500, raise_on_error=False)
    es.indices.refresh(index=NUTRITION_INDEX)
    count = es.count(index=NUTRITION_INDEX)["count"]
    print(f"적재 완료: {success}건 성공, {len(errors) if isinstance(errors, list) else 0}건 실패")
    print(f"인덱스 내 총 문서 수: {count}")


if __name__ == "__main__":
    main()
