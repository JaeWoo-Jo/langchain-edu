"""공공데이터 참가격 생필품 가격정보 → Elasticsearch 적재 스크립트

KAMIS(농산물유통정보) API를 호출하여 생필품 가격 데이터를 수집하고
Elasticsearch에 적재합니다.

사용법:
    cd agent
    uv run python scripts/collect_prices.py --date 2026-03-11
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from app.core.config import settings
from app.utils.opensearch_client import get_elasticsearch_client

INDEX_NAME = "prices-daily-goods"
API_URL = "https://www.kamis.or.kr/service/price/xml.do"


def fetch_prices(date: str):
    """KAMIS API에서 소매 가격 정보를 조회합니다."""
    params = {
        "action": "dailyPriceByCategoryList",
        "p_product_cls_code": "01",  # 01=소매
        "p_country_code": "1101",    # 서울
        "p_regday": date,
        "p_convert_kg_yn": "N",
        "p_cert_key": settings.PUBLIC_DATA_API_KEY,
        "p_cert_id": "didim365",
        "p_returntype": "json",
    }
    resp = httpx.get(API_URL, params=params, verify=False, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def parse_price(price_str: str) -> int:
    """가격 문자열에서 숫자만 추출합니다. 예: '62,756' → 62756"""
    if not price_str or price_str == "-":
        return 0
    return int(price_str.replace(",", ""))


def create_index_if_not_exists(client):
    """인덱스가 없으면 생성합니다."""
    if not client.indices.exists(index=INDEX_NAME):
        mapping = {
            "mappings": {
                "properties": {
                    "item_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "item_code": {"type": "keyword"},
                    "kind_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "rank": {"type": "keyword"},
                    "unit": {"type": "keyword"},
                    "price_today": {"type": "integer"},
                    "price_1day_ago": {"type": "integer"},
                    "price_1week_ago": {"type": "integer"},
                    "price_2week_ago": {"type": "integer"},
                    "price_1month_ago": {"type": "integer"},
                    "price_1year_ago": {"type": "integer"},
                    "price_avg_year": {"type": "integer"},
                    "date": {"type": "date", "format": "yyyy-MM-dd"},
                }
            }
        }
        client.indices.create(index=INDEX_NAME, body=mapping)
        print(f"인덱스 '{INDEX_NAME}' 생성 완료")


def index_prices(client, items: list, date: str):
    """가격 데이터를 Elasticsearch에 적재합니다."""
    count = 0
    for item in items:
        doc = {
            "item_name": item.get("item_name", ""),
            "item_code": item.get("item_code", ""),
            "kind_name": item.get("kind_name", ""),
            "rank": item.get("rank", ""),
            "unit": item.get("unit", ""),
            "price_today": parse_price(item.get("dpr1", "")),
            "price_1day_ago": parse_price(item.get("dpr2", "")),
            "price_1week_ago": parse_price(item.get("dpr3", "")),
            "price_2week_ago": parse_price(item.get("dpr4", "")),
            "price_1month_ago": parse_price(item.get("dpr5", "")),
            "price_1year_ago": parse_price(item.get("dpr6", "")),
            "price_avg_year": parse_price(item.get("dpr7", "")),
            "date": date,
        }
        client.index(index=INDEX_NAME, body=doc)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="참가격 생필품 가격 데이터 수집")
    parser.add_argument("--date", default="2026-03-11", help="조회 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"[수집 시작] 날짜: {args.date}")

    client = get_elasticsearch_client()
    create_index_if_not_exists(client)

    data = fetch_prices(args.date)
    items = data.get("data", {}).get("item", [])

    if not items or not isinstance(items, list):
        print("수집할 데이터가 없습니다.")
        print(f"API 응답: {data}")
        return

    count = index_prices(client, items, args.date)
    print(f"[수집 완료] {args.date} 데이터 {count}건 적재")

    # 적재 확인
    result = client.count(index=INDEX_NAME)
    print(f"[인덱스 현황] 총 {result['count']}건")


if __name__ == "__main__":
    main()
