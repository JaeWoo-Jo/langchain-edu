"""품목의 기간별 가격 변동을 비교하는 도구."""

import json

from langchain.tools import tool

from app.agents.tools._es_common import get_es_client, get_price_index


@tool
def compare_prices(item_name: str, period: str = "1주") -> str:
    """품목의 기간별 가격 변동을 비교합니다. 날짜별 가격 데이터를 테이블로 반환합니다.

    Args:
        item_name: 비교할 품목명
        period: 비교 기간 (예: "1주", "2주", "1개월")
    """
    try:
        client = get_es_client()
    except Exception:
        return "가격 데이터 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요."
    query = {
        "query": {
            "match": {
                "item_name": item_name
            }
        },
        "sort": [{"date": {"order": "desc"}}],
        "size": 10,
    }
    try:
        result = client.search(index=get_price_index(), body=query)
    except Exception:
        return f"'{item_name}' 비교 데이터 조회 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    hits = result.get("hits", {}).get("hits", [])

    if not hits:
        return f"'{item_name}'의 가격 데이터를 찾을 수 없습니다."

    # 각 품종별로 기간 가격 비교 테이블 구성
    names = []
    units = []
    today_prices = []
    day1_prices = []
    week1_prices = []
    week2_prices = []
    month1_prices = []

    for hit in hits:
        src = hit["_source"]
        names.append(f"{src['item_name']}({src.get('kind_name', '')})")
        units.append(src.get("unit", ""))
        today_prices.append(str(src.get("price_today", 0)))
        day1_prices.append(str(src.get("price_1day_ago", 0)))
        week1_prices.append(str(src.get("price_1week_ago", 0)))
        week2_prices.append(str(src.get("price_2week_ago", 0)))
        month1_prices.append(str(src.get("price_1month_ago", 0)))

    table_data = json.dumps({
        "dataTable": {
            "columns": {
                "품목": names,
                "단위": units,
                "오늘": today_prices,
                "1일전": day1_prices,
                "1주전": week1_prices,
                "2주전": week2_prices,
                "1개월전": month1_prices,
            }
        }
    }, ensure_ascii=False)

    # 요약 텍스트
    if hits:
        src = hits[0]["_source"]
        today = src.get("price_today", 0)
        week_ago = src.get("price_1week_ago", 0)
        diff = today - week_ago
        direction = "올랐어" if diff > 0 else "내렸어" if diff < 0 else "그대로야"
        summary = f"{item_name} 1주 전 대비 {abs(diff):,}원 {direction}."
    else:
        summary = f"{item_name} 가격 데이터입니다."

    return f"{summary}\n\n[TABLE_DATA]{table_data}[/TABLE_DATA]"
