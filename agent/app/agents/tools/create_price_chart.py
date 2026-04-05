"""품목의 가격 추이 차트를 생성하는 도구."""

import json

from langchain.tools import tool

from app.agents.tools._es_common import get_es_client, get_price_index


@tool
def create_price_chart(item_name: str) -> str:
    """품목의 가격 추이 차트를 생성합니다. 당일~1년전까지의 가격 변동을 차트로 보여줍니다.

    Args:
        item_name: 차트를 생성할 품목명
    """
    client = get_es_client()
    query = {
        "query": {
            "match": {
                "item_name": item_name
            }
        },
        "sort": [{"date": {"order": "desc"}}],
        "size": 5,
    }
    result = client.search(index=get_price_index(), body=query)
    hits = result.get("hits", {}).get("hits", [])

    if not hits:
        return f"'{item_name}'의 차트 데이터를 찾을 수 없습니다."

    # 첫 번째 품종으로 차트 생성
    src = hits[0]["_source"]
    kind = src.get("kind_name", "")
    categories = ["1년전", "1개월전", "2주전", "1주전", "1일전", "오늘"]
    prices = [
        src.get("price_1year_ago", 0),
        src.get("price_1month_ago", 0),
        src.get("price_2week_ago", 0),
        src.get("price_1week_ago", 0),
        src.get("price_1day_ago", 0),
        src.get("price_today", 0),
    ]

    chart_data = json.dumps({
        "title": {"text": f"{src['item_name']}({kind}) 가격 추이"},
        "chart": {"type": "line"},
        "xAxis": {"categories": categories},
        "yAxis": {"title": {"text": "가격(원)"}},
        "series": [{"name": f"{src['item_name']}({kind})", "data": prices}],
    }, ensure_ascii=False)

    today = src.get("price_today", 0)
    year_ago = src.get("price_1year_ago", 0)
    diff = today - year_ago
    direction = "올랐어" if diff > 0 else "내렸어" if diff < 0 else "그대로야"

    return f"{src['item_name']}({kind}) 1년간 {abs(diff):,}원 {direction}.\n\n[CHART_DATA]{chart_data}[/CHART_DATA]"
