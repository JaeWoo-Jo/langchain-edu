"""품목명으로 최신 가격을 검색하는 도구."""

from langchain.tools import tool

from app.agents.tools._es_common import get_es_client, get_price_index


@tool
def search_price(item_name: str) -> str:
    """품목명으로 최신 시세만 빠르게 조회합니다. 가격만 필요할 때 사용.
    예: 쌀, 찹쌀, 콩, 팥, 녹두, 고구마, 감자

    Args:
        item_name: 검색할 품목명
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
        return f"'{item_name}' 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    hits = result.get("hits", {}).get("hits", [])

    if not hits:
        return f"'{item_name}'에 대한 가격 정보를 찾을 수 없습니다."

    lines = []
    for hit in hits:
        src = hit["_source"]
        today = src.get("price_today", 0)
        one_week = src.get("price_1week_ago", 0)
        one_month = src.get("price_1month_ago", 0)

        diff_week = today - one_week if one_week else 0
        diff_month = today - one_month if one_month else 0

        lines.append(
            f"- {src['item_name']} ({src.get('kind_name', '')}): "
            f"오늘 {today:,}원/{src.get('unit', '')} "
            f"(1주전 대비 {diff_week:+,}원, 1개월전 대비 {diff_month:+,}원)"
        )
    return "\n".join(lines)
