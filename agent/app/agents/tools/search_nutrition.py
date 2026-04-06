"""식품 영양성분을 검색하는 도구."""

from langchain.tools import tool

from app.agents.tools._es_common import get_es_client, get_nutrition_index


@tool
def search_nutrition(food_name: str) -> str:
    """식품의 영양성분(칼로리, 단백질, 지방, 탄수화물 등)을 조회합니다.
    영양 정보나 칼로리가 궁금할 때 사용.
    예: 양파볶음, 김치찌개, 된장국, 삼겹살

    Args:
        food_name: 검색할 식품명
    """
    try:
        client = get_es_client()
    except Exception:
        return "영양성분 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요."

    query = {
        "query": {
            "match": {
                "food_name": {
                    "query": food_name,
                    "operator": "or",
                }
            }
        },
        "size": 5,
    }
    try:
        result = client.search(index=get_nutrition_index(), body=query)
    except Exception:
        return f"'{food_name}' 영양성분 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    hits = result.get("hits", {}).get("hits", [])
    if not hits:
        return f"'{food_name}'에 대한 영양성분 정보를 찾을 수 없습니다."

    lines = []
    for hit in hits:
        src = hit["_source"]
        name = src.get("food_name", "")
        serving = src.get("serving_size", "")
        cal = src.get("calories")
        protein = src.get("protein")
        fat = src.get("fat")
        carbs = src.get("carbs")
        sugar = src.get("sugar")
        fiber = src.get("fiber")
        sodium = src.get("sodium")

        parts = [f"- {name} ({serving})"]
        if cal is not None:
            parts.append(f"  칼로리: {cal}kcal")
        if protein is not None:
            parts.append(f"  단백질: {protein}g")
        if fat is not None:
            parts.append(f"  지방: {fat}g")
        if carbs is not None:
            parts.append(f"  탄수화물: {carbs}g")
        if sugar is not None:
            parts.append(f"  당류: {sugar}g")
        if fiber is not None:
            parts.append(f"  식이섬유: {fiber}g")
        if sodium is not None:
            parts.append(f"  나트륨: {sodium}mg")
        lines.append("\n".join(parts))

    return "\n\n".join(lines)
