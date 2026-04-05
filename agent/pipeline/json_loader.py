"""공공 API JSON 데이터를 LangChain Document로 변환하는 로더."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def load_recipe_json(file_path: str | Path) -> list[Document]:
    """레시피 JSON 파일을 Document 리스트로 변환한다.

    식품안전나라 COOKRCP01 API 응답 구조 기준.
    각 레시피를 하나의 Document로 변환하며, 레시피명·재료·조리법·영양정보를
    하나의 page_content에 합친다.
    """
    file_path = Path(file_path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("row", raw.get("data", [raw]))

    documents: list[Document] = []
    for item in raw:
        name = item.get("RCP_NM", "")
        if not name:
            continue

        parts = [f"[{name}]"]

        ingredients = item.get("RCP_PARTS_DTLS", "")
        if ingredients:
            parts.append(f"재료: {ingredients}")

        steps = []
        for i in range(1, 21):
            step = item.get(f"MANUAL{i:02d}", "")
            if step and step.strip():
                steps.append(step.strip())
        if steps:
            parts.append("조리법:\n" + "\n".join(steps))

        nutrition_parts = []
        for key, label in [
            ("INFO_ENG", "칼로리(kcal)"),
            ("INFO_CAR", "탄수화물(g)"),
            ("INFO_PRO", "단백질(g)"),
            ("INFO_FAT", "지방(g)"),
            ("INFO_NA", "나트륨(mg)"),
        ]:
            val = item.get(key, "")
            if val:
                nutrition_parts.append(f"{label}: {val}")
        if nutrition_parts:
            parts.append("영양정보: " + ", ".join(nutrition_parts))

        category = item.get("RCP_PAT2", "기타")

        documents.append(Document(
            page_content="\n".join(parts),
            metadata={
                "source": f"recipe_api/{name}",
                "source_type": "recipe",
                "category": category,
                "page": 0,
                "chunk_index": 0,
            },
        ))

    return documents


def load_nutrition_json(file_path: str | Path) -> list[Document]:
    """영양성분 JSON 파일을 Document 리스트로 변환한다.

    공공데이터 포털 FoodNtrCpntDbInfo02 API 응답 구조 기준.
    """
    file_path = Path(file_path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("items", raw.get("row", raw.get("data", [raw])))

    documents: list[Document] = []
    for item in raw:
        # 공공데이터 포털 필드명 우선, 식품안전나라 필드명 fallback
        name = item.get("FOOD_NM_KR", "") or item.get("DESC_KOR", "")
        if not name:
            continue

        serving = item.get("SERVING_SIZE", "100")
        parts = [f"[{name}] (1회 제공량: {serving})"]

        for key, fallback_key, label in [
            ("AMT_NUM1", "NUTR_CONT1", "칼로리(kcal)"),
            ("AMT_NUM6", "NUTR_CONT2", "탄수화물(g)"),
            ("AMT_NUM3", "NUTR_CONT3", "단백질(g)"),
            ("AMT_NUM4", "NUTR_CONT4", "지방(g)"),
            ("AMT_NUM7", "NUTR_CONT5", "당류(g)"),
            ("AMT_NUM13", "NUTR_CONT6", "나트륨(mg)"),
        ]:
            val = item.get(key, "") or item.get(fallback_key, "")
            if val:
                parts.append(f"- {label}: {val}")

        category = item.get("FOOD_CAT1_NM", "") or item.get("GROUP_NAME", "기타")

        documents.append(Document(
            page_content="\n".join(parts),
            metadata={
                "source": f"nutrition_api/{name}",
                "source_type": "nutrition",
                "category": category,
                "page": 0,
                "chunk_index": 0,
            },
        ))

    return documents
