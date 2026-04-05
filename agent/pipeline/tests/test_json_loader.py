import sys
from pathlib import Path
# pipeline 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from json_loader import load_recipe_json, load_nutrition_json


def test_load_recipe_json_creates_documents(tmp_path):
    """레시피 JSON 파일을 Document 리스트로 변환한다."""
    recipe_data = [
        {
            "RCP_NM": "김치찌개",
            "RCP_PARTS_DTLS": "김치 200g, 돼지고기 150g, 두부 1/2모",
            "MANUAL01": "1. 김치를 한입 크기로 썬다",
            "MANUAL02": "2. 냄비에 참기름을 두르고 김치를 볶는다",
            "MANUAL03": "",
            "RCP_PAT2": "국류",
            "INFO_ENG": "200",
            "INFO_CAR": "10",
            "INFO_PRO": "15",
            "INFO_FAT": "8",
            "INFO_NA": "800",
        }
    ]
    json_path = tmp_path / "recipes.json"
    json_path.write_text(json.dumps(recipe_data, ensure_ascii=False), encoding="utf-8")

    docs = load_recipe_json(json_path)
    assert len(docs) == 1
    assert "김치찌개" in docs[0].page_content
    assert "김치 200g" in docs[0].page_content
    assert docs[0].metadata["source_type"] == "recipe"
    assert docs[0].metadata["category"] == "국류"


def test_load_nutrition_json_creates_documents(tmp_path):
    """영양성분 JSON 파일을 Document 리스트로 변환한다."""
    nutrition_data = [
        {
            "DESC_KOR": "감자",
            "SERVING_SIZE": "100",
            "NUTR_CONT1": "66",
            "NUTR_CONT2": "15.4",
            "NUTR_CONT3": "2.0",
            "NUTR_CONT4": "0.1",
            "NUTR_CONT5": "0",
            "NUTR_CONT6": "3",
            "GROUP_NAME": "채소류",
        }
    ]
    json_path = tmp_path / "nutrition.json"
    json_path.write_text(json.dumps(nutrition_data, ensure_ascii=False), encoding="utf-8")

    docs = load_nutrition_json(json_path)
    assert len(docs) == 1
    assert "감자" in docs[0].page_content
    assert "66" in docs[0].page_content
    assert docs[0].metadata["source_type"] == "nutrition"
    assert docs[0].metadata["category"] == "채소류"


def test_load_recipe_json_skips_empty_name(tmp_path):
    """RCP_NM이 비어있는 항목은 건너뛴다."""
    data = [{"RCP_NM": "", "RCP_PARTS_DTLS": "test"}]
    json_path = tmp_path / "recipes.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    docs = load_recipe_json(json_path)
    assert len(docs) == 0
