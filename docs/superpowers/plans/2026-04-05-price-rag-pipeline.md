# 가격 도메인 RAG 파이프라인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 의료 도메인 RAG 파이프라인을 가격/식재료 도메인으로 전환하여, "자취고수" 에이전트가 레시피·식재료·영양정보 문서를 검색할 수 있게 한다.

**Architecture:** 파이프라인(`agent/pipeline/`)의 config·인덱스 매핑·데이터 수집·적재를 가격 도메인에 맞게 수정한다. 새로운 `json_loader.py`로 API JSON을 Document로 변환하고, 3개의 수집 스크립트(레시피/영양정보/식재료)를 추가한다. 에이전트의 `search_agent.py`에 `rag_search` 노드를 추가하여 새 RAG 인덱스를 병렬 검색한다.

**Tech Stack:** Python, Elasticsearch, OpenAI Embeddings, LangChain, LangGraph, 공공데이터 포털 API (식품안전나라 COOKRCP01, I2790), BeautifulSoup

**Spec:** `docs/superpowers/specs/2026-04-05-price-rag-pipeline-design.md`

**주의:** 파이프라인(데이터 수집/적재) 스크립트는 직접 실행하지 않는다. 코드 작성과 명령어 안내만 한다.

---

## 파일 구조

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `agent/pipeline/config.py` | ES_INDEX → `edu-price-info`, 데이터 경로 상수 추가 |
| `agent/pipeline/index_mapping.py` | `source_type`, `category` 메타데이터 필드 추가 |
| `agent/pipeline/es_client.py` | `source_type`, `category` 메타데이터 적재 |
| `agent/pipeline/main.py` | `--source` 옵션, JSON 파일 지원, 로더 자동 판별 |
| `agent/pipeline/search.py` | 인덱스명 변경 (config.ES_INDEX 사용) |
| `agent/app/agents/tools/_es_common.py` | RAG 인덱스 상수 추가 (기존 클라이언트 재사용) |
| `agent/app/agents/search_agent.py` | `rag_search` 노드 추가 (3-way 병렬 검색) |
| `agent/app/core/config.py` | `ELASTICSEARCH_INDEX_RAG` 설정 추가 |
| `agent/env.sample` | RAG 인덱스 환경변수 추가 |

### 신규 파일

| 파일 | 역할 |
|------|------|
| `agent/pipeline/json_loader.py` | API JSON → LangChain Document 변환 |
| `agent/pipeline/download_recipes.py` | 식품안전나라 COOKRCP01 레시피 API 수집 |
| `agent/pipeline/download_nutrition.py` | 식품안전나라 I2790 영양성분 API 수집 |
| `agent/pipeline/download_ingredients.py` | 농촌진흥청 식재료 가이드 크롤링 |

### 테스트 파일

| 파일 | 대상 |
|------|------|
| `agent/pipeline/tests/test_json_loader.py` | json_loader 단위 테스트 |
| `agent/tests/test_search_agent.py` | rag_search 노드 포함 search_agent 테스트 |

---

## Task 1: config.py + index_mapping.py 수정

**Files:**
- Modify: `agent/pipeline/config.py`
- Modify: `agent/pipeline/index_mapping.py`

- [ ] **Step 1: config.py에서 인덱스명과 데이터 경로 변경**

`agent/pipeline/config.py`에서:

```python
# 변경 전
ES_INDEX = os.getenv("ES_INDEX", "edu-medicine-info")

# 변경 후
ES_INDEX = os.getenv("ES_INDEX", "edu-price-info")

# 추가: 데이터 소스 경로
DATA_DIR = Path(__file__).parent / "data"
RECIPES_DIR = DATA_DIR / "recipes"
NUTRITION_DIR = DATA_DIR / "nutrition"
INGREDIENTS_DIR = DATA_DIR / "ingredients"

# 추가: 공공데이터 API 키
FOOD_SAFETY_API_KEY = os.getenv("FOOD_SAFETY_API_KEY", "")
```

파일 상단에 `from pathlib import Path` 추가.

- [ ] **Step 2: index_mapping.py에 source_type, category 필드 추가**

`agent/pipeline/index_mapping.py`의 INDEX_MAPPING에서 metadata.properties에 추가:

```python
"source_type": {"type": "keyword"},
"category": {"type": "keyword"},
```

- [ ] **Step 3: 커밋**

```bash
git add agent/pipeline/config.py agent/pipeline/index_mapping.py
git commit -m "refactor: 파이프라인 config/매핑을 가격 도메인으로 변경"
```

---

## Task 2: json_loader.py 생성 + 테스트

**Files:**
- Create: `agent/pipeline/json_loader.py`
- Create: `agent/pipeline/tests/test_json_loader.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
# agent/pipeline/tests/test_json_loader.py
import json
from pathlib import Path
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


def test_load_recipe_json_skips_empty():
    """빈 조리법은 건너뛴다."""
    # 빈 MANUAL 필드만 있으면 해당 단계를 제외한다.
    pass  # load_recipe_json 내부에서 빈 MANUAL 필터링 검증
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd agent/pipeline && uv run pytest tests/test_json_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'json_loader'`

- [ ] **Step 3: json_loader.py 구현**

```python
# agent/pipeline/json_loader.py
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

        # 재료
        ingredients = item.get("RCP_PARTS_DTLS", "")
        if ingredients:
            parts.append(f"재료: {ingredients}")

        # 조리법 (MANUAL01 ~ MANUAL20)
        steps = []
        for i in range(1, 21):
            step = item.get(f"MANUAL{i:02d}", "")
            if step and step.strip():
                steps.append(step.strip())
        if steps:
            parts.append("조리법:\n" + "\n".join(steps))

        # 영양정보
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

    식품안전나라 I2790 API 응답 구조 기준.
    각 식품을 하나의 Document로 변환한다.
    """
    file_path = Path(file_path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("row", raw.get("data", [raw]))

    documents: list[Document] = []
    for item in raw:
        name = item.get("DESC_KOR", "")
        if not name:
            continue

        serving = item.get("SERVING_SIZE", "100")
        parts = [f"[{name}] (1회 제공량: {serving}g)"]

        for key, label in [
            ("NUTR_CONT1", "칼로리(kcal)"),
            ("NUTR_CONT2", "탄수화물(g)"),
            ("NUTR_CONT3", "단백질(g)"),
            ("NUTR_CONT4", "지방(g)"),
            ("NUTR_CONT5", "당류(g)"),
            ("NUTR_CONT6", "나트륨(mg)"),
        ]:
            val = item.get(key, "")
            if val:
                parts.append(f"- {label}: {val}")

        category = item.get("GROUP_NAME", "기타")

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
```

- [ ] **Step 4: 테스트 실행 — 성공 확인**

```bash
cd agent/pipeline && uv run pytest tests/test_json_loader.py -v
```

Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add agent/pipeline/json_loader.py agent/pipeline/tests/
git commit -m "feat: JSON → Document 변환 로더 추가 (레시피/영양정보)"
```

---

## Task 3: 데이터 수집 스크립트 — download_recipes.py

**Files:**
- Create: `agent/pipeline/download_recipes.py`

- [ ] **Step 1: download_recipes.py 작성**

식품안전나라 COOKRCP01 API에서 레시피 데이터를 수집하여 `data/recipes/`에 JSON으로 저장한다.

```python
# agent/pipeline/download_recipes.py
"""식품안전나라 COOKRCP01 API에서 레시피 데이터를 수집한다.

사용법:
    uv run python download_recipes.py
    uv run python download_recipes.py --max-count 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from config import FOOD_SAFETY_API_KEY, RECIPES_DIR

# 식품안전나라 조리식품 레시피 DB
API_BASE = "https://openapi.foodsafetykorea.go.kr/api"
SERVICE_ID = "COOKRCP01"
PAGE_SIZE = 100


def fetch_recipes(api_key: str, start: int, end: int) -> list[dict]:
    """레시피 API에서 start~end 범위의 데이터를 가져온다."""
    url = f"{API_BASE}/{api_key}/{SERVICE_ID}/json/{start}/{end}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    service_data = data.get(SERVICE_ID, {})
    result = service_data.get("row", [])
    return result


def get_total_count(api_key: str) -> int:
    """API에서 전체 레시피 수를 조회한다."""
    url = f"{API_BASE}/{api_key}/{SERVICE_ID}/json/1/1"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return int(data.get(SERVICE_ID, {}).get("total_count", "0"))


def download_all_recipes(api_key: str, max_count: int | None = None):
    """전체 레시피를 페이지 단위로 수집하여 JSON 파일로 저장한다."""
    RECIPES_DIR.mkdir(parents=True, exist_ok=True)

    total = get_total_count(api_key)
    if max_count:
        total = min(total, max_count)
    print(f"전체 레시피 수: {total}")

    all_recipes: list[dict] = []
    for start in range(1, total + 1, PAGE_SIZE):
        end = min(start + PAGE_SIZE - 1, total)
        print(f"  수집 중: {start}~{end}")
        try:
            batch = fetch_recipes(api_key, start, end)
            all_recipes.extend(batch)
        except Exception as e:
            print(f"  오류 발생 ({start}~{end}): {e}")
        time.sleep(0.5)

    output_path = RECIPES_DIR / "recipes.json"
    output_path.write_text(
        json.dumps(all_recipes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"저장 완료: {output_path} ({len(all_recipes)}건)")


def main():
    parser = argparse.ArgumentParser(description="식품안전나라 레시피 데이터 수집")
    parser.add_argument("--max-count", type=int, default=None, help="최대 수집 건수")
    args = parser.parse_args()

    if not FOOD_SAFETY_API_KEY:
        print("오류: FOOD_SAFETY_API_KEY가 설정되지 않았습니다.")
        print("  .env 파일에 FOOD_SAFETY_API_KEY=your_key 를 추가하세요.")
        print("  공공데이터 포털(data.go.kr)에서 '조리식품의 레시피 DB' API 키를 발급받으세요.")
        return

    download_all_recipes(FOOD_SAFETY_API_KEY, args.max_count)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 커밋**

```bash
git add agent/pipeline/download_recipes.py
git commit -m "feat: 식품안전나라 레시피 API 수집 스크립트 추가"
```

---

## Task 4: 데이터 수집 스크립트 — download_nutrition.py

**Files:**
- Create: `agent/pipeline/download_nutrition.py`

- [ ] **Step 1: download_nutrition.py 작성**

식품안전나라 I2790 API에서 식품 영양성분 데이터를 수집한다.

```python
# agent/pipeline/download_nutrition.py
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
```

- [ ] **Step 2: 커밋**

```bash
git add agent/pipeline/download_nutrition.py
git commit -m "feat: 식품안전나라 영양성분 API 수집 스크립트 추가"
```

---

## Task 5: 데이터 수집 스크립트 — download_ingredients.py

**Files:**
- Create: `agent/pipeline/download_ingredients.py`

- [ ] **Step 1: download_ingredients.py 작성**

농촌진흥청 농식품올바로 사이트에서 식재료 가이드 PDF를 다운로드한다.
기존 `download_pdfs.py`의 크롤링 패턴을 재활용한다.

```python
# agent/pipeline/download_ingredients.py
"""농촌진흥청 농식품올바로에서 식재료 가이드 문서를 크롤링한다.

사용법:
    uv run python download_ingredients.py
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup

from config import INGREDIENTS_DIR

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 농촌진흥청 농식품올바로
BASE_URL = "https://koreanfood.rda.go.kr"
# 식재료 정보 게시판 목록 URL
LIST_URL = f"{BASE_URL}/kfi/foodSel/foodSelList.do"

SESSION = requests.Session()
SESSION.verify = False
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; edu-pipeline/1.0)"
})


def get_ingredient_pages(max_pages: int = 5) -> list[dict]:
    """식재료 정보 페이지 목록을 수집한다."""
    items: list[dict] = []
    for page_no in range(1, max_pages + 1):
        print(f"  페이지 {page_no} 스캔 중...")
        try:
            resp = SESSION.get(LIST_URL, params={"pageNo": page_no}, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            links = soup.select("a[href*='foodSelView']")
            if not links:
                break

            for link in links:
                href = link.get("href", "")
                title = link.get_text(strip=True)
                if href and title:
                    full_url = BASE_URL + href if href.startswith("/") else href
                    items.append({"title": title, "url": full_url})
        except Exception as e:
            print(f"  페이지 {page_no} 오류: {e}")
        time.sleep(0.5)

    return items


def download_ingredient_page(url: str, title: str, save_dir: Path) -> bool:
    """식재료 상세 페이지의 본문을 텍스트 파일로 저장한다."""
    safe_title = "".join(c if c.isalnum() or c in "가-힣 _-" else "_" for c in title)
    output_path = save_dir / f"{safe_title}.txt"

    if output_path.exists():
        print(f"  [스킵] {title} (이미 존재)")
        return False

    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # 본문 영역 추출 (사이트 구조에 따라 selector 조정 필요)
        content_area = soup.select_one(".view_cont, .cont_area, #contents, article")
        if content_area:
            text = content_area.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        output_path.write_text(f"[{title}]\n\n{text}", encoding="utf-8")
        print(f"  [저장] {title}")
        return True
    except Exception as e:
        print(f"  [오류] {title}: {e}")
        return False


def main():
    INGREDIENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("농촌진흥청 식재료 정보 수집 시작")
    items = get_ingredient_pages(max_pages=5)
    print(f"발견된 식재료 정보: {len(items)}건")

    success = 0
    for item in items:
        if download_ingredient_page(item["url"], item["title"], INGREDIENTS_DIR):
            success += 1
        time.sleep(0.3)

    print(f"\n수집 완료: {success}/{len(items)}건")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 커밋**

```bash
git add agent/pipeline/download_ingredients.py
git commit -m "feat: 농촌진흥청 식재료 가이드 크롤링 스크립트 추가"
```

---

## Task 6: es_client.py 수정 — 메타데이터 확장

**Files:**
- Modify: `agent/pipeline/es_client.py`

- [ ] **Step 1: es_client.py에 source_type, category 메타데이터 추가**

`_generate_actions()` 내부의 `_source.metadata`에 `source_type`과 `category` 필드를 추가한다.

현재 코드:
```python
"metadata": {
    "source": chunk.metadata.get("source", ""),
    "page": chunk.metadata.get("page", 0),
    "chunk_index": chunk.metadata.get("chunk_index", 0),
},
```

변경 후:
```python
"metadata": {
    "source": chunk.metadata.get("source", ""),
    "source_type": chunk.metadata.get("source_type", ""),
    "category": chunk.metadata.get("category", ""),
    "page": chunk.metadata.get("page", 0),
    "chunk_index": chunk.metadata.get("chunk_index", 0),
},
```

- [ ] **Step 2: 커밋**

```bash
git add agent/pipeline/es_client.py
git commit -m "refactor: ES 적재 시 source_type/category 메타데이터 포함"
```

---

## Task 7: main.py 수정 — JSON 지원 + --source 옵션

**Files:**
- Modify: `agent/pipeline/main.py`

- [ ] **Step 1: main.py에 JSON 로더 및 --source 옵션 추가**

주요 변경사항:

1. `SUPPORTED_EXTENSIONS`에 `.json`, `.txt` 추가
2. `load_file()`에 `.json` 분기 추가 (source_type에 따라 `load_recipe_json` 또는 `load_nutrition_json` 호출)
3. `load_file()`에 `.txt` 분기 추가 (식재료 텍스트 파일 → Document 변환)
4. `collect_files()`에 `source` 파라미터 추가
5. CLI에 `--source` 옵션 추가 (`recipes`, `nutrition`, `ingredients`, `all`)

파일 상단 import에 추가:

```python
from langchain_core.documents import Document
from json_loader import load_recipe_json, load_nutrition_json
```

`SUPPORTED_EXTENSIONS` 변경:

```python
SUPPORTED_EXTENSIONS = {".pdf", ".hwp", ".json", ".txt"}
```

`load_file()` 함수를 교체:

```python
def load_file(file_path: Path) -> list:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".hwp":
        return load_hwp(file_path)
    elif ext == ".json":
        # 상위 디렉토리명으로 로더 판별
        if "recipes" in str(file_path):
            return load_recipe_json(file_path)
        elif "nutrition" in str(file_path):
            return load_nutrition_json(file_path)
        else:
            return load_recipe_json(file_path)  # 기본값
    elif ext == ".txt":
        text = file_path.read_text(encoding="utf-8")
        return [Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "source_type": "ingredient",
                "category": "",
                "page": 0,
            },
        )]
    else:
        print(f"지원하지 않는 파일 형식: {ext}")
        return []
```

`collect_files()` 함수에 source 파라미터 추가:

```python
def collect_files(data_dir: str, source: str = "all") -> list[Path]:
    from config import RECIPES_DIR, NUTRITION_DIR, INGREDIENTS_DIR
    if source == "all":
        search_dir = Path(data_dir)
    elif source == "recipes":
        search_dir = RECIPES_DIR
    elif source == "nutrition":
        search_dir = NUTRITION_DIR
    elif source == "ingredients":
        search_dir = INGREDIENTS_DIR
    else:
        search_dir = Path(data_dir)

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(search_dir.rglob(f"*{ext}"))
    return sorted(files)
```

`main()` 함수에서 CLI 옵션 및 호출부 변경:

```python
def main():
    parser = argparse.ArgumentParser(...)
    # 기존 인자들...
    parser.add_argument("--source", default="all",
                        choices=["all", "recipes", "nutrition", "ingredients"],
                        help="처리할 데이터 소스")
    args = parser.parse_args()

    # collect_files 호출에 source 전달
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = collect_files(args.data_dir, source=args.source)

    run_pipeline(files, args.chunk_size, args.chunk_overlap, args.recreate_index)
```

- [ ] **Step 2: 커밋**

```bash
git add agent/pipeline/main.py
git commit -m "feat: 파이프라인 main.py에 JSON 지원 및 --source 옵션 추가"
```

---

## Task 8: search.py 수정 — 인덱스명 참조 변경

**Files:**
- Modify: `agent/pipeline/search.py`

- [ ] **Step 1: search.py가 config.ES_INDEX를 사용하도록 확인**

`search.py`에서 인덱스명이 하드코딩되어 있으면 `config.ES_INDEX`로 변경한다.
이미 config를 import하고 있으면 config.py의 ES_INDEX 변경으로 자동 반영된다.

검색 대상 필드도 확인: `content` 필드는 동일하므로 변경 불필요.

- [ ] **Step 2: 커밋 (변경 있을 경우)**

```bash
git add agent/pipeline/search.py
git commit -m "refactor: search.py 인덱스명을 config.ES_INDEX로 통일"
```

---

## Task 9: config.py + _es_common.py — RAG 인덱스 설정 추가

**Files:**
- Modify: `agent/app/core/config.py`
- Modify: `agent/app/agents/tools/_es_common.py`
- Modify: `agent/env.sample`

- [ ] **Step 1: agent/app/core/config.py에 ELASTICSEARCH_INDEX_RAG 설정 추가**

Settings 클래스의 Elasticsearch 설정 블록에 추가:

```python
ELASTICSEARCH_INDEX_RAG: str = "edu-price-info"
```

- [ ] **Step 2: agent/env.sample에 RAG 인덱스 환경변수 추가**

Elasticsearch 설정 블록에 추가:

```
ELASTICSEARCH_INDEX_RAG=edu-price-info
```

- [ ] **Step 3: _es_common.py에 RAG 인덱스 상수 추가 (별도 클라이언트 불필요, 기존 get_es_client() 재사용)**

```python
# 기존 상수 아래에 추가
RAG_CONTENT_FIELD = "content"

def get_rag_index() -> str:
    """RAG 문서 인덱스명을 반환한다."""
    from app.core.config import settings
    return settings.ELASTICSEARCH_INDEX_RAG
```

- [ ] **Step 4: 커밋**

```bash
git add agent/app/core/config.py agent/app/agents/tools/_es_common.py agent/env.sample
git commit -m "feat: RAG 인덱스 설정 및 _es_common 상수 추가"
```

---

## Task 10: search_agent.py — rag_search 노드 추가 + 테스트

**Files:**
- Modify: `agent/app/agents/search_agent.py`
- Create: `agent/tests/test_search_agent.py`

- [ ] **Step 1: 테스트 작성**

```python
# agent/tests/test_search_agent.py
"""search_agent의 rag_search 노드를 포함한 통합 테스트."""

from app.agents.search_agent import SearchState, rag_search


def test_rag_search_returns_empty_on_no_index():
    """RAG 인덱스가 없거나 연결 실패 시 빈 결과를 반환한다."""
    state: SearchState = {
        "query": "감자 보관법",
        "match_hits": [],
        "multi_hits": [],
        "rag_hits": [],
        "merged_hits": [],
        "result": "",
    }
    result = rag_search(state)
    # ES 연결 실패 시 빈 리스트 반환 (에러 없이)
    assert "rag_hits" in result
    assert isinstance(result["rag_hits"], list)
```

- [ ] **Step 2: SearchState에 rag_hits 필드 추가**

```python
class SearchState(TypedDict):
    query: str
    match_hits: list[dict]
    multi_hits: list[dict]
    rag_hits: list[dict]       # 추가: RAG 문서 검색 결과
    merged_hits: list[dict]
    result: str
```

- [ ] **Step 3: rag_search 노드 함수 추가**

```python
from app.agents.tools._es_common import (
    # 기존 import에 추가
    RAG_CONTENT_FIELD,
    get_es_client,
    get_rag_index,
)

def rag_search(state: SearchState) -> dict:
    """RAG 문서 인덱스(edu-price-info)에서 BM25 검색."""
    es = get_es_client()
    try:
        resp = es.search(
            index=get_rag_index(),
            body={
                "query": {
                    "match": {
                        RAG_CONTENT_FIELD: {
                            "query": state["query"],
                            "operator": "or",
                        }
                    }
                },
                "size": TOP_K,
            },
        )
        hits = resp["hits"]["hits"]
    except Exception:
        hits = []
    return {"rag_hits": hits}
```

- [ ] **Step 4: _build_search_graph에 rag_search 노드를 병렬로 추가**

```python
def _build_search_graph():
    builder = StateGraph(SearchState)

    builder.add_node("match_search", match_search)
    builder.add_node("multi_match_search", multi_match_search)
    builder.add_node("rag_search", rag_search)          # 추가
    builder.add_node("merge_results", merge_results)
    builder.add_node("format_results", format_results)

    # 3-way 병렬 fan-out
    builder.add_edge(START, "match_search")
    builder.add_edge(START, "multi_match_search")
    builder.add_edge(START, "rag_search")                # 추가

    # fan-in: 셋 다 완료 후 merge_results
    builder.add_edge("match_search", "merge_results")
    builder.add_edge("multi_match_search", "merge_results")
    builder.add_edge("rag_search", "merge_results")      # 추가

    builder.add_edge("merge_results", "format_results")
    builder.add_edge("format_results", END)

    return builder.compile()
```

- [ ] **Step 5: merge_results에 rag_hits 병합 추가**

```python
def merge_results(state: SearchState) -> dict:
    """Match + Multi-match + RAG 검색 결과를 병합하고 중복을 제거한다."""
    seen: set[str] = set()
    merged: list[dict] = []
    for hit in state["match_hits"] + state["multi_hits"] + state["rag_hits"]:
        doc_id = hit.get("_id", "")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            merged.append(hit)
    return {"merged_hits": merged[:TOP_K]}
```

- [ ] **Step 6: format_results에서 RAG 결과 포맷팅 추가**

RAG 결과(source_type이 있는 문서)는 가격 데이터와 다른 포맷으로 표시한다.

```python
def format_results(state: SearchState) -> dict:
    hits = state["merged_hits"]
    query = state["query"]

    if not hits:
        return {"result": f"'{query}'에 대한 정보를 찾을 수 없습니다."}

    price_hits = []
    rag_hits = []
    for hit in hits:
        source = hit.get("_source", {})
        if "metadata" in source and source["metadata"].get("source_type"):
            rag_hits.append(hit)
        else:
            price_hits.append(hit)

    parts: list[str] = []

    if price_hits:
        formatted = format_price_hits(price_hits)
        parts.append(f"■ 가격 검색 결과 (상위 {len(price_hits)}건)\n\n{formatted}")

    if rag_hits:
        rag_lines: list[str] = []
        for i, hit in enumerate(rag_hits, 1):
            source = hit["_source"]
            score = hit.get("_score", 0)
            content = source.get("content", "")[:300].replace("\n", " ")
            meta = source.get("metadata", {})
            source_type = meta.get("source_type", "")
            rag_lines.append(f"[{i}] ({source_type}) score={score:.4f}\n{content}")
        parts.append(f"■ 관련 문서 (상위 {len(rag_hits)}건)\n\n" + "\n\n".join(rag_lines))

    return {"result": "\n\n".join(parts)}
```

- [ ] **Step 7: 테스트 실행**

```bash
cd agent && uv run pytest tests/test_search_agent.py -v
```

- [ ] **Step 8: 커밋**

```bash
git add agent/app/agents/search_agent.py agent/app/agents/tools/_es_common.py agent/tests/test_search_agent.py
git commit -m "feat: search_agent에 rag_search 노드 추가 (3-way 병렬 검색)"
```

---

## Task 11: pipeline .env 업데이트

**Files:**
- Modify: `agent/pipeline/.env.example`

- [ ] **Step 1: .env.example에 FOOD_SAFETY_API_KEY 추가**

```
# 기존 설정 유지
ES_URL=https://elasticsearch-edu.didim365.app
ES_USER=elastic
ES_PASSWORD=
ES_INDEX=edu-price-info

OPENAI_API_KEY=
COHERE_API_KEY=

# 식품안전나라 API 키 (공공데이터 포털에서 발급)
FOOD_SAFETY_API_KEY=
```

- [ ] **Step 2: 커밋**

```bash
git add agent/pipeline/.env.example
git commit -m "docs: .env.example에 FOOD_SAFETY_API_KEY 및 인덱스명 업데이트"
```

---

## 실행 순서 안내 (참고용, 직접 실행 X)

파이프라인 구현 완료 후 실행 순서:

```bash
cd agent/pipeline

# 1. 환경 설정
cp .env.example .env
# .env에 실제 API 키 입력

# 2. 데이터 수집
uv run python download_recipes.py              # 레시피 수집
uv run python download_nutrition.py            # 영양성분 수집
uv run python download_ingredients.py          # 식재료 가이드 수집

# 3. 파이프라인 실행 (인덱싱)
uv run python main.py --recreate-index         # 전체 처리
uv run python main.py --source recipes         # 레시피만

# 4. 검색 테스트
uv run python search.py "감자 요리" --mode all
uv run python search.py "단백질 식품" --mode hybrid --rerank
```
