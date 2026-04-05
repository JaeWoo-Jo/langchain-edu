# 가격 도메인 RAG 파이프라인 설계

## 개요

기존 의료 도메인 RAG 파이프라인(`agent/pipeline/`)을 가격/식재료 도메인에 맞게 수정한다.
"자취고수" 에이전트가 레시피, 식재료 보관법, 영양정보를 검색하여 답변에 활용할 수 있도록 한다.

## 데이터 소스

| 소스 | 수집 방식 | 수집 내용 | 저장 경로 |
|------|----------|----------|----------|
| 한국식품연구원 레시피 DB | 공공데이터 포털 API | 레시피명, 재료, 조리법, 영양정보 | `data/recipes/` |
| 식약처 식품영양성분 DB | 공공데이터 포털 API | 식품별 칼로리, 탄수화물, 단백질 등 | `data/nutrition/` |
| 농촌진흥청 농식품올바로 | 웹 크롤링 + PDF | 식재료 보관법, 손질법, 제철 정보 | `data/ingredients/` |

## ES 인덱스 설계

인덱스명: `edu-price-info`

```json
{
  "mappings": {
    "properties": {
      "content": { "type": "text", "analyzer": "standard" },
      "content_vector": { "type": "dense_vector", "dims": 1536, "index": true, "similarity": "cosine" },
      "metadata": {
        "properties": {
          "source": { "type": "keyword" },
          "source_type": { "type": "keyword" },
          "category": { "type": "keyword" },
          "page": { "type": "integer" },
          "chunk_index": { "type": "integer" }
        }
      }
    }
  }
}
```

- `source_type`: `recipe`, `ingredient`, `nutrition`
- `category`: `채소류`, `곡류`, `육류`, `국류`, `반찬` 등

## 데이터 처리 흐름

```
recipes/*.json      → json_loader  ─┐
nutrition/*.json    → json_loader   ├→ chunker (500자/100 overlap) → embedder (text-embedding-3-small) → ES
ingredients/*.pdf   → pdf_loader   ─┘
```

- `json_loader.py` (신규): API JSON → LangChain Document 변환
- `pdf_loader.py`, `hwp_loader.py`, `chunker.py`, `embedder.py`: 기존 코드 재활용

## search_agent 연동

search_agent.py에 `rag_search` 노드를 추가하여 `edu-price-info` 인덱스를 병렬 검색한다.

```
match_search (prices-daily-goods)       ──┐
multi_match_search (prices-daily-goods)  ├→ merge → format
rag_search (edu-price-info)             ──┘
```

## 변경 파일

### 파이프라인 (agent/pipeline/)

| 파일 | 작업 |
|------|------|
| `config.py` | 수정 — 인덱스명, 데이터 경로 변경 |
| `index_mapping.py` | 수정 — source_type, category 필드 추가 |
| `main.py` | 수정 — --source 옵션, JSON/PDF 자동 판별 |
| `json_loader.py` | 신규 — JSON → Document 로더 |
| `download_recipes.py` | 신규 — 레시피 API 수집 |
| `download_nutrition.py` | 신규 — 영양성분 API 수집 |
| `download_ingredients.py` | 신규 — 식재료 가이드 크롤링 |
| `es_client.py` | 수정 — 카테고리 메타데이터 적재 |
| `search.py` | 수정 — 인덱스명 변경 |

### 에이전트 (agent/app/agents/)

| 파일 | 작업 |
|------|------|
| `search_agent.py` | 수정 — rag_search 노드 추가 |
| `tools/_es_common.py` | 수정 — RAG 인덱스 상수 추가 |

## CLI 사용법

```bash
# 데이터 수집
uv run python download_recipes.py
uv run python download_nutrition.py
uv run python download_ingredients.py

# 파이프라인 실행
uv run python main.py                    # 전체
uv run python main.py --source recipes   # 레시피만
uv run python main.py --recreate-index   # 인덱스 재생성

# 검색 테스트
uv run python search.py "감자 요리" --mode all
```
