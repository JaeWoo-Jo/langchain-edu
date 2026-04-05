# 3주차 품질 개선 기록

## 개선 1: 프롬프트 최적화

**파일:** `app/agents/prompts.py`

**문제:**
- "검색어를 바꾸어서 검색한다" 규칙이 불필요한 반복 호출을 유도
- 도구 6회 제한이 너무 관대
- `search`와 `search_price` 구분 기준 없음
- 응답에 날짜/단위/출처 정보 누락 (2주차 리포트 완결성 3/5)

**변경 내용:**
1. 도구 설명에 용도별 선택 가이드 추가 (search_price=단일 시세, search=통합 검색)
2. "검색어를 바꿔라" 삭제 → "첫 결과가 충분하면 바로 답변"
3. 도구 호출 제한 6회 → 3회
4. 응답 규칙 추가: 단위, 시점 정보, 비교 기준 명시

**효과 (감자 얼마야? 테스트):**

| | 변경 전 | 변경 후 |
|---|---|---|
| 도구 호출 | 5회 | 1회 |
| 호출 도구 | search_price → search → search_price → search → compare_prices | search_price |
| 응답 시간 | ~6초 | ~2초 (추정) |
| API 토큰 | 3,711 (마지막 호출 기준) | ~900 (추정) |

---

### 개선 1-b: PLANNER_PROMPT 최적화

**파일:** `app/agents/prompts.py` (PLANNER_PROMPT)

**문제:** "고구마 가격 비교해줘"에서 compare_prices 2번 + 불필요한 create_price_chart까지 호출 (4회)
- "검색 → 비교 → 차트 순서가 자연스러움" 가이드가 비교 요청에도 차트까지 계획하도록 유도
- 최대 6단계 이내 제한이 너무 관대

**변경 내용:**
1. 도구 설명을 메인 프롬프트와 일관되게 수정 (search_price=빠른 조회, search=통합 검색)
2. 최대 단계 6 → 3으로 축소
3. "같은 도구를 2번 이상 호출하지 않는다" 규칙 추가
4. "사용자가 명시적으로 요청한 것만 계획에 포함" 규칙 추가
5. "검색 → 비교 → 차트 순서가 자연스러움" 삭제

**효과:**

| | 변경 전 | 변경 후 |
|---|---|---|
| "고구마 가격 비교해줘" | 4회 (search_price, compare_prices×2, create_price_chart) | **1회** (compare_prices) |

---

## 개선 2: 검색 품질 (score=0 + multi_match 노이즈 + 중복)

**파일:** `app/agents/search_agent.py`

**문제:**
- match_search, multi_match_search가 `sort: date desc`를 사용하여 `_score=0`
- multi_match에서 관련 없는 품목이 섞임 (감자 검색 → 찹쌀/고구마 포함)
- merge_results가 `_id`로만 중복 제거 → 같은 품목이 날짜만 다르게 중복

**변경 내용:**
- match_search: `sort` 제거 → `_score` 기반 정렬
- multi_match_search: `minimum_should_match: "75%"` 추가
- merge_results: `item_name + kind_name` 기준 중복 제거, 최신 건 유지

**효과 (감자 검색 테스트):**

| | 변경 전 | 변경 후 |
|---|---|---|
| 가격 score | 0.0000 | 1.0609 (관련성 점수 정상) |
| 가격 중복 | 감자 4건 (노지 2 + 시설 2) | 2건 (노지 1 + 시설 1) |
| multi_match 노이즈 | 찹쌀, 고구마 섞임 | 감자만 출력 |
| RAG 결과 | 감자 레시피 정상 | 동일 (8건) |

---

## 개선 3: 도구 설명 명확화

**파일:** `app/agents/tools/search_price.py`, `app/agents/search_agent.py`

**문제:** search와 search_price의 docstring이 모호하여 LLM이 둘 다 호출

**변경 내용:**
- `search_price` 설명: "품목명으로 최신 가격을 검색합니다" → "품목명으로 최신 시세만 빠르게 조회합니다. 가격만 필요할 때 사용."
- `search` 설명: "가격 정보를 통합 검색하여..." → "가격 + 레시피·영양·식재료 문서를 함께 검색합니다. 요리 추천이나 식재료 정보가 필요할 때 사용."

**효과:**

| 질문 | 변경 전 도구 호출 | 변경 후 도구 호출 |
|------|------------------|------------------|
| "쌀 얼마야?" | search_price → search → search_price → search → compare_prices (5회) | search → search_price (2회) |
| "감자로 뭐 해먹지?" | - | search → search (2회, 정상) |

가격만 묻는 질문에서 search가 여전히 1회 호출되나, 총 호출 수는 5회→2회로 개선됨.
완전 분리는 LLM 판단에 의존하므로 프롬프트 추가 튜닝 또는 도구 통합이 필요할 수 있음.

---

## 개선 4: Opik 평가 데이터셋 업데이트

**파일:** `tests/test_opik_eval.py`

**문제:**
- 데이터셋이 3주차 변경(search 서브에이전트, RAG 검색)을 반영하지 않음
- `expected_tool`에 `search`가 없음
- 레시피/영양/식재료 관련 질문 케이스 없음
- "오늘 뭐 해먹지?"가 일반대화로 분류됨 (실제로는 search 도구 사용)

**변경 내용:**
1. 통합검색(search) 케이스 16개 추가 (레시피·영양·식재료 질문)
2. `ToolUsageMetric.ALL_TOOLS`에 `search` 추가
3. `ResponseCompletenessMetric`에 통합검색 카테고리 채점 기준 추가
4. "오늘 뭐 해먹지?"를 일반대화 → 통합검색으로 재분류
5. 데이터셋 100개 → 115개

**데이터셋 분포:**

| 카테고리 | 변경 전 | 변경 후 |
|---------|--------|--------|
| 가격검색 | 35 | 35 |
| 가격비교 | 35 | 35 |
| 차트생성 | 21 | 21 |
| 통합검색 | 0 | **16** |
| 복합질문 | 5 | 5 |
| 일반대화 | 3 | 2 |
| 에지케이스 | 1 | 1 |
| **합계** | **100** | **115** |

---

## 개선 5: DeepEval 데이터셋 업데이트 (보조)

**파일:** `tests/test_deepeval.py`

**문제:**
- RAG 관련 질문(레시피/영양/식재료) 테스트 없음
- `search` 도구 expected_tools 케이스 없음

**변경 내용:**
1. `RAG_CASES` 카테고리 추가 (3개): 레시피/영양/식재료 질문
2. `korean_quality_rag` GEval 메트릭 추가 (RAG 응답 품질 채점)
3. main()에 RAG 채점 단계 추가
4. 데이터셋 10개 → 13개

**데이터셋 분포:**

| 카테고리 | 변경 전 | 변경 후 |
|---------|--------|--------|
| 가격 (PRICE_CASES) | 8 | 8 |
| RAG (RAG_CASES) | 0 | **3** |
| 일반 (GENERAL_CASES) | 2 | 2 |
| **합계** | **10** | **13** |

---

## 개선 6: generate_report.py 범용화

**파일:** `tests/generate_report.py`

**문제:**
- 리포트 제목/출력 파일명이 "2주차" 하드코딩
- 주차 변경 시 코드 수정 필요

**변경 내용:**
1. `--week` CLI 인수 추가 (기본값: 3)
2. 리포트 제목: `{week}주차 에이전트 성능 진단 리포트`로 동적 생성
3. 출력 파일: `docs/week{week}-performance-report.md`로 동적 생성
4. 부록 명령어에서 데이터셋 수 업데이트 (100→115, 10→13)

**사용법:**
```bash
uv run python tests/generate_report.py           # 3주차 (기본)
uv run python tests/generate_report.py --week 4   # 4주차
```

---

## 개선 7: RAG 하이브리드 검색 (BM25 + kNN)

**파일:** `app/agents/search_agent.py`, `app/agents/tools/_es_common.py`

**문제:** `rag_search`가 BM25(텍스트 매칭)만 사용하여, 검색어와 동일 단어가 없는 문서를 놓침.
예: "감자 요리"로 검색 시 "감자"가 본문에 없는 레시피는 검색 불가.

**변경 내용:**
1. `_es_common.py`에 `RAG_VECTOR_FIELD` 상수 + `get_embeddings()` 싱글턴 추가
2. `rag_search` 노드: BM25 `query` + kNN `knn` 동시 전송 (ES 내부 score 합산)

**효과 ("감자 요리" 검색):**

| | 변경 전 (BM25만) | 변경 후 (하이브리드) |
|---|---|---|
| 상위 결과 | 감자가 본문에 있는 레시피만 | + 의미적으로 유사한 "요리" 레시피도 포함 |
| 신규 매칭 | - | 케이준 스타일 닭고기 요리, 비트와 호두 요리 등 |

---

## 개선 8: test_search_agent.py 작성

**파일:** `tests/test_search_agent.py` (신규)

**문제:** 3주차 계획에 있었지만 미작성. search_agent의 핵심 로직에 대한 테스트 없음.

**테스트 구성:**
- `TestMergeResults` (4개): 가격 중복 제거, RAG 중복 제거, 혼합 병합, TOP_K 제한
- `TestFormatResults` (3개): 빈 결과, 가격 포맷, RAG 포맷
- `TestSearchGraphIntegration` (2개): 통합 테스트 (ES 연결 필요, `@pytest.mark.integration`)

**결과:** 단위 테스트 7/7 PASS

---

## 개선 9: 도구 에러 핸들링

**파일:** `app/agents/tools/search_price.py`, `compare_prices.py`, `create_price_chart.py`

**문제:** ES 연결 실패나 쿼리 에러 시 예외가 그대로 전파되어 사용자에게 "처리 중 오류가 발생했습니다"만 표시. 원인 파악 불가.

**변경 내용:**
- ES 클라이언트 생성 실패 → "가격 데이터 서버에 연결할 수 없습니다" 반환
- 검색 쿼리 실패 → "'{품목명}' 검색 중 오류가 발생했습니다" 반환
- 3개 도구 모두 동일 패턴 적용

---

## 개선 10: Reflector replan 무한 루프 수정

**파일:** `app/agents/prompts.py` (REFLECTOR_PROMPT)

**문제:** "시설 감자 들어간 요리" 같은 질문에서 reflector가 "레시피 정보 불충분 → replan"을 반복하여 무한 루프에 빠짐.
검색 결과에 감자 레시피가 있지만 "시설 감자"라는 정확한 표현이 없어서 reflector가 계속 불충분으로 판단.
Opik에서 에러로 기록됨.

**원인:** REFLECTOR_PROMPT에 replan 판단 기준이 너무 관대함. 부분적 결과가 있어도 replan을 시도.

**변경 내용:**
REFLECTOR_PROMPT에 규칙 추가:
- 부분적으로라도 유용한 정보가 있으면 done으로 판단
- 완벽하지 않아도 있는 정보로 답변할 수 있으면 done
- replan은 결과가 완전히 비어있거나 오류일 때만 사용

**효과 ("시설 감자 들어간 요리" 테스트):**

| | 수정 전 | 수정 후 |
|---|---|---|
| reflect 횟수 | 5회 (replan 루프) | 1회 |
| tool 호출 | 2회 (같은 search 반복) | 1회 |
| 응답 시간 | ~45초 | ~3초 |
| Opik 상태 | 에러 | 정상 |

---

## 개선 11: Reflector replan 무한 루프 근본 해결

**���일:** `app/agents/prompts.py`, `app/agents/deep_agent.py`

**문제:** 개선 10에서 REFLECTOR_PROMPT를 수정했지만 여전히 다양한 질문에서 replan 루프 발생.
19개 질문 테스트 중 8개 ERROR (reflect 3~6회, 응답 13~24초).

**에러 패턴:**
- 레시피 상세 요청 ("감자전 만드는 법") → 레시피가 있지만 "구체적 조리법 부족" 판단
- 복��� 비교+추천 → 각 단계마다 replan
- 모호한 질문 ("뭐 먹지") → "사용자 의도 불명확" 판단
- 도메인 외 ("김치찌개 맛집") → 맛집 정보 없어서 계속 replan

**수정 1: REFLECTOR_PROMPT 전�� 재작성**
- "1건이라도 데이터가 있으면 → done" 규칙 명시
- "도메인 밖이면 → done" 규칙 명시
- replan 조건을 "결과가 완전히 비어있고(0건)" 경우로 극도로 제한

**수정 2: 코드 안전장치 (deep_agent.py)**
- `MAX_REPLAN = 1` 상수 추가
- `replan_count` 상태 필드 추가
- reflector에서 replan 시 `replan_count` 체크 → 한도 초과 시 강제 done

**효과 (이전 에러 10개 질문 재테스트):**

| 질문 | 수정 전 | 수정 후 |
|------|---------|---------|
| 유기농 쌀로 밥 짓는 법 | ERROR 22s reflect=4 | OK 8s reflect=1 |
| 쌀이랑 감자 중 싼 걸로 추천 | ERROR 19s reflect=5 | OK 7s reflect=1 |
| 뭐 먹지 | ERROR 13s reflect=3 | OK 5s reflect=1 |
| 배고파 | ERROR 14s reflect=3 | OK 9s reflect=1 |
| 감자전 만드는 법 | ERROR 18s reflect=4 | OK 4s reflect=1 |
| 콩나물국 레시피 | ERROR 24s reflect=3 | OK 12s reflect=2 |
| 맛있는거 | ERROR 18s reflect=3 | OK 5s reflect=1 |
| 고구마 가격이랑 요리법 | ERROR 21s reflect=6 | OK 9s reflect=2 |
| 김치찌개 맛집 | ERROR 18s reflect=4 | OK 6s reflect=1 |
| 시설 감자 들어간 요리 | ERROR 45s reflect=5 | OK 6s reflect=1 |
| **결과** | **10/10 ERROR** | **10/10 OK** |
