# 4주차: Deep Agent 아키텍처

## 개요

3주차까지 구현한 "자취고수" 에이전트를 **Deep Agent(다중 에이전트) 아키텍처**로 확장한다.
현재는 하나의 에이전트가 가격 조회, 비교, 차트, RAG 검색을 모두 담당하지만,
4주차에서는 역할별 서브 에이전트로 분리하고 **슈퍼바이저**가 작업을 위임하는 구조로 전환한다.

## 현재 구조 (3주차까지)

```
사용자 질문
    │
    ▼
price_agent (단일 에이전트)
    ├── search (search_agent: 3-way 병렬 검색)
    ├── search_price
    ├── compare_prices
    └── create_price_chart
```

**문제점:**
- 하나의 시스템 프롬프트에 모든 역할을 담아야 해서 복잡
- 도구가 늘어날수록 LLM의 도구 선택 정확도가 떨어짐
- "요즘 싼 재료로 요리 추천해줘" 같은 복합 질문은 단일 에이전트로 처리하기 어려움

## 4주차 목표 구조

```
사용자 질문
    │
    ▼
supervisor_agent (슈퍼바이저)
    ├── price_agent     — 가격 조회/비교/차트
    ├── recipe_agent    — 레시피 추천/조리법 안내
    └── nutrition_agent — 영양 정보/식단 조언
```

**핵심 변경:**
- 슈퍼바이저가 질문 의도를 분석하여 적절한 서브 에이전트에 위임
- 복합 질문("싼 재료로 요리 추천")은 price_agent → recipe_agent 순차 호출
- 각 서브 에이전트는 자신의 도구만 보유하여 도구 선택 정확도 향상

## 요일별 실습 계획

### 월요일: 슈퍼바이저 + 라우팅

**목표:** 슈퍼바이저 에이전트가 질문 의도를 분류하여 서브 에이전트로 라우팅하는 구조 구현

**구현 내용:**
- `app/agents/supervisor.py` 생성 — LangGraph StateGraph 기반 슈퍼바이저
- 의도 분류 노드: 가격/레시피/영양/일반 대화로 분류
- 라우팅 엣지: 분류 결과에 따라 서브 에이전트 노드로 분기
- 기존 `price_agent.py`를 서브 에이전트로 리팩토링 (도구 구성 유지)

**참고 개념:**
```python
# 슈퍼바이저 라우팅 패턴 (LangGraph conditional_edge)
def route(state):
    intent = state["intent"]
    if intent == "price":
        return "price_agent"
    elif intent == "recipe":
        return "recipe_agent"
    elif intent == "nutrition":
        return "nutrition_agent"
    return "general_response"
```

### 화요일: recipe_agent 구현

**목표:** RAG 인덱스(edu-price-info)의 레시피 문서를 검색하여 요리를 추천하는 서브 에이전트

**구현 내용:**
- `app/agents/recipe_agent.py` 생성
- 도구: `search_recipe` — source_type=recipe 필터로 RAG 검색
- 도구: `recommend_by_budget` — 가격 에이전트 결과 + 레시피 매칭
- 프롬프트: 자취생 맞춤 요리 추천 페르소나

### 수요일: nutrition_agent 구현

**목표:** 영양성분 데이터를 기반으로 식단 조언을 제공하는 서브 에이전트

**구현 내용:**
- `app/agents/nutrition_agent.py` 생성
- 도구: `search_nutrition` — source_type=nutrition 필터로 RAG 검색
- 도구: `analyze_nutrition` — 식재료 조합의 영양 균형 분석
- 프롬프트: 건강 관리 조언 페르소나 (자취고수 톤 유지)

### 목요일: 멀티 에이전트 협업 + AgentService 통합

**목표:** 복합 질문을 서브 에이전트 간 협업으로 처리하고, 기존 AgentService에 통합

**구현 내용:**
- 슈퍼바이저에 멀티 에이전트 호출 로직 추가 (순차/병렬)
- "싼 재료로 요리 추천" → price_agent(가격 조회) → recipe_agent(레시피 매칭)
- `agent_service.py` 수정: `create_price_agent` → `create_supervisor_agent`로 교체
- SSE 스트리밍: 서브 에이전트별 진행 상태 전달 (step 이벤트 확장)

### 금요일: 평가 재실행 + 비교 분석

**목표:** 2주차 베이스라인 대비 성능 변화를 측정하고 리포트 작성

**구현 내용:**
- 기존 DeepEval 테스트셋에 복합 질문 케이스 추가
- 동일 평가(test_deepeval.py, test_opik_eval.py) 재실행
- 베이스라인 대비 지표 비교 리포트 작성

**비교 대상 (2주차 베이스라인):**

| 마일스톤 | Task Completion | GEval | 종합 |
|---------|:-:|:-:|:-:|
| 2주차 (베이스라인) | 80% | 70% | 60% |
| 4주차: Deep Agent | — | — | — |

## 신규/수정 파일 목록

### 신규 파일

| 파일 | 역할 |
|------|------|
| `app/agents/supervisor.py` | 슈퍼바이저 에이전트 (의도 분류 + 라우팅) |
| `app/agents/recipe_agent.py` | 레시피 추천 서브 에이전트 |
| `app/agents/nutrition_agent.py` | 영양 정보 서브 에이전트 |
| `app/agents/tools/search_recipe.py` | 레시피 RAG 검색 도구 |
| `app/agents/tools/search_nutrition.py` | 영양성분 RAG 검색 도구 |
| `tests/test_supervisor.py` | 슈퍼바이저 라우팅 테스트 |
| `docs/week4-performance-report.md` | 4주차 성능 비교 리포트 |

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `app/agents/price_agent.py` | 서브 에이전트로 리팩토링 (슈퍼바이저에서 호출) |
| `app/agents/prompts.py` | 서브 에이전트별 프롬프트 추가 |
| `app/agents/tools/_es_common.py` | source_type 필터 검색 유틸리티 추가 |
| `app/services/agent_service.py` | 슈퍼바이저 에이전트로 교체 |
| `tests/test_deepeval.py` | 복합 질문 테스트 케이스 추가 |

## 핵심 학습 포인트

1. **LangGraph 멀티 에이전트 패턴** — supervisor, handoff, subagent-as-tool
2. **의도 분류(Intent Classification)** — LLM 기반 라우팅 vs 규칙 기반 라우팅
3. **에이전트 간 상태 공유** — StateGraph에서 서브 에이전트 결과를 다음 에이전트에 전달
4. **재귀 제한 관리** — DEEPAGENT_RECURSION_LIMIT과 멀티 에이전트 호출 깊이
5. **성능 비교** — 단일 에이전트 vs 멀티 에이전트의 정확도/응답시간 트레이드오프
