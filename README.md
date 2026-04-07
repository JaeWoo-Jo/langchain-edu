# 자취고수 - 생필품 가격 에이전트

LangChain + LangGraph 기반 AI 에이전트 프로젝트. 공공데이터 포털의 생필품 가격정보를 Elasticsearch에 적재하고, "자취고수" 페르소나의 에이전트가 가격 조회/비교/차트/레시피 추천까지 해주는 챗봇입니다.

```
사용자: 쌀 가격 알려줘
자취고수: 쌀 20kg 한 포대는 오늘 62,038원인데, 1주 전보다 250원 내려갔고
         한 달 전보다는 12,622원 올랐어. 20kg이 1kg당 약 3,102원으로
         10kg짜리보다 경제적이야 ㅋㅋ

사용자: 감자로 뭐 해먹을 수 있어?
자취고수: 감자 볶음밥, 된장국, 함박스테이크, 감자 냉채 등 다양하게 가능해!
         [레시피 상세 안내]
```

## 데이터 현황

| 인덱스 | 건수 | 소스 | 용도 |
|--------|------|------|------|
| `prices-daily-goods` | 175건 | KAMIS 공공데이터 API (6개 카테고리) | search_price, compare_prices, create_price_chart |
| `edu-recipe-rag` | 1,443건 | 식품안전나라 COOKRCP01 API (1,146 레시피) | search 서브에이전트 (BM25 + kNN 하이브리드 RAG) |
| `nutrition-info` | 75,200건 | 식품안전나라 영양성분 API | search_nutrition |

**가격 데이터 카테고리**: 식량작물(쌀, 감자 등), 채소류(당근, 양파 등 30종), 과일류(사과, 바나나 등), 축산물(소, 돼지, 닭, 계란), 수산물(고등어, 갈치 등), 특용작물(참깨, 버섯 등)

---

### 아키텍처 진화 과정

4주간 3단계에 걸쳐 에이전트 아키텍처를 발전시켰습니다:

```
2주차                    3주차                      4주차
LangChain ReAct    →    LangChain + LangGraph   →   LangGraph Deep Agent
(단일 에이전트)          (서브에이전트 도입)        (Plan-Execute-Reflect)

┌────────────┐       ┌──────────┐                ┌──────────────────────┐
│ ReAct      │       │ price    │                │ Planner → Executor   │
│ Agent      │       │ _agent   │                │    ↕         ↕       │
│            │       │    │     │                │ Reflector  Tools     │
│ all tools  │       │  search  │(sub-agent)     │    ↓                 │
│ (direct)   │       │  _agent  │(3-way)         │ Synthesizer          │
└────────────┘       └──────────┘                └──────────────────────┘

문제:                    개선:                      개선:
· 도구 5~6회 반복        · 검색 3종 병렬 실행       · 계획 수립 후 단계별 실행
· 불필요한 차트 생성     · 중복 제거 + 노이즈 제거  · 자체 평가로 자가 교정
· 복합 질문 처리 불가    · BM25 + kNN 하이브리드    · 구조화 출력으로 안정적 분기
                                                    · MAX_REPLAN=1 무한 루프 방지
```

---

## 성능 비교 — Before & After

4주간 에이전트를 개선하며 동일한 평가 체계(DeepEval)로 성능을 추적했습니다.

### 전체 성능 변화

| 메트릭 | 2주차 (베이스라인) | 4주차 (최종) | 변화 |
|--------|:-:|:-:|:-:|
| Task Completion | 80% | **92.3%** | +12.3%p |
| GEval (한국어 품질) | 70% | **92.3%** | +22.3%p |
| 종합 Pass Rate | 70% | **84.6%** | +14.6%p |

### 카테고리별 상세 (4주차)

| 카테고리 | Task Completion | GEval | Pass Rate |
|---------|:-:|:-:|:-:|
| 가격 질문 (8건) | 87.5% | 87.5% | 75% |
| RAG 통합검색 (3건) | 100% | 100% | 100% |
| 일반 대화 (2건) | 100% | 100% | 100% |

### 주요 케이스 개선

| 입력 | 2주차 GEval | 4주차 GEval | 원인 |
|------|:-:|:-:|------|
| "안녕!" | 0.26 | **0.81** | 카테고리별 GEval 분리 |
| "존재하지않는품목 가격 알려줘" | 0.21 | **0.89** | 카테고리별 GEval 분리 |
| "감자 얼마야?" | 에러 | **통과** | 도구 에러 핸들링 추가 |

### 수치 해석 시 주의점

성능 향상은 **에이전트 자체 개선**과 **평가 방법론 개선** 두 가지가 섞여 있습니다.

| 구분 | 내용 | 예시 |
|------|------|------|
| 에이전트 개선 | 프롬프트 최적화, 검색 품질, 에러 핸들링 | "감자 얼마야?" 에러→통과, 도구 호출 5→1회 |
| 평가 방법론 개선 | GEval 카테고리 분리, 데이터셋 확장 | "안녕!" 0.26→0.81 (채점 기준이 공정해진 것) |

"안녕!"의 GEval이 0.26→0.81로 오른 것은 에이전트가 나아진 것이 아니라, 인사에 "가격 정보 포함 여부"를 묻던 부당한 채점 기준을 분리한 결과입니다. ~~"쌀이랑 콩 가격 비교해줘"는 4주차에서도 Task Completion 0.6으로 여전히 실패하며, 복수 품목 동시 조회는 남은 과제입니다.~~ → 아래 트러블슈팅에서 해결.

---

## 주요 개선 사항

### 1. 프롬프트 최적화

**문제**: 에이전트가 도구를 5~6회 반복 호출하고, 불필요한 차트까지 생성

```
변경 전: "감자 얼마야?"
→ search_price → search → search_price → search → compare_prices (5회, ~6초)

변경 후: "감자 얼마야?"
→ search_price (1회, ~2초)
```

**해결** (`agent/app/agents/prompts.py` — PRICE_SYSTEM_PROMPT, PLANNER_PROMPT):
- 도구 호출 제한: 6회 → **3회**
- 도구 설명 명확화: `search_price`(시세만) vs `search`(레시피/영양 포함)
- PLANNER 규칙: "사용자가 명시한 것만 계획에 포함" (초기에는 "같은 도구 2번 호출 금지"도 추가했으나, 다품목 비교 문제로 4번에서 삭제)
- "검색어를 바꿔서 재검색" 삭제 → "첫 결과가 충분하면 바로 답변"

**효과**:

| 질문 | 변경 전 | 변경 후 |
|------|---------|---------|
| "감자 얼마야?" | 5회 호출, ~6초 | **1회**, ~2초 |
| "고구마 가격 비교해줘" | 4회 (비교×2 + 차트) | **1회** (비교만) |

### 2. 검색 품질 개선 (RAG 하이브리드)

**문제**: 검색 점수가 0점(date 정렬), 관련 없는 품목 노이즈, 중복 결과

**해결** (`agent/app/agents/search_agent.py`):

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 정렬 기준 | `sort: date desc` (_score=0) | **`_score` 기반** (관련성 점수) |
| multi_match | 감자 검색 → 찹쌀/고구마 포함 | **`minimum_should_match: 75%`** |
| 중복 제거 | `_id` 기준 | **`item_name + kind_name`** 기준 |
| RAG 검색 | BM25만 | **BM25 + kNN 하이브리드** |

**효과** ("감자" 검색):

| 지표 | 변경 전 | 변경 후 |
|------|---------|---------|
| 검색 점수 | 0.0000 | **1.0609** |
| 중복 | 4건 (노지2 + 시설2) | **2건** (노지1 + 시설1) |
| 노이즈 | 찹쌀, 고구마 섞임 | **감자만** |

### 3. Plan-Execute-Reflect 딥에이전트

**문제**: 단순 ReAct 패턴으로는 "감자 가격 알려주고 1주일 추이도 보여줘" 같은 복합 질문을 체계적으로 처리하기 어려움

**해결** (`agent/app/agents/deep_agent.py`, `agent/app/agents/prompts.py`): LangGraph StateGraph 기반 4단계 파이프라인

```
사용자 질문
    │
    ▼
┌─────────┐    계획 수립
│ Planner │───→ "1. search_price(감자)  2. create_price_chart(감자)"
└─────────┘
    │
    ▼
┌──────────┐    단계별 실행
│ Executor │───→ 도구 호출: search_price("감자")
└──────────┘
    │
    ▼
┌───────────┐    자체 평가
│ Reflector │───→ "결과 충분? → done / 부족? → replan"
└───────────┘
    │
    ▼
┌──────────────┐    최종 답변 생성
│ Synthesizer  │───→ "감자 시세 XX원, 1주 추이는..."
└──────────────┘
```

**핵심 특징**:
- **구조화 출력**: Planner가 `Plan(steps=[...])` Pydantic 모델로 계획 생성
- **자가 교정**: Reflector가 `continue/replan/done` 판정 → 부족하면 재계획
- **무한 루프 방지**: `MAX_REPLAN=1` 코드 안전장치 + 프롬프트 규칙("1건이라도 있으면 done")

**Reflector replan 루프 해결**:

"시설 감자 들어간 요리", "감자전 만드는 법" 등에서 Reflector가 "레시피 정보 불충분 → replan"을 무한 반복하는 문제를 발견하고 수정했습니다 (`agent/app/agents/deep_agent.py` MAX_REPLAN, `agent/app/agents/prompts.py` REFLECTOR_PROMPT).

| 질문 | 수정 전 | 수정 후 |
|------|---------|---------|
| 시설 감자 들어간 요리 | reflect 5회, 45초 | **1회, 6초** |
| 감자전 만드는 법 | reflect 4회, 18초 | **1회, 4초** |
| 쌀이랑 감자 중 싼 걸로 추천 | reflect 5회, 19초 | **1회, 7초** |
| 뭐 먹지 | reflect 3회, 13초 | **1회, 5초** |

### 4. 복수 품목 비교 시 검색 누락 해결

**문제**: "쌀과 콩 가격 비교" 질문 시 첫 응답에서 쌀 가격을 못 찾고 콩만 반환. 두 번째 질문에서야 쌀이 나옴 (Task Completion 0.6)

**원인 분석**:
- PLANNER_PROMPT의 "같은 도구를 2번 이상 호출하지 않는다" 규칙이 `search_price`를 품목별로 나눠 호출하는 것을 차단
- Executor가 "쌀과 콩"을 하나의 검색어로 합쳐 ES에 전달 → 매칭 실패
- 최대 3단계 제한으로 두 품목 개별 검색 + 비교까지 단계 부족

**해결** (`agent/app/agents/prompts.py` PLANNER_PROMPT 수정):

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 같은 도구 재호출 | 금지 | **허용** (품목별 개별 호출) |
| 최대 단계 | 3단계 | **5단계** |
| 다품목 처리 | 규칙 없음 | **각 품목을 별도 단계로 분리** 명시 |

```
변경 전: "쌀과 콩 비교" → search_price("쌀과 콩") → 매칭 실패
변경 후: "쌀과 콩 비교" → 1. search_price("쌀") → 2. search_price("콩") → 3. 결과 종합
```

### 5. 평가 체계 고도화

**문제**: 단일 GEval이 "가격 정보 포함 여부"를 모든 케이스에 적용 → "안녕!" 같은 인사에도 부당 감점 (0.26점)

**해결** (`agent/tests/test_deepeval.py`, `agent/tests/test_opik_eval.py`):

```
변경 전: GEval 1개 → 모든 케이스에 동일 기준
변경 후: GEval 3개 → 카테고리별 맞춤 기준

  가격 질문용:  "구체적인 가격 숫자와 단위 포함 여부"
  RAG 질문용:   "레시피/영양 정보의 실용성"
  일반 대화용:  "질문 의도에 맞는 적절한 응답 여부"
```

**평가 프레임워크 구성**:

| 프레임워크 | 메트릭 | 데이터셋 |
|-----------|--------|---------|
| DeepEval | TaskCompletion + GEval(3종) + ToolCorrectness | 13건 |
| Opik | ToolUsage + ResponseQuality + ResponseCompleteness + LLM-as-a-judge | 115건 |

**데이터셋 확장**: 100건 → **115건** (통합검색 15건 + 에지케이스 재분류)

---

## 시스템 아키텍처

### 요청 흐름

> **설명 스크립트** (다이어그램을 따라 읽으면서)
>
> ① 사용자가 "감자 가격 알려주고 1주일 추이도 보여줘"를 채팅창에 입력하면, React UI가 POST `/api/v1/chat`으로 FastAPI에 요청을 보냅니다.
>
> ② FastAPI는 이 메시지를 HumanMessage로 감싸서 Deep Agent의 `astream()`을 호출합니다. 여기서부터 SSE 스트리밍이 시작됩니다.
>
> ③ 먼저 **Planner**가 질문을 분석해서 실행 계획을 세웁니다. "1단계: search_price(감자), 2단계: create_price_chart(감자)" — 이 계획이 `Plan` Pydantic 모델로 구조화되어 나오고, SSE `step: plan` 이벤트로 UI에 전달됩니다.
>
> ④ 다음으로 **Executor**가 첫 번째 단계를 실행합니다. 계획을 보고 `search_price("감자")` 도구를 호출하죠. 이때 SSE `step: model` 이벤트가 나갑니다.
>
> ⑤ **ToolNode**가 실제로 도구를 실행합니다. Elasticsearch에서 감자 가격을 조회하고, 결과를 Executor에게 돌려줍니다. SSE `step: tools` 이벤트로 중간 결과가 UI에 표시됩니다. 이것이 **루프 A**(도구 실행 루프)입니다.
>
> ⑥ 도구 호출이 끝나면 **Reflector**가 결과를 평가합니다. "감자 가격 데이터 확보, 남은 단계 있으니 continue" — `Reflection` 구조체로 판정하고, SSE `step: reflect` 이벤트가 나갑니다. 남은 단계가 있으면 `advance_step`에서 다음 단계를 꺼내 Executor로 돌아갑니다. 이것이 **루프 B**(단계 실행 루프)입니다.
>
> ⑦ 모든 단계가 끝나면 **Synthesizer**가 수집된 정보를 종합해서 "자취고수" 페르소나로 최종 답변을 생성합니다. 차트 데이터가 있으면 metadata에 포함시켜 SSE `step: done`으로 전달하고, UI가 텍스트 + 차트를 렌더링합니다.

```
"감자 가격 알려주고 1주일 추이도 보여줘"

① POST /api/v1/chat
┌──────────────┐            ┌──────────────────────────────────────────┐
│   React UI   │───────────►│  FastAPI                                  │
│  (Vite, TS)  │            │  StreamingResponse(text/event-stream)     │
└──────┬───────┘            └──────────────┬───────────────────────────┘
       │                                   │
       │ ← SSE 스트리밍                     │ ② HumanMessage → agent.astream()
       │                                   ▼
       │    ┌──────────────────────────────────────────────────────┐
       │    │              Deep Agent (LangGraph StateGraph)        │
       │    │                                                      │
       │    │  ③ Planner (ChatOpenAI)                              │
       │    │  │  "search_price(X), create_price_chart(X)"         │
       │    │  │  → Plan(steps=[...]) structured output            │
       │    │  │                                                    │
  SSE  │    │  ▼                                                    │
 plan  │◄───│  ④ Executor (ChatOpenAI)  ◄───────────────────┐      │
       │    │  │  execute step, decide tool calls            │     │
       │    │  │                                             │      │
       │    │  │── tool calls? ──────┐                        │    │
       │    │  │                    ▼                        │      │
 model │◄───│  │              ⑤ Tools (ToolNode)            │      │
       │    │  │              │  search_price("potato")       │    │
 tools │◄───│  │              │                              │      │
       │    │  │              └──► back to Executor ──────────┘    │
       │    │  │                   (tool result applied, Loop A)   │
       │    │  │                                                    │
       │    │  │── no tool calls                                   │
       │    │  ▼                                                    │
       │    │  ⑥ Reflector (ChatOpenAI)                             │
       │    │  │  Reflection(continue/replan/done)                  │
       │    │  │  MAX_REPLAN=1 (prevent infinite loop)             │
       │    │  │                                                    │
reflect│◄───│  │── remaining steps? ──► advance_step ──► Executor  │
       │    │  │                       pop next step from plan     │
       │    │  │                       (Loop B)                    │
       │    │  │                                                    │
       │    │  │── no remaining steps (done)                       │
       │    │  ▼                                                    │
       │    │  ⑦ Synthesizer (ChatOpenAI)                           │
       │    │     final answer + metadata (data, chart)            │
       │    └──────────────────────────────────────────────────────┘
       │                              │
  done │◄─────────────────────────────┘
       │
       ▼
  UI 렌더링 완료
```

### 루프 구조

```
루프 A: Executor ↔ Tools (도구 실행 루프)
  Executor가 도구를 호출하면 Tools 실행 후 결과를 들고 Executor로 복귀.
  도구 호출이 없을 때까지 반복.

루프 B: Executor → Reflector → advance_step → Executor (단계 실행 루프)
  Reflector가 "남은 단계 있음"으로 판정하면 plan에서 다음 단계를 꺼내
  Executor로 복귀. 모든 단계를 소진하면 Synthesizer로 이동.
```

**반복 제한**:

| 제한 장치 | 값 | 설명 |
|----------|---|------|
| Planner 최대 단계 | **5단계** | 프롬프트 규칙으로 계획 단계 수 제한 |
| MAX_REPLAN | **1회** | Reflector의 replan 최대 허용 횟수 (코드 안전장치) |
| recursion_limit | **40회** (기본값, `DEEPAGENT_RECURSION_LIMIT` 환경변수로 조정 가능) | LangGraph 전체 노드 실행 횟수 상한, 초과 시 LLM 폴백 응답 |

### 도구 상세

```
┌──────────────────────────────────────────────────────────────────────┐
│  Tools (x5)                                                          │
│                                                                      │
│  ┌────────────┐ ┌────────────┐ ┌───────────────┐ ┌───────────────┐  │
│  │search      │ │compare     │ │create_price   │ │search         │  │
│  │_price      │ │_prices     │ │_chart         │ │_nutrition     │  │
│  │latest price│ │compare by  │ │price trend    │ │nutrient info  │   │
│  └─────┬──────┘ └─────┬──────┘ └──────┬────────┘ └──────┬────────┘  │
│        │              │               │                 │           │
│        ▼              ▼               ▼                 ▼           │
│  ┌──────────────────────────────┐  ┌────────────────────────────┐   │
│  │  ES: prices-daily-goods      │  │  ES: nutrition-info         │   │
│  │  (prices: 175)               │  │  (nutrition: 75,200)       │    │
│  └──────────────────────────────┘  └────────────────────────────┘   │
│                      ▲                                               │
│  ┌───────────────────┴──────────────────────────────────────────┐   │
│  │  search (sub-agent, LangGraph 3-way parallel)                │    │
│  │  recipe search → ES: edu-recipe-rag (RAG 1,443 docs)        │     │
│  │                                                              │   │
│  │  START ──┬── match_search ────┐                              │   │
│  │          ├── multi_match ─────┼──► merge_results             │   │
│  │          └── rag_search(kNN) ─┘   (dedup, TOP_K)             │    │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### SSE 이벤트 프로토콜

```
클라이언트가 수신하는 SSE 이벤트 순서:

  {"step":"plan",    "plan":["search_price(감자)","create_price_chart(감자)"]}
  {"step":"model",   "tool_calls":["search_price"]}
  {"step":"tools",   "name":"search_price", "content":{...}}
  {"step":"reflect", "content":"가격 정보 확보, 다음 단계 진행"}
  {"step":"model",   "tool_calls":["create_price_chart"]}
  {"step":"tools",   "name":"create_price_chart", "content":{...}}
  {"step":"reflect", "content":"모든 단계 완료"}
  {"step":"done",    "content":"감자 시세는...", "metadata":{...}}
```

---

## 프로젝트 구조

```
├── agent/                      # 백엔드 (FastAPI + LangChain)
│   ├── app/
│   │   ├── main.py             # FastAPI 앱 진입점
│   │   ├── agents/
│   │   │   ├── deep_agent.py   # Plan-Execute-Reflect StateGraph
│   │   │   ├── price_agent.py  # 딥에이전트 래퍼
│   │   │   ├── search_agent.py # 3-way 병렬 검색 서브에이전트
│   │   │   ├── prompts.py      # Planner/Executor/Reflector/Synthesizer 프롬프트
│   │   │   └── tools/          # 도구 5종
│   │   │       ├── search_price.py      # 최신 시세 조회
│   │   │       ├── compare_prices.py    # 기간별 가격 비교
│   │   │       ├── create_price_chart.py # 가격 추이 차트
│   │   │       ├── search_nutrition.py  # 영양성분 조회
│   │   │       └── _es_common.py        # ES 클라이언트 싱글턴
│   │   ├── api/routes/         # 채팅(SSE), 스레드 API
│   │   ├── services/           # 에이전트 서비스, 대화 관리
│   │   └── core/               # 환경 변수 설정
│   ├── pipeline/               # 데이터 파이프라인 (공공 API → 청킹 → 임베딩 → ES)
│   ├── tests/                  # 테스트 & 평가
│   │   ├── test_deep_agent.py  # 딥에이전트 단위 테스트 (24개)
│   │   ├── test_search_agent.py # 검색 에이전트 단위 테스트 (9개)
│   │   ├── test_deepeval.py    # DeepEval 평가 (13건)
│   │   ├── test_opik_eval.py   # Opik 평가 (115건)
│   │   └── generate_report.py  # 성능 리포트 자동 생성
│   └── docs/                   # 설계 문서 & 성능 리포트
├── ui/                         # 프론트엔드 (React 19 + Vite)
│   └── src/
│       ├── components/         # ChartViewer, GridViewer, CodeEditor 등
│       ├── hooks/              # useChat (SSE), useHistory
│       ├── pages/              # ChatPage, InitPage
│       └── services/           # API 서비스
└── docs/                       # 프로젝트 문서
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 백엔드 | FastAPI, LangChain v1.0, LangGraph |
| LLM | OpenAI GPT-4.1-mini |
| 검색 엔진 | Elasticsearch (BM25 + kNN 하이브리드) |
| 임베딩 | OpenAI text-embedding-3-small |
| 프론트엔드 | React 19, Vite, TypeScript, Jotai, TanStack Query |
| 평가 | DeepEval (TaskCompletion, GEval), Opik (LLM-as-a-judge) |
| 데이터 | KAMIS 공공데이터 API, 공공데이터 포털 (영양성분 75,200건) |
| 패키지 관리 | uv (Python), pnpm (Node.js) |

## 시연 대본 (1~2분)

4개 질문으로 **모든 도구와 핵심 기능**을 자연스럽게 보여줌:

| 순서 | 질문 | 트리거되는 장치 |
|------|------|---------------|
| Q1 | 월급은 안 오르는데 물가는 오르네... 요즘 뭐가 싸? | Planner 카테고리→품목 분해, search_price x4 |
| Q2 | 감자 vs 양파 가격 대결! 누가 이겨? | 다품목 분리 검색 (search_price x2), 결과 종합 |
| Q3 | 양파가 금값이라던데 진짜야? 추이 보여줘 | create_price_chart 차트 렌더링 |
| Q4 | 양파가 싸다고? 그럼 오늘 저녁은 양파 파티다! 뭐 해먹지? | search 서브에이전트 (3-way 병렬 RAG 통합검색) |
| Q5 | 양파볶음 칼로리 얼마야? 살찌는 거 아니지? | search_nutrition 영양성분 조회 |

### Q1. "월급은 안 오르는데 물가는 오르네... 요즘 뭐가 싸?"

- Planner가 "배추, 무, 양파, 당근" 4개 품목으로 자동 분해
- 4개 품목 가격을 비교해서 "무가 제일 싸다" 추천
- **포인트**: 모호한 질문 → 카테고리→품목 분해 전략 (프롬프트에 명시)

### Q2. "감자 vs 양파 가격 대결! 누가 이겨?"

- 계획: search_price("감자") → search_price("양파") → 결과 종합
- Synthesizer가 두 품목 가격을 비교 정리
- **포인트**: 다품목 비교 시 각 품목을 별도 단계로 분리 (초기 매칭 실패 문제 해결)

### Q3. "양파가 금값이라던데 진짜야? 추이 보여줘"

- Highcharts 라인 차트 (1년전 → 오늘)
- **포인트**: create_price_chart 도구가 Highcharts 데이터 생성 → 프론트엔드 즉시 렌더링

### Q4. "양파가 싸다고? 그럼 오늘 저녁은 양파 파티다! 뭐 해먹지?"

- search 서브에이전트 호출 (match + multi_match + kNN 병렬), 레시피 추천
- 자취고수 말투 유지 ("ㅋㅋ", 반말)
- **포인트**: BM25 + kNN 하이브리드 검색

### Q5. "양파볶음 칼로리 얼마야? 살찌는 거 아니지?"

- search_nutrition 도구로 영양성분 DB(75,200건) 조회
- 칼로리, 단백질, 지방 등 영양 정보 반환
- **포인트**: 가격→레시피→영양까지 자연스러운 대화 흐름 완성

### 시연 안정성 테스트 (각 20회, 총 80회)

| 질문 | 성공률 |
|------|--------|
| Q1: 월급은 안 오르는데... 요즘 뭐가 싸? | **20/20 (100%)** |
| Q2: 감자 vs 양파 가격 대결! 누가 이겨? | **19/20 (95%)** |
| Q3: 양파가 금값이라던데 진짜야? 추이 보여줘 | **20/20 (100%)** |
| Q4: 양파가 싸다고? 오늘 저녁은 양파 파티다! 뭐 해먹지? | **20/20 (100%)** |
| **전체** | **79/80 (99%)** |

---

## 실행 방법

### 환경 변수

```bash
cp agent/env.sample agent/.env    # OPENAI_API_KEY, ES 접속 정보 입력
cp ui/env.sample ui/.env
```

### 서버 실행

```bash
# 백엔드
cd agent && uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 프론트엔드
cd ui && pnpm install && pnpm dev
```

http://localhost:5173 에서 접속

### 평가 실행

```bash
cd agent

# DeepEval 평가 (13건)
uv run python tests/test_deepeval.py

# Opik 평가 (115건)
uv run python -m tests.test_opik_eval

# 성능 리포트 생성
uv run python tests/generate_report.py --week 4
```

### 테스트

```bash
cd agent
uv run pytest tests/test_deep_agent.py tests/test_search_agent.py -v
```
