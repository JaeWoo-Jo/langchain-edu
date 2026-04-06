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

**해결**:
- 도구 호출 제한: 6회 → **3회**
- 도구 설명 명확화: `search_price`(시세만) vs `search`(레시피/영양 포함)
- PLANNER 규칙: "같은 도구 2번 호출 금지", "사용자가 명시한 것만 계획에 포함"
- "검색어를 바꿔서 재검색" 삭제 → "첫 결과가 충분하면 바로 답변"

**효과**:

| 질문 | 변경 전 | 변경 후 |
|------|---------|---------|
| "감자 얼마야?" | 5회 호출, ~6초 | **1회**, ~2초 |
| "고구마 가격 비교해줘" | 4회 (비교×2 + 차트) | **1회** (비교만) |

### 2. 검색 품질 개선 (RAG 하이브리드)

**문제**: 검색 점수가 0점(date 정렬), 관련 없는 품목 노이즈, 중복 결과

**해결**:

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

**해결**: LangGraph StateGraph 기반 4단계 파이프라인

```
사용자 질문
    │
    ▼
┌─────────┐    계획 수립
│ Planner │───→ "1. search_price(감자)  2. compare_prices(감자, 1주)"
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

"시설 감자 들어간 요리", "감자전 만드는 법" 등에서 Reflector가 "레시피 정보 불충분 → replan"을 무한 반복하는 문제를 발견하고 수정했습니다.

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

**해결**:

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
       │    │  │  "1. search_price(감자)  2. compare_prices(1주)"  │
       │    │  │  → Plan(steps=[...]) 구조화 출력                   │
       │    │  │                                                    │
  SSE  │    │  ▼                                                    │
 plan  │◄───│  ④ Executor (ChatOpenAI)  ◄───────────────────┐      │
       │    │  │  현재 단계 실행, 도구 호출 결정                │      │
       │    │  │                                             │      │
       │    │  │── 도구 호출 있음? ──┐                        │      │
       │    │  │                    ▼                        │      │
 model │◄───│  │              ⑤ Tools (ToolNode)            │      │
       │    │  │              │  search_price("감자") 실행    │      │
 tools │◄───│  │              │                              │      │
       │    │  │              └──► Executor로 복귀 ───────────┘      │
       │    │  │                   (도구 결과 반영, 루프 A)          │
       │    │  │                                                    │
       │    │  │── 도구 호출 없음                                    │
       │    │  ▼                                                    │
       │    │  ⑥ Reflector (ChatOpenAI)                             │
       │    │  │  Reflection(action=continue/replan/done) 구조화 출력│
       │    │  │  MAX_REPLAN=1 (무한 루프 방지)                      │
       │    │  │                                                    │
reflect│◄───│  │── 남은 단계 있음? ──► advance_step ──► Executor    │
       │    │  │                       plan에서 다음 단계 pop        │
       │    │  │                       (루프 B)                     │
       │    │  │                                                    │
       │    │  │── 남은 단계 없음 (done)                             │
       │    │  ▼                                                    │
       │    │  ⑦ Synthesizer (ChatOpenAI)                           │
       │    │     최종 답변 + metadata (code_snippet, data, chart)   │
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

### 도구 상세

```
┌──────────────────────────────────────────────────────────────┐
│  Tools (4종)                                                  │
│                                                              │
│  ┌─────────────┐  ┌───────────────┐  ┌──────────────────┐   │
│  │ search      │  │ compare       │  │ create_price     │   │
│  │ _price      │  │ _prices       │  │ _chart           │   │
│  │ 최신 시세    │  │ 기간별 비교    │  │ 가격 추이 차트    │   │
│  └──────┬──────┘  └───────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Elasticsearch (prices, RAG)                │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ▲                                 │
│  ┌─────────────────────────┴───────────────────────────┐    │
│  │  search (서브에이전트, LangGraph 3-way 병렬)          │    │
│  │  레시피·영양·식재료 통합 검색                          │    │
│  │                                                     │    │
│  │  START ──┬── match_search ────┐                     │    │
│  │          ├── multi_match ─────┼──► merge_results    │    │
│  │          └── rag_search(kNN) ─┘   (중복 제거, TOP_K) │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### SSE 이벤트 프로토콜

```
클라이언트가 수신하는 SSE 이벤트 순서:

  {"step":"plan",    "plan":["search_price(감자)","compare_prices(1주)"]}
  {"step":"model",   "tool_calls":["search_price"]}
  {"step":"tools",   "name":"search_price", "content":{...}}
  {"step":"reflect", "content":"가격 정보 확보, 다음 단계 진행"}
  {"step":"model",   "tool_calls":["compare_prices"]}
  {"step":"tools",   "name":"compare_prices", "content":{...}}
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
│   │   │   └── tools/          # 도구 4종
│   │   │       ├── search_price.py      # 최신 시세 조회
│   │   │       ├── compare_prices.py    # 기간별 가격 비교
│   │   │       ├── create_price_chart.py # 가격 추이 차트
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
| 데이터 | KAMIS 공공데이터 API, 공공데이터 포털 |
| 패키지 관리 | uv (Python), pnpm (Node.js) |

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
