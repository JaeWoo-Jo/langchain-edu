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

### 4. 평가 체계 고도화

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

```
┌──────────────┐     SSE      ┌──────────────────────────────────────────┐
│   React UI   │◄────────────►│  FastAPI (/api/v1/chat)                  │
│  (Vite, TS)  │  스트리밍     │                                          │
└──────────────┘              │  ┌──────────────────────────────────────┐ │
                              │  │         Deep Agent (LangGraph)       │ │
                              │  │                                      │ │
                              │  │  Planner → Executor → Reflector      │ │
                              │  │              │           │            │ │
                              │  │              ▼       (replan)        │ │
                              │  │          ┌───────┐      │            │ │
                              │  │          │ Tools │◄─────┘            │ │
                              │  │          └───┬───┘                   │ │
                              │  │              │                       │ │
                              │  │              ▼                       │ │
                              │  │         Synthesizer                  │ │
                              │  └──────────────────────────────────────┘ │
                              │                 │                         │
                              └─────────────────┼─────────────────────────┘
                                                │
                              ┌─────────────────┼─────────────────────┐
                              │                 ▼                     │
                              │  ┌─────────┐ ┌──────────┐ ┌───────┐  │
                              │  │ search  │ │ compare  │ │ chart │  │
                              │  │ _price  │ │ _prices  │ │       │  │
                              │  └────┬────┘ └────┬─────┘ └───┬───┘  │
                              │       │           │           │      │
                              │       ▼           ▼           ▼      │
                              │    ┌──────────────────────────────┐   │
                              │    │  Elasticsearch (prices, RAG) │   │
                              │    └──────────────────────────────┘   │
                              │                                      │
                              │  ┌──────────────────────────────────┐ │
                              │  │ search (서브에이전트, 3-way 병렬)  │ │
                              │  │  match + multi_match + rag(kNN)  │ │
                              │  └──────────────────────────────────┘ │
                              └──────────────────────────────────────┘
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
