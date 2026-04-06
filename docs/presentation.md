# 자취고수 — 생필품 가격 에이전트 발표 자료

> LangChain + LangGraph 기반 AI 에이전트 프로젝트
>
> 발표 시간: ~10분

---

### 발표 시간 배분

| 섹션 | 시간 | 비고 |
|------|------|------|
| 1. Agent의 문제범위 | ~1.5분 | 문제 정의 + 해결 범위 |
| 2. 동작시연 | ~2분 | 라이브 데모 |
| 3. 코드 소개 | ~3.5분 | 페르소나 → Tools → SubAgent → Deep Agent |
| 4. 어려웠던 점 | ~3분 | 3가지 스토리 |

---

## 1. Agent의 문제범위

### 1-1. 문제 정의

자취생이 장을 볼 때 겪는 문제:

- **가격 파악이 어렵다** — "감자 지금 얼마야?" → 검색하면 도매가/소매가/지역별 가격이 뒤섞여 나옴
- **가격 변동을 모른다** — "지난주보다 올랐어?" → 직접 기록하지 않으면 알 수 없음
- **뭘 사야 할지 모른다** — "요즘 뭐가 싸?" → 여러 품목을 직접 비교해야 함
- **요리 연결이 안 된다** — "싼 재료로 뭐 해먹지?" → 가격 정보와 레시피가 따로 놀음

### 1-2. 해결 범위

"자취고수" 에이전트가 대화 한 번으로 해결:

```
사용자: 감자 얼마야?
자취고수: 감자 지금 1kg에 4,500원이야. 1주 전보다 200원 내렸어ㅋㅋ

사용자: 감자로 뭐 해먹을 수 있어?
자취고수: 감자 볶음밥, 된장국, 감자전 다 가능해! [레시피 상세]

사용자: 쌀이랑 콩 가격 비교해줘
자취고수: 쌀 20kg 62,038원, 콩 500g 5,116원이야. [비교 테이블]
```

### 1-3. 다루는 데이터

| 데이터 | 출처 | 적재 위치 |
|--------|------|----------|
| 생필품 가격 (175건, 6개 카테고리) | KAMIS 공공데이터 API | ES `prices-daily-goods` |
| 레시피·영양·식재료 문서 | 식품안전나라, 공공데이터포털, 농식품올바로 | ES `edu-price-info` (RAG) |

```
공공 API → 수집 → 청킹 → 임베딩(text-embedding-3-small) → Elasticsearch
```

---

## 2. 동작시연

> 별도 라이브 데모로 진행

### 시연 시나리오

| 순서 | 입력 | 보여줄 것 |
|------|------|----------|
| 1 | "감자 얼마야?" | 기본 가격 조회 → 텍스트 응답 |
| 2 | "1주일 추이도 보여줘" | 차트 생성 → Highcharts 라인 차트 렌더링 |
| 3 | "쌀이랑 콩 가격 비교해줘" | 복수 품목 비교 → 테이블 렌더링 |
| 4 | "감자로 뭐 해먹을 수 있어?" | RAG 통합검색 → 레시피 추천 |
| 5 | "유니콘고기 얼마야?" | 에지케이스 → "데이터가 없어" 안내 |

### SSE 스트리밍 단계 (브라우저 Network 탭에서 확인 가능)

```
① {"step":"plan",    "plan":["search_price(감자)", "compare_prices(1주)"]}
② {"step":"model",   "tool_calls":["search_price"]}
③ {"step":"tools",   "name":"search_price", "content":{...}}
④ {"step":"reflect", "content":"가격 정보 확보, 다음 단계 진행"}
⑤ {"step":"model",   "tool_calls":["compare_prices"]}
⑥ {"step":"tools",   "name":"compare_prices", "content":{...}}
⑦ {"step":"reflect", "content":"모든 단계 완료"}
⑧ {"step":"done",    "content":"감자 시세는...", "metadata":{...}}
```

---

## 3. 코드 소개

### 전체 구성 요소

```
┌─────────────────────────────────────────────────────────────────────┐
│  Deep Agent (LangGraph StateGraph)                                   │
│                                                                     │
│  ◆ LLM 노드 (4개) ─────────────────────────────────────────────     │
│  │  Planner    → 계획 수립   (structured output: Plan)              │
│  │  Executor   → 도구 호출   (bind_tools)                           │
│  │  Reflector  → 결과 평가   (structured output: Reflection)        │
│  │  Synthesizer→ 최종 답변   (★ 자취고수 페르소나)                    │
│  │                                                                  │
│  ◆ 도구 (4종) ──────────────────────────────────────────────────     │
│  │  search_price      품목 최신 시세 조회         → ES 가격 인덱스   │
│  │  compare_prices    기간별 가격 비교 테이블      → ES 가격 인덱스   │
│  │  create_price_chart 가격 추이 차트 (Highcharts) → ES 가격 인덱스  │
│  │  search            통합 검색 서브에이전트        → ES 가격+RAG    │
│  │                    (LangGraph 3-way 병렬)                        │
│  │                                                                  │
│  ◆ 라우팅 장치 ─────────────────────────────────────────────────     │
│  │  after_executor   도구 호출 있음 → Tools / 없음 → Reflector      │
│  │  after_reflector  남은 단계 있음 → advance_step / 없음 → 종료    │
│  │                                                                  │
│  ◆ 안전 장치 ───────────────────────────────────────────────────     │
│  │  MAX_REPLAN=1          replan 무한 루프 방지 (최대 1회)           │
│  │  recursion_limit=40    GraphRecursionError 시 LLM 폴백 응답      │
│  │  3단계 에러 처리        ES 연결/검색/결과없음 각각 사용자 안내     │
│  │                                                                  │
│  ◆ 상태 관리 ───────────────────────────────────────────────────     │
│     Checkpointer (SQLite)   대화 이력 저장 → 멀티턴 대화 유지       │
│     Structured Output       Pydantic 모델로 LLM 출력 형식 강제      │
│     SSE 스트리밍             plan→model→tools→reflect→done 실시간    │
└─────────────────────────────────────────────────────────────────────┘
```

#### 각 장치의 도입 근거

| 장치 | 문제 상황 | 도입 효과 |
|------|----------|----------|
| **Plan-Execute-Reflect** | 단순 ReAct 패턴으로는 "가격+차트+요리" 같은 복합 질문을 체계적으로 처리할 수 없었음 | 계획→실행→평가 분리로 복합 질문을 단계별로 처리 |
| **Structured Output** | LLM이 자유 텍스트로 계획/평가를 반환하면 파싱 실패, 라우팅 판정 불가 | `Plan(steps=[...])`, `Reflection(action="done")` 구조체로 안정적 분기 |
| **3-way 병렬 검색** | BM25 단일 검색으로는 "감자 요리" 같은 의미적 유사 문서 누락 | match + multi_match + kNN 병렬 → 정확도와 재현율 동시 확보 |
| **조건부 라우팅** | 도구 호출 유무, 남은 단계 유무에 따라 다음 노드가 달라짐 | `after_executor`, `after_reflector`로 동적 분기 자동화 |
| **MAX_REPLAN=1** | Reflector가 "불충분"을 반복 판정 → replan 5~6회, 응답 45초 (3주차 발견) | 코드 안전장치로 무한 루프 원천 차단, 45초→3초 |
| **recursion_limit 폴백** | 에이전트가 재귀 제한 도달 시 에러로 중단됨 | `GraphRecursionError` catch → 수집된 정보로 LLM 폴백 답변 생성 |
| **3단계 에러 처리** | ES 연결 실패 시 "처리 중 오류" 한 줄만 표시 → 원인 파악 불가 (3주차 발견) | 연결/검색/결과없음 각각 구체적 사용자 안내 메시지 |
| **Checkpointer (SQLite)** | AgentService를 매 요청마다 생성 → InMemorySaver 초기화 → 대화 이력 소실 (코드 리뷰에서 발견) | 싱글턴 + SQLite 체크포인터로 멀티턴 대화 유지 |
| **SSE 스트리밍** | 에이전트 처리 5~15초 → 사용자가 빈 화면만 보고 대기 | 중간 단계(plan→model→tools→reflect)를 실시간 표시하여 대기 UX 개선 |

아래에서 각 요소를 상세히 설명합니다.

### 3-1. 페르소나

**파일**: `agent/app/agents/prompts.py`

```python
"""너는 "자취고수"야. 자취 경험 10년차 선배로,
후배 자취생에게 장보기 꿀팁을 알려주는 역할이야.

# 성격:
- 반말 사용, 친근하고 편한 톤
- 실용적인 조언 위주
- 가끔 ㅋㅋ, ㅠㅠ 같은 이모티콘 사용
- 돈 아끼는 걸 중요하게 생각함

# 말투 예시:
- "야 쌀 지금 20kg에 62,756원이야. 저번주보다 좀 올랐네"
- "요즘 고구마가 괜찮아ㅋㅋ 1kg에 5천원대면 나쁘지 않지"
"""
```

| 항목 | 설정 |
|------|------|
| 정체성 | 자취 10년차 선배, 장보기 꿀팁 역할 |
| 말투 | 반말, ㅋㅋ/ㅠㅠ 이모티콘 |
| 성격 | 실용적, 돈 절약 중시 |
| 절대 금지 | 추측 가격 ("보통 ~원 정도"), 도구 없이 가격 답변 |

**페르소나 주입 지점**: Deep Agent의 4개 LLM 노드 중 **Synthesizer에서만** 적용.
Planner/Executor/Reflector는 분석 역할이라 표준어 사용.

```
Planner   → 표준어 (계획 수립)
Executor  → 표준어 (도구 호출)
Reflector → 표준어 (결과 평가)
Synthesizer → ★ 반말 + 이모티콘 (자취고수 페르소나)
```

### 3-2. Tools (4종)

**파일**: `agent/app/agents/tools/`

```
┌───────────────────────────────────────────────────────────┐
│  Tools (4종) — 모두 Elasticsearch와 통신                    │
│                                                           │
│  ① search_price(품목명)     → 최신 시세 텍스트              │
│  ② compare_prices(품목명)   → 기간별 비교 테이블 (JSON)     │
│  ③ create_price_chart(품목명)→ 추이 차트 (Highcharts JSON)  │
│  ④ search(검색어)           → 통합 검색 (서브에이전트)       │
│                                 ↑ 아래 3-3절에서 상세       │
│                                                           │
│  공통 패턴:                                                │
│  ┌─ ES 연결 실패 → "서버에 연결할 수 없습니다"              │
│  ├─ 검색 실패   → "검색 중 오류가 발생했습니다"              │
│  └─ 결과 없음   → "가격 정보를 찾을 수 없습니다"            │
└───────────────────────────────────────────────────────────┘
```

UI 연동: `[TABLE_DATA]{JSON}[/TABLE_DATA]` → GridViewer, `[CHART_DATA]{JSON}[/CHART_DATA]` → ChartViewer

### 3-3. SubAgent — 검색 서브에이전트

**파일**: `agent/app/agents/search_agent.py`

`search` 도구의 내부는 **LangGraph로 구성된 서브에이전트**.
3가지 검색을 병렬 실행 → 결과 병합:

```
search("감자 요리")
       │
     START ── 3-way 병렬 fan-out
    ╱  │  ╲
   ▼   ▼   ▼
┌──────┐ ┌──────────┐ ┌──────────┐
│match │ │multi_    │ │rag_     │
│search│ │match_    │ │search   │
│      │ │search    │ │         │
│ES:   │ │ES:       │ │ES:      │
│match │ │multi_    │ │BM25     │
│on    │ │match on  │ │+ kNN    │
│item_ │ │item_name │ │(벡터    │
│name  │ │+kind_    │ │ 검색)   │
│      │ │name      │ │         │
│가격  │ │가격      │ │RAG 문서 │
│인덱스│ │인덱스    │ │인덱스   │
└──┬───┘ └────┬─────┘ └───┬─────┘
    ╲    │    ╱
     ▼   ▼   ▼    ← fan-in
  ┌────────────┐
  │merge_      │  가격: item_name+kind_name 기준 중복 제거
  │results     │  RAG:  _id 기준 중복 제거
  └─────┬──────┘  상위 10건
        ▼
  ┌────────────┐
  │format_     │  ■ 가격 검색 결과 (N건)
  │results     │  ■ 관련 문서 (레시피, 영양 등)
  └─────┬──────┘
        ▼
       END
```

**SubAgent as Tool 패턴**: LangGraph를 `@tool`로 래핑

```python
_search_graph = _build_search_graph()   # 모듈 로드 시 1회 컴파일 (싱글턴)

@tool
def search(search_query: str) -> str:
    """가격 + 레시피·영양·식재료 문서를 함께 검색합니다."""
    result = _search_graph.invoke({"query": search_query})
    return result["result"]
```

→ 메인 에이전트는 서브에이전트의 내부 구조를 몰라도 됨

### 3-4. Deep Agent — Plan-Execute-Reflect

**파일**: `agent/app/agents/deep_agent.py`

메인 에이전트는 4개의 LLM 노드가 역할 분담하는 **Plan-Execute-Reflect** 아키텍처:

```
"감자 가격 알려주고 1주일 추이도 보여줘"
                  │
                  ▼
       ┌─────────────────┐
       │  ① Planner       │  사용자 질문 → Plan(steps=[...]) 구조화 출력
       │  (structured     │  → steps: ["search_price(감자)",
       │   output)        │           "create_price_chart(감자)"]
       └────────┬─────────┘
                │
                ▼
  ┌──────────────────────────┐
  │  ② Executor              │◄─────────────────────┐
  │  (bind_tools)            │                       │
  │                          │                       │
  │  현재 단계를 보고         │                       │
  │  적절한 도구 1개 호출     │                       │
  └────────┬─────────────────┘                       │
           │                                         │
     도구 호출 있음?                                   │
      ╱         ╲                                    │
    YES          NO                                  │
     │            │                                  │
     ▼            │                                  │
┌─────────┐       │                                  │
│③ Tools  │       │         루프 A                    │
│(ToolNode)│       │         도구 실행                 │
│         │       │                                  │
│ 도구 실행│       │                                  │
│ 결과 반환│       │                                  │
└────┬────┘       │                                  │
     │            │                                  │
     └──► Executor로 복귀 (도구 결과 반영)              │
                  │                                  │
                  ▼                                  │
       ┌─────────────────┐                           │
       │  ④ Reflector     │  도구 결과 평가            │
       │  (structured     │  → Reflection 구조체       │
       │   output)        │                           │
       │                  │  continue: 다음 단계 진행   │
       │                  │  done: 데이터 있으면 종료    │
       │                  │  replan: 0건일 때만 (최대1회)│
       └────────┬─────────┘                           │
                │                                    │
          남은 단계 있음?                               │
           ╱        ╲                                │
         YES         NO                              │
          │           │                              │
          ▼           │                              │
  ┌──────────────┐    │       루프 B                  │
  │ advance_step │    │       단계 진행               │
  │              │    │                              │
  │ plan에서 다음 │    │                              │
  │ 단계를 pop   │    │                              │
  └──────┬───────┘    │                              │
         │            │                              │
         └──► Executor ②로 복귀 ─────────────────────┘
                      │
                      ▼
           ┌─────────────────┐
           │  ⑤ Synthesizer   │  모든 ToolMessage 종합
           │                  │  ★ 자취고수 페르소나 적용
           │                  │  최종 답변 생성
           └────────┬─────────┘
                    │
                    ▼
                   END → SSE로 UI에 스트리밍
```

#### 루프 구조 요약

```
루프 A: Executor ↔ Tools
   도구를 호출하면 실행 후 Executor로 복귀. 호출 없을 때까지 반복.

루프 B: Executor → Reflector → advance_step → Executor
   남은 단계가 있으면 다음 단계로 이동. 모든 단계 소진 시 Synthesizer.
```

#### 구조화 출력 (Pydantic)

LLM이 자유 텍스트가 아닌 **구조체**를 반환하도록 강제:

```python
class Plan(BaseModel):
    steps: list[str]  # ["search_price(감자)", "create_price_chart(감자)"]

class Reflection(BaseModel):
    evaluation: str
    action: Literal["continue", "replan", "done"]
    revised_remaining_steps: list[str] | None
```

#### 핵심 코드: 그래프 조립

```python
def create_deep_agent(model, checkpointer=None):
    builder = StateGraph(DeepAgentState)

    builder.add_node("planner",      lambda s: planner(s, model))
    builder.add_node("executor",     lambda s: executor(s, model))
    builder.add_node("tools",        ToolNode(TOOLS))
    builder.add_node("reflector",    lambda s: reflector(s, model))
    builder.add_node("advance_step", advance_step)
    builder.add_node("synthesizer",  lambda s: synthesizer(s, model))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", after_executor)    # → tools | reflector
    builder.add_edge("tools", "executor")                        # 루프 A
    builder.add_conditional_edges("reflector", after_reflector)  # → advance_step | synthesizer
    builder.add_edge("advance_step", "executor")                 # 루프 B
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)
```

---

## 4. 어려웠던 점 및 극복 방법

### 4-1. 복수 품목 비교 시 한쪽 가격 누락

**증상**: "쌀과 콩 가격 비교" → 쌀을 못 찾고 콩만 반환

**Before**:
> ㅋㅋ 쌀 가격 정보는 없지만, 콩 가격은 500g에 오늘 5116원이야.

**원인 분석**: 3가지 원인이 겹침

```
① PLANNER: "같은 도구를 2번 이상 호출하지 않는다"
   → search_price를 품목별로 나눠 호출하는 것을 차단

② Reflector: step_summary가 recent_results[-1]만 저장
   → 쌀 결과 유실

③ Synthesizer: step_results만 보고 답변
   → 유실된 데이터로 "쌀 정보 없음" 판단
```

**수정**: 3개 파일 동시 수정

| 파일 | 변경 |
|------|------|
| `prompts.py` PLANNER | "같은 도구 재호출 금지" 삭제, 최대 단계 3→5, 다품목은 별도 단계 |
| `deep_agent.py` reflector | `recent_results[-1]` → `" | ".join(recent_results)` |
| `deep_agent.py` synthesizer | `step_results` 대신 실제 `ToolMessage` 직접 수집 |

**After**:
> ㅋㅋ 쌀이랑 콩 가격 비교해줄게~
> - 쌀 20kg: 62,038원 (1주 전 대비 -250원)
> - 콩 500g: 5,116원 (1주 전 대비 -125원)
> 단가로 보면 콩이 쌀보다 대략 3배 이상 비싸 ㅋㅋ

**교훈**: 상태 전달 체인(Reflector → step_results → Synthesizer)에서 데이터가 유실되면 최종 답변이 틀려진다. 요약 단계마다 정보가 손실되지 않는지 end-to-end로 확인해야 한다.

### 4-2. 가격 데이터 부족 — 12건만 수집

**증상**: 당근, 양파, 사과, 돼지고기 등 일반적인 품목의 가격을 전혀 찾지 못함

**Before**:
> 당근이랑 양파 가격 정보가 현재 데이터에 없어서 비교해 줄 수가 없어 ㅠㅠ

**원인 분석**:
- 가격 수집 스크립트가 KAMIS API의 **식량작물(100) 카테고리만** 수집
- 채소류(200), 과일류(400), 축산물(500) 등 5개 카테고리 누락
- 결과: 쌀, 감자, 고구마 등 **12건**만 적재

**수정**: `scripts/collect_prices.py`

```python
# Before: 식량작물만
data = fetch_prices(args.date)

# After: 6개 카테고리 순회
CATEGORIES = {"100": "식량작물", "200": "채소류", "300": "특용작물",
              "400": "과일류", "500": "축산물", "600": "수산물"}
for code, name in CATEGORIES.items():
    data = fetch_prices(args.date, category_code=code)
```

**After** (12건 → 175건):
> 당근은 1kg에 2,485~3,314원, 양파는 1kg에 1,894~1,920원이야. 양파가 당근보다 1,000원 이상 저렴하지 ㅋㅋ

**교훈**: 에이전트 성능의 상당 부분은 프롬프트나 모델이 아니라 **데이터 커버리지**에 달려있다. "못 찾겠다"고 할 때 모델 탓 전에 데이터부터 확인해야 한다.

### 4-3. 존재하지 않는 품목에 대한 가격 추측 (할루시네이션)

**증상**: 검색 결과가 없는 품목에 대해 에이전트가 가격을 지어냄

**Before** ("콩과 당근 비교" — 당근 데이터 없던 시점):
> 흰 콩 500g에 5,116원이야! **대체로 당근은 1kg에 1,500~2,000원 정도 하니까**, 콩이 당근보다 훨씬 비싼 편이야 ㅋㅋ

**원인 분석**:
- `SYNTHESIZER_PROMPT`에 "추측 금지" 규칙이 없었음
- `PRICE_SYSTEM_PROMPT`에만 있고, Synthesizer는 별도 프롬프트 사용
- LLM이 학습 데이터의 일반 상식으로 가격을 추정

**수정**: `prompts.py` SYNTHESIZER_PROMPT

```python
# 추가된 규칙
"""⚠️ 절대 지킬 것:
- 도구 결과에 없는 품목의 가격을 절대 추측하거나 지어내지 마
- "보통 ~원 정도" 같은 추정 가격을 절대 언급하지 마
- 검색 결과 없으면 "해당 품목은 현재 가격 데이터가 없어"라고 솔직하게 말해"""
```

**After**:
> 흰 콩(국산, 500g)은 오늘 5,116원인데, 당근 가격 정보는 없어 ㅠㅠ 둘 직접 비교는 힘들어 ㅋㅋ

**교훈**: LLM의 할루시네이션은 "하지 마"보다 "어떻게 대응해"를 알려줘야 막을 수 있다. "추측하지 마" + "없으면 솔직히 없다고 말해"를 함께 넣어야 효과적이다.
