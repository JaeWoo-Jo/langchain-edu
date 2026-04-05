# 2주차 목요일: DeepEval 라이브러리 도입

## 개요

DeepEval은 LLM 에이전트 평가에 특화된 오픈소스 프레임워크다. 수요일에 직접 만든 LLM-as-a-judge와 달리, **학술적으로 검증된 표준 메트릭**을 제공한다.

## 수요일(직접 구현) vs 목요일(DeepEval) 비교

| | 수요일: LLM-as-a-judge | 목요일: DeepEval |
|---|---|---|
| **방식** | GPT-4o 프롬프트를 직접 작성 | 라이브러리가 검증된 프롬프트 제공 |
| **메트릭** | 커스텀 4가지 기준 | 표준 에이전틱 메트릭 |
| **추적** | 없음 (결과만 채점) | `@observe` 데코레이터로 실행 흐름 자동 추적 |
| **비유** | 채점 기준을 직접 만든 시험 | 공인 시험 (토익 같은) |

## 핵심 개념: @observe 데코레이터

DeepEval의 가장 중요한 차별점은 **에이전트 실행 흐름을 자동 추적**한다는 것이다.

```python
@observe(type="agent")     # 최상위 에이전트 추적
def run_agent(user_input):
    ...

@observe(type="tool")      # 개별 도구 호출 추적
def search_price_traced(item_name):
    ...

@observe(type="llm")       # LLM 호출 추적 (도구 선택 판단)
def call_llm(messages):
    ...
```

이 데코레이터가 기록한 정보를 메트릭이 자동으로 분석한다:
- 어떤 도구가 호출되었는지
- 실행 순서가 어떠했는지
- 최종 결과가 무엇인지

## 적용한 메트릭 3종

### 1. TaskCompletionMetric — 업무 완료율

```python
task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")
```

- **역할**: 에이전트가 사용자의 요청을 성공적으로 완수했는지 LLM이 판단
- **동작**: @observe로 추적된 전체 실행 흐름을 보고 "완료/미완료" 판정
- **threshold=0.7**: 70% 이상이면 통과
- **예시**: "쌀 가격 알려줘" → 가격 정보를 포함한 답변 반환 → 완료 판정

### 2. ToolCorrectnessMetric — 도구 선택 정확도

```python
tool_correctness = ToolCorrectnessMetric(threshold=0.7)
```

- **역할**: 에이전트가 올바른 도구를 선택했는지 검증
- **동작**: Golden에 지정한 `expected_tools`와 실제 호출된 도구를 비교
- **예시**:
  - 질문: "쌀 가격 알려줘"
  - expected: `search_price` → actual: `search_price` → 통과
  - expected: `search_price` → actual: `create_price_chart` → 실패

### 3. GEval (한국어 응답 품질) — 커스텀 채점

```python
korean_quality = GEval(
    name="한국어 응답 품질",
    criteria="에이전트의 한국어 응답이 정확하고 자연스러운지 평가",
    evaluation_steps=[
        "사용자의 질문 의도를 정확히 파악했는지 확인한다",
        "답변에 포함된 가격 정보가 구체적인 숫자와 단위를 포함하는지 확인한다",
        "한국어 표현이 자연스럽고 문법적으로 올바른지 확인한다",
        "불필요한 영어 혼용이나 기계적 표현이 없는지 확인한다",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o",
    threshold=0.7,
)
```

- **역할**: G-Eval 논문(Liu et al., 2023) 기반 커스텀 채점
- **핵심**: `evaluation_steps`에 채점 절차를 단계별로 명시하면 LLM이 더 정밀하게 채점
- **criteria vs evaluation_steps**: criteria는 전체 기준, steps는 구체적 절차 (둘 중 하나만 사용)

## 테스트 데이터셋 (Golden)

DeepEval에서 하나의 테스트 케이스를 `Golden`이라 부른다.

```python
Golden(
    input="쌀 가격 알려줘",                          # 사용자 질문
    expected_tools=[ToolCall(name="search_price")],  # 기대하는 도구
)
```

10개의 대표 케이스를 구성했다:

| 카테고리 | 개수 | 기대 도구 |
|----------|------|-----------|
| 가격 검색 | 3개 | `search_price` |
| 가격 비교 | 2개 | `compare_prices` |
| 차트 생성 | 2개 | `create_price_chart` |
| 일반 대화 | 1개 | (없음) |
| 에지 케이스 | 2개 | `search_price` |

## 동작 흐름

```
[1] Golden 데이터 준비
    input: "쌀 가격 알려줘"
    expected_tools: [search_price]

[2] @observe가 에이전트 실행 추적
    run_agent("쌀 가격 알려줘")
      └─ LLM이 search_price 호출 결정
      └─ search_price 실행 → ES에서 가격 조회
      └─ LLM이 최종 답변 생성

[3] 메트릭별 채점
    TaskCompletionMetric: "가격 정보를 반환했으므로 완료" → 0.9
    ToolCorrectnessMetric: search_price == search_price → 1.0
    GEval(한국어 품질): "자연스러운 한국어, 숫자+단위 포함" → 0.85
```

## 실행 방법

```bash
cd agent/

# DeepEval 전용 러너 (상세 리포트 출력)
uv run deepeval test run tests/test_deepeval.py

# 또는 직접 실행
uv run python tests/test_deepeval.py
```

## 파일 위치

`tests/test_deepeval.py`

## 전체 평가 체계 (2주차 완성 후)

| 파일 | 프레임워크 | 메트릭 |
|------|-----------|--------|
| `test_opik_eval.py` | Opik | tool_usage, response_quality, response_completeness, **llm_judge** |
| `test_deepeval.py` | DeepEval | **TaskCompletion, ToolCorrectness, GEval(한국어품질)** |

Opik은 트레이싱 + 규칙 기반 메트릭, DeepEval은 에이전틱 표준 메트릭으로 역할을 분담한다.

## 참고 자료

- [DeepEval AI Agent 평가 가이드](https://deepeval.com/guides/guides-ai-agent-evaluation)
- [G-Eval 논문 기반 메트릭](https://deepeval.com/docs/metrics-llm-evals)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
