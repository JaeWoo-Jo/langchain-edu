# 2주차 수요일: LLM-as-a-judge 평가 설계

## 개요

에이전트의 답변 품질을 **GPT-4o가 자동으로 채점**하는 LLM-as-a-judge 메트릭을 구현한다.

기존에 구현된 규칙 기반 메트릭(키워드 매칭)과 달리, LLM이 사람처럼 답변을 읽고 판단하므로 더 정밀한 품질 평가가 가능하다.

## 규칙 기반 vs LLM-as-a-judge 비교

### 규칙 기반 (기존 메트릭)

```python
# "원"이라는 글자만 있으면 점수를 올림
if "원" in output:
    score_val += 0.25
```

- 장점: 빠르고, API 비용 없음
- 단점: "쌀 가격은 50,000원입니다"도, "1원도 없어요"도 동일하게 점수를 받음

### LLM-as-a-judge (새 메트릭)

```python
# GPT-4o가 질문과 답변을 읽고 여러 기준으로 1~5점 채점
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "채점 기준 프롬프트..."},
        {"role": "user", "content": f"질문: {input}\n답변: {output}"},
    ],
    response_format={"type": "json_object"},
    temperature=0,
)
```

- 장점: 의미를 이해하고 판단, 사람 평가와 유사
- 단점: API 호출 비용 발생, 100건 평가 시 100번 추가 호출

## 채점 기준 (4가지)

| 기준 | 영문 키 | 설명 |
|------|---------|------|
| 정확성 | `accuracy` | 사실에 기반한 정확한 정보를 제공하는가? |
| 유용성 | `usefulness` | 사용자가 실제로 원하는 정보를 담고 있는가? |
| 완결성 | `completeness` | 빠진 정보 없이 충분히 완전한가? |
| 한국어 자연스러움 | `naturalness` | 한국어 표현이 자연스럽고 읽기 쉬운가? |

각 기준은 1~5점 (1=매우 미흡, 5=매우 우수)으로 채점된다.

## 동작 흐름

```
[1] 에이전트 실행
    사용자 질문: "쌀 가격 알려줘"
    에이전트 답변: "쌀(일반계) 오늘 55,000원/20kg (1주전 대비 +500원)"

[2] GPT-4o 판사에게 전달
    시스템 프롬프트: 채점 기준 4가지 + JSON 형식 지정
    사용자 메시지: 질문 + 답변

[3] GPT-4o 채점 결과 (JSON)
    {
      "accuracy": 4,
      "usefulness": 5,
      "completeness": 3,
      "naturalness": 5,
      "reason": "가격 정보 정확하나 전일 대비 변동 누락"
    }

[4] Opik에 기록
    점수: (4+5+3+5) / 4 / 5 = 0.85
    사유: "정확성:4 유용성:5 완결성:3 자연스러움:5 | 가격 정보 정확하나..."
```

## 구현 위치

`tests/test_opik_eval.py` 내 `LLMJudgeMetric` 클래스

### 핵심 설계 포인트

**1. JSON 강제 응답**

```python
response_format={"type": "json_object"}
```

GPT-4o가 반드시 JSON으로만 응답하게 강제한다. 자연어 설명 없이 정형화된 점수만 받을 수 있다.

**2. temperature=0**

```python
temperature=0
```

동일한 질문/답변에 대해 매번 같은 점수를 내도록 랜덤성을 제거한다. 평가의 재현성(reproducibility)을 보장.

**3. 점수 정규화**

```python
normalized = avg / 5.0  # 1~5점 → 0.0~1.0
```

Opik의 메트릭은 0.0~1.0 스케일을 사용하므로, 5점 만점 기준으로 정규화한다.

**4. 에러 핸들링**

API 호출 실패 시 0점을 반환하되, 사유에 에러 메시지를 기록하여 디버깅이 가능하도록 한다.

## 전체 메트릭 목록 (evaluate 실행 시)

| # | 메트릭 | 유형 | 설명 |
|---|--------|------|------|
| 1 | `tool_usage_accuracy` | 규칙 기반 | 올바른 도구를 호출했는지 |
| 2 | `response_quality` | 규칙 기반 | 응답 길이, 한글 포함, 가격 정보 |
| 3 | `response_completeness` | 규칙 기반 | 카테고리별 응답 완결성 |
| 4 | `llm_judge` | **LLM-as-a-judge** | GPT-4o 기반 4가지 기준 종합 채점 |

## 실행 방법

```bash
cd agent/
uv run python -m tests.test_opik_eval
```

결과는 Opik 대시보드에서 확인: https://opik-edu.didim365.app

## 참고 자료

- [LLM-as-a-judge 방법론 개요](https://deepeval.com/docs/metrics-llm-evals)
- [Opik 에이전트 평가 가이드](https://www.comet.com/docs/opik/evaluation/evaluate_agents)
