# 딥에이전트 (Plan-Execute-Reflect) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 현재 단일 ReAct 에이전트를 Plan-Execute-Reflect 다단계 추론 아키텍처로 교체한다.

**Architecture:** LangGraph StateGraph로 Planner(계획) → Executor(실행) → Reflector(평가) 루프를 구성한다. Executor는 기존 도구들(search, search_price, compare_prices, create_price_chart)을 재사용하며, ToolNode를 통해 도구를 실행한다. Reflector가 결과를 평가하여 다음 단계 진행/계획 수정/완료를 결정하고, Synthesizer가 최종 응답을 생성한다.

**Tech Stack:** LangGraph 1.0.6+, LangChain 1.0+, langchain-openai 1.0+, Pydantic v2

---

## 그래프 구조

```
START → planner → executor ⟷ tools (ReAct 루프)
                      ↓ (도구 호출 없음)
                  reflector → advance_step → executor (다음 단계)
                      ↓ (남은 단계 없음)
                  synthesizer → END
```

## 파일 구조

| 동작 | 파일 | 역할 |
|------|------|------|
| 생성 | `agent/app/agents/deep_agent.py` | 딥에이전트 그래프 전체 (State, 노드, 라우팅, 팩토리) |
| 생성 | `agent/tests/test_deep_agent.py` | 노드 단위 테스트 + 그래프 컴파일 테스트 |
| 수정 | `agent/app/agents/prompts.py` | Planner/Executor/Reflector/Synthesizer 프롬프트 추가 |
| 수정 | `agent/app/agents/price_agent.py` | create_agent → create_deep_agent로 교체 |
| 수정 | `agent/app/services/agent_service.py` | _parse_chunk, _create_agent, recursion_limit 업데이트 |
| 수정 | `agent/app/core/config.py` | DEEPAGENT_RECURSION_LIMIT 기본값 40으로 변경 |

---

### Task 1: 딥에이전트 프롬프트 추가

**Files:**
- Modify: `agent/app/agents/prompts.py`

- [ ] **Step 1: prompts.py에 4개 프롬프트 추가**

`agent/app/agents/prompts.py` 파일 끝에 추가:

```python
# ---------------------------------------------------------------------------
# 딥에이전트 프롬프트
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """당신은 가격 정보 분석 전문 플래너입니다.
사용자의 질문을 분석하여, 답변에 필요한 실행 단계를 수립하세요.

사용 가능한 도구:
- search: 가격 정보 통합 검색 (품목명, 품종명 등)
- search_price: 특정 품목의 현재 가격 상세 조회
- compare_prices: 품목의 기간별 가격 비교 (1주, 2주, 1개월)
- create_price_chart: 품목의 가격 추이 차트 생성

규칙:
- 각 단계는 하나의 구체적 작업(도구 호출 1회)에 대응하도록 작성
- 간단한 질문 → 1~2단계, 비교/분석 → 3~5단계
- 최대 6단계 이내
- 검색 → 비교 → 차트 순서가 자연스러움"""

EXECUTOR_PROMPT = """당신은 계획된 단계를 실행하는 실행자입니다.
주어진 단계에 적절한 도구를 호출하세요.

규칙:
- 현재 단계에 필요한 도구를 정확히 하나 호출하세요.
- 도구 결과가 돌아오면 간단히 요약하세요.
- 추가 도구 호출이 필요 없으면 텍스트로만 응답하세요."""

REFLECTOR_PROMPT = """당신은 실행 결과를 평가하는 평가자입니다.

평가 기준:
- 도구 결과에 유효한 데이터가 있는가?
- 결과가 사용자 질문에 답하는 데 유용한가?
- 빈 결과나 오류가 있는가?

행동 결정:
- continue: 결과가 적절하고 남은 단계가 있을 때
- replan: 결과가 불충분하여 남은 계획 수정이 필요할 때 (검색어 변경 등)
- done: 남은 단계 없이 충분한 정보가 수집되었을 때

replan 시 revised_remaining_steps에 수정된 남은 단계를 작성하세요."""

SYNTHESIZER_PROMPT = """당신은 자취고수! 수집된 정보를 종합하여 답변하는 전문가야.

규칙:
- 반말로 친근하게 답변해 ㅋㅋ
- 가격은 구체적인 수치를 포함
- 비교 분석이 있으면 핵심 인사이트를 강조
- 간결하고 실용적인 답변
- 가끔 ㅋㅋ, ㅠㅠ 같은 이모티콘 사용"""
```

- [ ] **Step 2: 임포트 확인**

Run: `cd agent && uv run python -c "from app.agents.prompts import PLANNER_PROMPT, EXECUTOR_PROMPT, REFLECTOR_PROMPT, SYNTHESIZER_PROMPT; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 커밋**

```bash
cd agent && git add app/agents/prompts.py
git commit -m "feat: 딥에이전트 프롬프트 4종 추가 (Planner/Executor/Reflector/Synthesizer)"
```

---

### Task 2: DeepAgentState, Pydantic 모델, 라우팅 함수 + 테스트

**Files:**
- Create: `agent/app/agents/deep_agent.py`
- Create: `agent/tests/test_deep_agent.py`

- [ ] **Step 1: 라우팅 함수 테스트 작성**

`agent/tests/test_deep_agent.py` 생성:

```python
"""딥에이전트 단위 테스트."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# 테스트 헬퍼
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    """테스트용 기본 DeepAgentState 딕셔너리 생성."""
    base = {
        "messages": [HumanMessage(content="배추 가격 알려줘")],
        "plan": [],
        "current_step": "",
        "step_results": [],
        "response": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 라우팅 함수 테스트
# ---------------------------------------------------------------------------

class TestAfterExecutor:
    def test_도구호출이_있으면_tools로_라우팅(self):
        from app.agents.deep_agent import after_executor

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"search_query": "배추"}, "id": "1"}],
        )
        state = _make_state(messages=[HumanMessage(content=""), msg])
        assert after_executor(state) == "tools"

    def test_도구호출이_없으면_reflector로_라우팅(self):
        from app.agents.deep_agent import after_executor

        msg = AIMessage(content="배추 가격을 찾았습니다.")
        state = _make_state(messages=[HumanMessage(content=""), msg])
        assert after_executor(state) == "reflector"


class TestAfterReflector:
    def test_남은_계획이_있으면_advance_step으로_라우팅(self):
        from app.agents.deep_agent import after_reflector

        state = _make_state(plan=["무 가격 검색"])
        assert after_reflector(state) == "advance_step"

    def test_남은_계획이_없으면_synthesizer로_라우팅(self):
        from app.agents.deep_agent import after_reflector

        state = _make_state(plan=[])
        assert after_reflector(state) == "synthesizer"


class TestAdvanceStep:
    def test_다음_단계를_꺼내고_plan에서_제거(self):
        from app.agents.deep_agent import advance_step

        state = _make_state(plan=["단계2", "단계3"], current_step="단계1")
        result = advance_step(state)
        assert result["current_step"] == "단계2"
        assert result["plan"] == ["단계3"]

    def test_마지막_단계(self):
        from app.agents.deep_agent import advance_step

        state = _make_state(plan=["단계2"], current_step="단계1")
        result = advance_step(state)
        assert result["current_step"] == "단계2"
        assert result["plan"] == []

    def test_빈_계획(self):
        from app.agents.deep_agent import advance_step

        state = _make_state(plan=[], current_step="단계1")
        result = advance_step(state)
        assert result["current_step"] == ""
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.agents.deep_agent'`

- [ ] **Step 3: deep_agent.py 기본 구조 작성**

`agent/app/agents/deep_agent.py` 생성:

```python
"""딥에이전트: Plan-Execute-Reflect 아키텍처.

사용자 질문을 [계획 수립 → 단계별 실행 → 자체 평가] 루프로 처리하는
다단계 추론 파이프라인.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field

from app.agents.prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    REFLECTOR_PROMPT,
    SYNTHESIZER_PROMPT,
)
from app.agents.search_agent import search
from app.agents.tools import search_price, compare_prices, create_price_chart


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class DeepAgentState(TypedDict):
    """딥에이전트 상태.

    - messages: 대화 이력 (add_messages 리듀서로 자동 누적)
    - plan: 남은 실행 단계 목록
    - current_step: 현재 실행 중인 단계 설명
    - step_results: 각 단계 실행 결과 요약 (operator.add 리듀서로 자동 누적)
    - response: 최종 응답 텍스트
    """

    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step: str
    step_results: Annotated[list[str], operator.add]
    response: str


# ---------------------------------------------------------------------------
# Structured Output Models
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    """실행 계획."""

    steps: list[str] = Field(description="순서대로 실행할 단계 목록")


class Reflection(BaseModel):
    """실행 결과 평가."""

    evaluation: str = Field(description="현재 단계 실행 결과 평가")
    action: Literal["continue", "replan", "done"] = Field(
        description="continue=다음 단계, replan=계획 수정, done=완료",
    )
    revised_remaining_steps: list[str] | None = Field(
        default=None,
        description="action=replan일 때: 수정된 남은 단계 목록",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [search, search_price, compare_prices, create_price_chart]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def after_executor(state: DeepAgentState) -> Literal["tools", "reflector"]:
    """도구 호출이 있으면 tools, 없으면 reflector로 라우팅."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "reflector"


def after_reflector(state: DeepAgentState) -> Literal["advance_step", "synthesizer"]:
    """남은 단계가 있으면 advance_step, 없으면 synthesizer로 라우팅."""
    if state.get("plan"):
        return "advance_step"
    return "synthesizer"


# ---------------------------------------------------------------------------
# Nodes (플레이스홀더 — Task 3~5에서 구현)
# ---------------------------------------------------------------------------


def advance_step(state: DeepAgentState) -> dict:
    """다음 단계로 이동한다."""
    plan = list(state.get("plan", []))
    if not plan:
        return {"current_step": ""}
    next_step = plan.pop(0)
    return {"plan": plan, "current_step": next_step}
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py -v`
Expected: 7 PASSED

- [ ] **Step 5: 커밋**

```bash
cd agent && git add app/agents/deep_agent.py tests/test_deep_agent.py
git commit -m "feat: DeepAgentState, Pydantic 모델, 라우팅 함수 + 단위 테스트"
```

---

### Task 3: Planner 노드 구현 + 테스트

**Files:**
- Modify: `agent/app/agents/deep_agent.py`
- Modify: `agent/tests/test_deep_agent.py`

- [ ] **Step 1: Planner 테스트 추가**

`agent/tests/test_deep_agent.py` 끝에 추가:

```python
from unittest.mock import MagicMock


class TestPlanner:
    def _make_mock_model(self, steps: list[str]):
        from app.agents.deep_agent import Plan

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = Plan(steps=steps)
        return mock_model

    def test_여러_단계_계획_수립(self):
        from app.agents.deep_agent import planner

        model = self._make_mock_model(["배추 가격 검색", "무 가격 검색", "비교"])
        state = _make_state()
        result = planner(state, model)

        assert result["current_step"] == "배추 가격 검색"
        assert result["plan"] == ["무 가격 검색", "비교"]
        assert len(result["messages"]) == 1
        assert "3단계" in result["messages"][0].content

    def test_단일_단계_계획(self):
        from app.agents.deep_agent import planner

        model = self._make_mock_model(["배추 가격 검색"])
        state = _make_state()
        result = planner(state, model)

        assert result["current_step"] == "배추 가격 검색"
        assert result["plan"] == []

    def test_이전_결과가_있으면_프롬프트에_포함(self):
        from app.agents.deep_agent import planner

        model = self._make_mock_model(["재검색"])
        state = _make_state(step_results=["배추: 데이터 없음"])
        planner(state, model)

        call_args = model.with_structured_output.return_value.invoke.call_args
        prompt_content = call_args[0][0][-1].content
        assert "배추: 데이터 없음" in prompt_content
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py::TestPlanner -v`
Expected: FAIL — `ImportError` (planner 함수가 아직 없음)

- [ ] **Step 3: Planner 노드 구현**

`agent/app/agents/deep_agent.py`의 `# Nodes` 섹션에 추가 (advance_step 위에):

```python
def planner(state: DeepAgentState, model: ChatOpenAI) -> dict:
    """사용자 질문을 분석하여 실행 계획을 수립한다."""
    planner_llm = model.with_structured_output(Plan)

    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_content = user_msgs[-1].content if user_msgs else ""

    prompt = f"사용자 질문: {user_content}"
    if state.get("step_results"):
        prompt += "\n\n이전 실행 결과:\n" + "\n".join(state["step_results"])

    result = planner_llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=prompt),
    ])

    steps = result.steps
    first_step = steps[0] if steps else ""
    remaining = steps[1:] if len(steps) > 1 else []

    plan_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))

    return {
        "plan": remaining,
        "current_step": first_step,
        "messages": [AIMessage(content=f"[계획 수립] {len(steps)}단계:\n{plan_text}")],
    }
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py::TestPlanner -v`
Expected: 3 PASSED

- [ ] **Step 5: 커밋**

```bash
cd agent && git add app/agents/deep_agent.py tests/test_deep_agent.py
git commit -m "feat: Planner 노드 구현 — 사용자 질문을 실행 계획으로 분해"
```

---

### Task 4: Executor 노드 구현 + 테스트

**Files:**
- Modify: `agent/app/agents/deep_agent.py`
- Modify: `agent/tests/test_deep_agent.py`

- [ ] **Step 1: Executor 테스트 추가**

`agent/tests/test_deep_agent.py` 끝에 추가:

```python
class TestExecutor:
    def test_도구_호출_응답_반환(self):
        from app.agents.deep_agent import executor

        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind_tools.return_value = mock_bound

        response_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_price", "args": {"item_name": "배추"}, "id": "tc1"}],
        )
        mock_bound.invoke.return_value = response_msg

        state = _make_state(current_step="배추 가격 검색")
        result = executor(state, mock_model)

        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls[0]["name"] == "search_price"

    def test_도구_없이_텍스트_응답(self):
        from app.agents.deep_agent import executor

        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind_tools.return_value = mock_bound

        response_msg = AIMessage(content="배추 가격 검색 완료")
        mock_bound.invoke.return_value = response_msg

        state = _make_state(current_step="결과 정리")
        result = executor(state, mock_model)

        assert result["messages"][0].content == "배추 가격 검색 완료"

    def test_이전_결과가_프롬프트에_포함(self):
        from app.agents.deep_agent import executor

        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind_tools.return_value = mock_bound
        mock_bound.invoke.return_value = AIMessage(content="ok")

        state = _make_state(
            current_step="무 가격 검색",
            step_results=["배추: 2,500원/포기"],
        )
        executor(state, mock_model)

        call_args = mock_bound.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "무 가격 검색" in system_content
        assert "배추: 2,500원/포기" in system_content
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py::TestExecutor -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Executor 노드 구현**

`agent/app/agents/deep_agent.py`의 planner 함수 뒤에 추가:

```python
def executor(state: DeepAgentState, model: ChatOpenAI) -> dict:
    """현재 단계를 실행하기 위해 도구를 호출한다."""
    executor_llm = model.bind_tools(TOOLS)

    step = state.get("current_step", "")
    results_ctx = "\n".join(state.get("step_results", []))

    instruction = f"현재 실행할 단계: {step}"
    if results_ctx:
        instruction += f"\n\n이전 단계 실행 결과:\n{results_ctx}"

    response = executor_llm.invoke(
        [SystemMessage(content=EXECUTOR_PROMPT + "\n\n" + instruction)]
        + list(state["messages"])
    )

    return {"messages": [response]}
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py::TestExecutor -v`
Expected: 3 PASSED

- [ ] **Step 5: 커밋**

```bash
cd agent && git add app/agents/deep_agent.py tests/test_deep_agent.py
git commit -m "feat: Executor 노드 구현 — 계획 단계별 도구 호출"
```

---

### Task 5: Reflector + Synthesizer 노드 구현 + 테스트

**Files:**
- Modify: `agent/app/agents/deep_agent.py`
- Modify: `agent/tests/test_deep_agent.py`

- [ ] **Step 1: Reflector + Synthesizer 테스트 추가**

`agent/tests/test_deep_agent.py` 끝에 추가:

```python
class TestReflector:
    def _make_mock_model(self, action, revised=None):
        from app.agents.deep_agent import Reflection

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = Reflection(
            evaluation="테스트 평가",
            action=action,
            revised_remaining_steps=revised,
        )
        return mock_model

    def test_continue_시_step_results_누적(self):
        from app.agents.deep_agent import reflector

        model = self._make_mock_model("continue")
        state = _make_state(
            current_step="배추 가격 검색",
            messages=[
                HumanMessage(content="배추 가격"),
                AIMessage(content="", tool_calls=[{"name": "search_price", "args": {}, "id": "1"}]),
                ToolMessage(content="배추 2,500원", tool_call_id="1", name="search_price"),
            ],
        )
        result = reflector(state, model)

        assert len(result["step_results"]) == 1
        assert "배추 가격 검색" in result["step_results"][0]

    def test_replan_시_plan_교체(self):
        from app.agents.deep_agent import reflector

        model = self._make_mock_model("replan", revised=["고구마 검색", "비교"])
        state = _make_state(
            current_step="배추 검색",
            plan=["무 검색"],
            messages=[
                HumanMessage(content="test"),
                ToolMessage(content="결과 없음", tool_call_id="1", name="search"),
            ],
        )
        result = reflector(state, model)

        assert result["plan"] == ["고구마 검색", "비교"]

    def test_done_시_plan_비움(self):
        from app.agents.deep_agent import reflector

        model = self._make_mock_model("done")
        state = _make_state(
            current_step="차트 생성",
            plan=["불필요한 단계"],
            messages=[
                HumanMessage(content="test"),
                ToolMessage(content="차트 데이터", tool_call_id="1", name="create_price_chart"),
            ],
        )
        result = reflector(state, model)

        assert result["plan"] == []


class TestSynthesizer:
    def test_최종_응답_생성(self):
        from app.agents.deep_agent import synthesizer

        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="배추 지금 2,500원이야!")

        state = _make_state(step_results=["배추: 2,500원/포기"])
        result = synthesizer(state, mock_model)

        assert result["response"] == "배추 지금 2,500원이야!"

    def test_사용자_질문이_프롬프트에_포함(self):
        from app.agents.deep_agent import synthesizer

        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="답변")

        state = _make_state(
            messages=[HumanMessage(content="배추 무 비교해줘")],
            step_results=["결과1"],
        )
        synthesizer(state, mock_model)

        call_args = mock_model.invoke.call_args[0][0]
        user_content = call_args[-1].content
        assert "배추 무 비교해줘" in user_content
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py::TestReflector tests/test_deep_agent.py::TestSynthesizer -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Reflector 노드 구현**

`agent/app/agents/deep_agent.py`의 executor 함수 뒤에 추가:

```python
def reflector(state: DeepAgentState, model: ChatOpenAI) -> dict:
    """실행 결과를 평가하고 다음 행동을 결정한다."""
    reflector_llm = model.with_structured_output(Reflection)

    # 최근 도구 결과 추출
    recent_results: list[str] = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            recent_results.insert(0, f"[{msg.name}] {msg.content[:300]}")
        elif isinstance(msg, AIMessage) and "[계획" in getattr(msg, "content", ""):
            break

    step = state.get("current_step", "")
    step_summary = f"{step}: " + (
        recent_results[-1] if recent_results else "결과 없음"
    )

    context = (
        f"현재 단계: {step}\n"
        f"남은 단계: {state.get('plan', [])}\n"
        f"이 단계 도구 결과:\n" + "\n".join(recent_results)
    )

    result = reflector_llm.invoke([
        SystemMessage(content=REFLECTOR_PROMPT),
        HumanMessage(content=context),
    ])

    update: dict = {
        "step_results": [step_summary],
        "messages": [AIMessage(content=f"[평가] {result.evaluation}")],
    }

    if result.action == "replan" and result.revised_remaining_steps:
        update["plan"] = result.revised_remaining_steps
    elif result.action == "done":
        update["plan"] = []

    return update
```

- [ ] **Step 4: Synthesizer 노드 구현**

`agent/app/agents/deep_agent.py`의 advance_step 함수 뒤에 추가:

```python
def synthesizer(state: DeepAgentState, model: ChatOpenAI) -> dict:
    """모든 결과를 종합하여 최종 응답을 생성한다."""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_content = user_msgs[-1].content if user_msgs else ""
    results = "\n".join(state.get("step_results", []))

    response = model.invoke([
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=f"사용자 질문: {user_content}\n\n수집된 정보:\n{results}"),
    ])

    return {"response": response.content}
```

- [ ] **Step 5: 테스트 실행 — 통과 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py::TestReflector tests/test_deep_agent.py::TestSynthesizer -v`
Expected: 5 PASSED

- [ ] **Step 6: 전체 테스트 실행**

Run: `cd agent && uv run pytest tests/test_deep_agent.py -v`
Expected: 15 PASSED (라우팅 7 + Planner 3 + Executor 3 + Reflector 3 + Synthesizer 2 - 중복 제거하여 대략 15개)

- [ ] **Step 7: 커밋**

```bash
cd agent && git add app/agents/deep_agent.py tests/test_deep_agent.py
git commit -m "feat: Reflector + Synthesizer 노드 구현 — 결과 평가 및 최종 응답 생성"
```

---

### Task 6: 그래프 조립 + create_deep_agent 팩토리 + 테스트

**Files:**
- Modify: `agent/app/agents/deep_agent.py`
- Modify: `agent/tests/test_deep_agent.py`

- [ ] **Step 1: 그래프 컴파일 테스트 추가**

`agent/tests/test_deep_agent.py` 끝에 추가:

```python
class TestCreateDeepAgent:
    def test_그래프가_정상_컴파일(self):
        from app.agents.deep_agent import create_deep_agent

        mock_model = MagicMock()
        graph = create_deep_agent(mock_model)
        assert graph is not None

    def test_checkpointer_전달(self):
        from app.agents.deep_agent import create_deep_agent

        mock_model = MagicMock()
        mock_checkpointer = MagicMock()
        graph = create_deep_agent(mock_model, checkpointer=mock_checkpointer)
        assert graph is not None

    def test_그래프_노드_목록(self):
        from app.agents.deep_agent import create_deep_agent

        mock_model = MagicMock()
        graph = create_deep_agent(mock_model)
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", "planner", "executor", "tools", "reflector", "advance_step", "synthesizer"}
        assert expected.issubset(node_names)
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py::TestCreateDeepAgent -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: create_deep_agent 팩토리 구현**

`agent/app/agents/deep_agent.py` 파일 끝에 추가:

```python
# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------


def create_deep_agent(
    model: ChatOpenAI,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Plan-Execute-Reflect 딥에이전트 그래프를 생성한다."""
    tool_node = ToolNode(TOOLS)

    builder = StateGraph(DeepAgentState)

    # 노드 등록
    builder.add_node("planner", lambda s: planner(s, model))
    builder.add_node("executor", lambda s: executor(s, model))
    builder.add_node("tools", tool_node)
    builder.add_node("reflector", lambda s: reflector(s, model))
    builder.add_node("advance_step", advance_step)
    builder.add_node("synthesizer", lambda s: synthesizer(s, model))

    # 엣지 연결
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", after_executor)
    builder.add_edge("tools", "executor")
    builder.add_conditional_edges("reflector", after_reflector)
    builder.add_edge("advance_step", "executor")
    builder.add_edge("synthesizer", END)

    # 컴파일
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(**compile_kwargs)
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `cd agent && uv run pytest tests/test_deep_agent.py -v`
Expected: ALL PASSED

- [ ] **Step 5: 커밋**

```bash
cd agent && git add app/agents/deep_agent.py tests/test_deep_agent.py
git commit -m "feat: 딥에이전트 그래프 조립 — Plan-Execute-Reflect 파이프라인 완성"
```

---

### Task 7: agent_service.py + price_agent.py + config 업데이트

**Files:**
- Modify: `agent/app/services/agent_service.py`
- Modify: `agent/app/agents/price_agent.py`
- Modify: `agent/app/core/config.py`

- [ ] **Step 1: config.py — DEEPAGENT_RECURSION_LIMIT 기본값 변경**

`agent/app/core/config.py`에서 변경:

```python
# 변경 전:
DEEPAGENT_RECURSION_LIMIT: int = 20

# 변경 후:
DEEPAGENT_RECURSION_LIMIT: int = 40
```

- [ ] **Step 2: price_agent.py — create_deep_agent로 교체**

`agent/app/agents/price_agent.py` 전체를 교체:

```python
"""가격 정보 딥에이전트 팩토리.

create_deep_agent를 래핑하여 기존 인터페이스(create_price_agent)를 유지한다.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.deep_agent import create_deep_agent


def create_price_agent(
    model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None
):
    """
    ChatOpenAI 모델과 checkpointer를 받아 딥에이전트를 생성한다.

    Args:
        model: 초기화된 ChatOpenAI 인스턴스
        checkpointer: 대화 이력 저장용 checkpointer

    Returns:
        Plan-Execute-Reflect 딥에이전트 (LangGraph CompiledGraph)
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
    return create_deep_agent(model=model, checkpointer=checkpointer)
```

- [ ] **Step 3: agent_service.py — _RECURSION_LIMIT을 config에서 가져오기**

`agent/app/services/agent_service.py` 상단 변경:

```python
# 변경 전:
# 도구 호출 총 횟수 제한 (ReAct 루프: 도구 1회 = LLM노드 + Tool노드 = 2 step)
_RECURSION_LIMIT = 15

# 변경 후:
from app.core.config import settings as _settings
# Plan-Execute-Reflect: 단계당 ~5 노드 사용, 기본 40
_RECURSION_LIMIT = _settings.DEEPAGENT_RECURSION_LIMIT
```

- [ ] **Step 4: agent_service.py — _create_agent 수정**

`agent/app/services/agent_service.py`의 `_create_agent` 메서드 변경:

```python
    def _create_agent(self):
        """LangChain 딥에이전트 생성"""
        from app.agents.price_agent import create_price_agent

        assert self.checkpointer is not None, (
            "checkpointer가 초기화되지 않았습니다. "
            "_init_checkpointer를 먼저 호출하세요."
        )
        self.agent = create_price_agent(
            model=self.model,
            checkpointer=self.checkpointer,
        )

        # Opik LangGraph 트래킹 적용
        if self.opik_tracer is not None:
            from opik.integrations.langchain import track_langgraph

            self.agent = track_langgraph(self.agent, self.opik_tracer)
```

- [ ] **Step 5: agent_service.py — _parse_chunk 교체**

`agent/app/services/agent_service.py`의 `_parse_chunk` 메서드를 교체:

```python
    def _parse_chunk(self, chunk: dict):
        """딥에이전트 스트림 청크를 SSE 이벤트 문자열 리스트로 변환한다."""
        events: list[str] = []

        for node_name, event in chunk.items():
            if not event:
                continue

            messages = event.get("messages", [])

            # --- 계획 수립 ---
            if node_name == "planner":
                plan = event.get("plan", [])
                current = event.get("current_step", "")
                all_steps = ([current] + plan) if current else plan
                events.append(json.dumps({
                    "step": "plan",
                    "plan": all_steps,
                }, ensure_ascii=False))

            # --- 도구 호출 (Executor LLM 응답) ---
            elif node_name == "executor":
                if not messages:
                    continue
                message = messages[0]
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    events.append(json.dumps({
                        "step": "model",
                        "tool_calls": [tc["name"] for tc in tool_calls],
                    }))

            # --- 도구 실행 결과 ---
            elif node_name == "tools":
                if not messages:
                    continue
                message = messages[0]
                events.append(
                    f'{{"step": "tools", "name": {json.dumps(message.name)}, '
                    f'"content": {message.content}}}'
                )

            # --- 실행 결과 평가 ---
            elif node_name == "reflector":
                content = ""
                if messages:
                    content = getattr(messages[0], "content", "")
                events.append(json.dumps({
                    "step": "reflect",
                    "content": content,
                }, ensure_ascii=False))

            # --- 최종 응답 ---
            elif node_name == "synthesizer":
                response = event.get("response", "")
                events.append(self._done_event(
                    content=response,
                    metadata=self._handle_metadata(event.get("metadata")),
                ))

        return events
```

- [ ] **Step 6: 임포트 확인**

Run: `cd agent && uv run python -c "from app.agents.price_agent import create_price_agent; print('OK')"`
Expected: `OK`

- [ ] **Step 7: 커밋**

```bash
cd agent && git add app/core/config.py app/agents/price_agent.py app/services/agent_service.py
git commit -m "feat: agent_service + price_agent를 딥에이전트로 전환"
```

---

### Task 8: 기존 테스트 수정 + 통합 테스트

**Files:**
- Modify: `agent/tests/test_agent_service.py`
- Modify: `agent/tests/test_deep_agent.py`

- [ ] **Step 1: test_agent_service.py 업데이트**

기존 테스트가 `create_react_agent`를 mock하고 있으므로 `create_price_agent`로 변경:

```python
from unittest.mock import patch, MagicMock, AsyncMock


def test_agent_created():
    """_create_agent()가 에이전트를 정상 생성하는지 확인"""
    with patch("app.services.agent_service.ChatOpenAI"), \
         patch("app.services.agent_service._settings") as mock_settings:

        mock_settings.OPENAI_MODEL = "gpt-4o"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPIK = None
        mock_settings.DEEPAGENT_RECURSION_LIMIT = 40

        from app.services.agent_service import AgentService
        service = AgentService()

        # checkpointer를 수동 설정
        service.checkpointer = MagicMock()

        with patch("app.agents.price_agent.create_deep_agent") as mock_create:
            mock_create.return_value = MagicMock()
            service._create_agent()
            mock_create.assert_called_once()
```

- [ ] **Step 2: _parse_chunk 단위 테스트 추가**

`agent/tests/test_deep_agent.py` 끝에 추가:

```python
class TestParseChunk:
    """agent_service._parse_chunk가 딥에이전트 청크를 올바르게 파싱하는지 테스트."""

    def _make_service(self):
        with patch("app.services.agent_service.ChatOpenAI"), \
             patch("app.services.agent_service._settings") as mock_settings:
            mock_settings.OPENAI_MODEL = "gpt-4o"
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.OPIK = None
            mock_settings.DEEPAGENT_RECURSION_LIMIT = 40
            from app.services.agent_service import AgentService
            return AgentService()

    def test_planner_청크_파싱(self):
        import json
        service = self._make_service()
        chunk = {
            "planner": {
                "plan": ["무 검색"],
                "current_step": "배추 검색",
                "messages": [AIMessage(content="[계획 수립] 2단계")],
            }
        }
        events = service._parse_chunk(chunk)
        assert len(events) == 1
        parsed = json.loads(events[0])
        assert parsed["step"] == "plan"
        assert parsed["plan"] == ["배추 검색", "무 검색"]

    def test_executor_도구호출_파싱(self):
        import json
        service = self._make_service()
        chunk = {
            "executor": {
                "messages": [AIMessage(
                    content="",
                    tool_calls=[{"name": "search_price", "args": {}, "id": "1"}],
                )]
            }
        }
        events = service._parse_chunk(chunk)
        assert len(events) == 1
        parsed = json.loads(events[0])
        assert parsed["step"] == "model"
        assert "search_price" in parsed["tool_calls"]

    def test_synthesizer_done_이벤트_파싱(self):
        import json
        service = self._make_service()
        chunk = {
            "synthesizer": {
                "response": "배추 가격 2,500원이야ㅋㅋ",
            }
        }
        events = service._parse_chunk(chunk)
        assert len(events) == 1
        parsed = json.loads(events[0])
        assert parsed["step"] == "done"
        assert "2,500원" in parsed["content"]
```

- [ ] **Step 3: 전체 테스트 실행**

Run: `cd agent && uv run pytest tests/test_deep_agent.py tests/test_agent_service.py -v`
Expected: ALL PASSED

- [ ] **Step 4: 커밋**

```bash
cd agent && git add tests/test_agent_service.py tests/test_deep_agent.py
git commit -m "test: 딥에이전트 전환에 따른 기존 테스트 수정 + _parse_chunk 단위 테스트"
```

---

## 후속 작업 (이 계획 범위 밖)

- **프론트엔드**: `step: "plan"`, `step: "reflect"` SSE 이벤트를 UI에 표시
- **성능 최적화**: 간단한 질문은 Planner가 1단계 계획을 만들어 오버헤드를 최소화
- **DeepEval 평가**: 딥에이전트 전환 전후 답변 품질 비교
- **Opik 트레이싱**: 딥에이전트 노드별 실행 시간 모니터링
