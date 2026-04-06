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
from app.agents.tools import search_price, compare_prices, create_price_chart, search_nutrition


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


MAX_REPLAN = 1  # replan 최대 허용 횟수


class DeepAgentState(TypedDict):
    """딥에이전트 상태.

    - messages: 대화 이력 (add_messages 리듀서로 자동 누적)
    - plan: 남은 실행 단계 목록
    - current_step: 현재 실행 중인 단계 설명
    - step_results: 각 단계 실행 결과 요약 (operator.add 리듀서로 자동 누적)
    - replan_count: replan 횟수 추적
    - response: 최종 응답 텍스트
    """

    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step: str
    step_results: Annotated[list[str], operator.add]
    replan_count: int
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

TOOLS = [search, search_price, compare_prices, create_price_chart, search_nutrition]


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
# Nodes
# ---------------------------------------------------------------------------


def advance_step(state: DeepAgentState) -> dict:
    """다음 단계로 이동한다."""
    plan = list(state.get("plan", []))
    if not plan:
        return {"current_step": ""}
    next_step = plan.pop(0)
    return {"plan": plan, "current_step": next_step}


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
        " | ".join(recent_results) if recent_results else "결과 없음"
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

    current_replan = state.get("replan_count", 0)

    if result.action == "replan" and result.revised_remaining_steps:
        if current_replan < MAX_REPLAN:
            update["plan"] = result.revised_remaining_steps
            update["replan_count"] = current_replan + 1
        else:
            # replan 한도 초과 → 강제 done
            update["plan"] = []
    elif result.action == "done":
        update["plan"] = []

    return update


def synthesizer(state: DeepAgentState, model: ChatOpenAI) -> dict:
    """모든 결과를 종합하여 최종 응답을 생성한다."""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_content = user_msgs[-1].content if user_msgs else ""

    # step_results 외에 실제 ToolMessage 내용도 수집하여 데이터 유실 방지
    tool_results = [
        f"[{m.name}] {m.content[:500]}"
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    ]
    results = "\n".join(tool_results) if tool_results else "\n".join(state.get("step_results", []))

    response = model.invoke([
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=f"사용자 질문: {user_content}\n\n수집된 정보:\n{results}"),
    ])

    return {"response": response.content}


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
