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
# Nodes
# ---------------------------------------------------------------------------


def advance_step(state: DeepAgentState) -> dict:
    """다음 단계로 이동한다."""
    plan = list(state.get("plan", []))
    if not plan:
        return {"current_step": ""}
    next_step = plan.pop(0)
    return {"plan": plan, "current_step": next_step}
