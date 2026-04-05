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
