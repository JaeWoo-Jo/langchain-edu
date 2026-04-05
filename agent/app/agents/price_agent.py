"""가격 정보 에이전트 팩토리.

create_agent + ToolStrategy(ChatResponse) 패턴으로 구조화 응답을 생성한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.search_agent import search
from app.agents.tools import search_price, compare_prices, create_price_chart
from app.agents.prompts import PRICE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------


@dataclass
class ChatResponse:
    """에이전트의 최종 응답 스키마."""

    message_id: str
    content: str
    metadata: dict[str, object]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_price_agent(model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None):
    """
    ChatOpenAI 모델과 checkpointer를 받아 가격 정보 에이전트를 생성합니다.

    Args:
        model: 초기화된 ChatOpenAI 인스턴스
        checkpointer: 대화 이력을 저장할 checkpointer 인스턴스 (MemorySaver 또는 AsyncSqliteSaver 등)

    Returns:
        create_agent()로 생성된 LangChain 에이전트
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[
            search,
            search_price,
            compare_prices,
            create_price_chart,
        ],
        system_prompt=PRICE_SYSTEM_PROMPT,
        response_format=ToolStrategy(ChatResponse),
        checkpointer=checkpointer,
    )
    return agent
