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


# ---------------------------------------------------------------------------
# 노드 함수 테스트
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 그래프 조립 테스트
# ---------------------------------------------------------------------------


class TestCreateDeepAgent:
    def test_그래프가_정상_컴파일(self):
        from app.agents.deep_agent import create_deep_agent

        mock_model = MagicMock()
        graph = create_deep_agent(mock_model)
        assert graph is not None

    def test_checkpointer_전달(self):
        from app.agents.deep_agent import create_deep_agent
        from langgraph.checkpoint.base import BaseCheckpointSaver

        mock_model = MagicMock()
        mock_checkpointer = MagicMock(spec=BaseCheckpointSaver)
        graph = create_deep_agent(mock_model, checkpointer=mock_checkpointer)
        assert graph is not None

    def test_그래프_노드_목록(self):
        from app.agents.deep_agent import create_deep_agent

        mock_model = MagicMock()
        graph = create_deep_agent(mock_model)
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"__start__", "__end__", "planner", "executor", "tools", "reflector", "advance_step", "synthesizer"}
        assert expected.issubset(node_names)
