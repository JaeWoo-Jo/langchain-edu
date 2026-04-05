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
