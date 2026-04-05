"""2주차 목요일 과제: DeepEval 라이브러리를 활용한 에이전틱 평가

DeepEval의 핵심 메트릭 3종을 사용하여 에이전트를 평가한다:
  1. TaskCompletionMetric - 사용자 요청을 완수했는가?
  2. ToolCorrectnessMetric - 올바른 도구��� 선택했는가?
  3. GEval - 커스텀 기준(한국어 품질)으로 채점

실행 방법:
  cd agent/
  uv run deepeval test run tests/test_deepeval.py
"""
import os
import json
import asyncio
import uuid

# ============================================================
# 1. 환경 설정
# ============================================================

# DeepEval이 LLM 판사를 사용하기 위해 API 키 필요
# app.core.config에서 .env 파일의 키를 가져와 환경변수에 설정
from app.core.config import settings
os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)

from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden, EvaluationDataset, get_current_golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval import evaluate


# ============================================================
# 2. 에이전트를 DeepEval의 @observe로 래핑
# ============================================================
#
# DeepEval은 @observe 데코레이터로 ��이전트의 실행 흐름을 추적한다.
# - @observe(type="agent") : 최상위 에이전트
# - @observe(type="tool")  : 개별 도구 호출
# - @observe(type="llm")   : LLM 호출 (도구 선택 판단)
#
# 이렇게 추적된 정보를 TaskCompletionMetric, ToolCorrectnessMetric이
# 자동으로 분석하여 점수를 매긴다.

@observe(type="tool")
def search_price_traced(item_name: str) -> str:
    """search_price 도구를 DeepEval 추적과 함께 호출"""
    from app.agents.tools import search_price
    return search_price.invoke({"item_name": item_name})


@observe(type="tool")
def compare_prices_traced(item_name: str, period: str = "1주") -> str:
    """compare_prices 도구를 DeepEval 추적과 함께 호출"""
    from app.agents.tools import compare_prices
    return compare_prices.invoke({"item_name": item_name, "period": period})


@observe(type="tool")
def create_price_chart_traced(item_name: str) -> str:
    """create_price_chart 도구를 DeepEval 추적과 함께 호출"""
    from app.agents.tools import create_price_chart
    return create_price_chart.invoke({"item_name": item_name})


@observe(type="agent")
def run_agent(user_input: str) -> str:
    """에이전트를 실행하고 최종 응답�� 반환

    실제 LangChain 에이전트를 호출하여 도구 선택/실행/응답 생성 전 과정을 수행한다.
    DeepEval의 @observe가 이 과정을 자동으로 추적한다.
    """
    from app.services.agent_service import AgentService

    service = AgentService()
    thread_id = uuid.uuid4()

    final_content = ""
    tool_calls = []

    async def _run():
        nonlocal final_content, tool_calls
        async for chunk_str in service.process_query(user_input, thread_id):
            try:
                chunk = json.loads(chunk_str)
                if chunk.get("step") == "model" and chunk.get("tool_calls"):
                    tool_calls.extend(chunk["tool_calls"])
                if chunk.get("step") == "done":
                    final_content = chunk.get("content", "")
            except json.JSONDecodeError:
                pass

    asyncio.run(_run())

    return final_content


# ============================================================
# 3. 메트릭 정의
# ============================================================

# (A) TaskCompletionMetric — 사용자 요청을 완수했는가?
#     LLM이 에이전트의 전체 실행 흐름을 보고 "요청이 완료되었는가"를 판단한다.
#     threshold=0.7 → 70% 이상이면 통과
task_completion = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-4.1-mini",
)

# (B) ToolCorrectnessMetric — 올바른 도구를 선택했는가?
#     expected_tools에 지정한 도구와 실제 호출된 도구를 비교한다.
#     예: "쌀 가격 알려줘" → expected: search_price, actual: search_price → 통과
tool_correctness = ToolCorrectnessMetric(
    threshold=0.7,
)

# (C) GEval — 카테고리별 한국어 응답 품질
#     G-Eval 논문(Liu et al., 2023) 기반.
#
#     [개선] 기존에는 단일 GEval로 모든 케이스를 채점했지만,
#     "안녕!" 같은 일반 대화에도 "가격 정보 포함 여부"를 체크하여 부당 감점이 발생했다.
#     → 가격 질문용 / 일반 대화용으로 분리하여 각 카테고리에 맞는 기준을 적용한다.

korean_quality_price = GEval(
    name="한국어 응답 품질 (가격)",
    criteria="가격 관련 질문에 대한 에이전트 응답의 정확성과 자연스러움을 평가",
    evaluation_steps=[
        "사용자의 질문 의도를 정확히 파악했는지 확인한다",
        "답변에 포함된 가격 정보가 구체적인 숫자와 단위를 포함하는지 확인한다",
        "한국어 표현이 자연스럽고 문법적으로 올바른지 확인한다",
        "불필요한 영어 혼용이나 기계적 표현이 없는지 확인한다",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4.1-mini",
    threshold=0.7,
)

korean_quality_general = GEval(
    name="한국어 응답 품질 (일반)",
    criteria="일반 대화 또는 에지케이스에 대한 에이전트 응답의 적절성과 자연스러움을 평가",
    evaluation_steps=[
        "사용자의 질문 의도를 정확히 파악했는지 확인한다",
        "질문에 맞는 적절한 응답을 했는지 확인한다 (존재하지 않는 품목이면 없다고 안내, 인사면 인사 응답)",
        "한국어 표현이 자연스럽고 문법적으로 올바른지 확인한다",
        "불필요한 영어 혼용이나 기계적 표현이 없는지 확인한다",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4.1-mini",
    threshold=0.7,
)

# (D) GEval — RAG 통합검색 응답 품질
#     레시피·영양·식재료 질문에 대한 답변이 검색된 문서를 잘 활용하는지 평가
korean_quality_rag = GEval(
    name="한국어 응답 품질 (RAG)",
    criteria="레시피, 영양정보, 식재료 관련 질문에 대한 에이전트 응답의 정확성과 유용성을 평가",
    evaluation_steps=[
        "사용자의 질문 의도를 정확히 파악했는지 확인한다",
        "답변에 구체적인 레시피, 영양성분, 또는 식재료 정보가 포함되어 있는지 확인한다",
        "답변이 실용적이고 자취생에게 도움이 되는 내용인지 확인한다",
        "한국어 표현이 자연스럽고 문법적으로 올바른지 확인한다",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4.1-mini",
    threshold=0.7,
)


# ============================================================
# 4. 테스트 데이터셋
# ============================================================
#
# DeepEval의 Golden = 하나의 테스트 케이스 정의
#   - input: 사용자 질문
#   - expected_tools: 호출되어야 할 도구 목록 (ToolCorrectnessMetric용)

# 카테고리별로 분류: "price"(가격 질문) / "general"(일반 대화·에지케이스)
# 각 카테고리에 맞는 GEval 메트릭이 적용된다.

PRICE_CASES = [
    # === 가격 검색 (search_price) ===
    Golden(input="쌀 가격 알려줘", expected_tools=[ToolCall(name="search_price")]),
    Golden(input="감자 얼마야?", expected_tools=[ToolCall(name="search_price")]),
    Golden(input="오늘 콩 가격이 어떻게 돼?", expected_tools=[ToolCall(name="search_price")]),
    # === 가격 비교 (compare_prices) ===
    Golden(input="고구마 가격 비교해줘", expected_tools=[ToolCall(name="compare_prices")]),
    Golden(input="팥 기간별 가격 변동 알려줘", expected_tools=[ToolCall(name="compare_prices")]),
    # === 차트 생성 (create_price_chart) ===
    Golden(input="녹두 가격 차트 보여줘", expected_tools=[ToolCall(name="create_price_chart")]),
    Golden(input="찹쌀 1년간 가격 변동 차트 보여줘", expected_tools=[ToolCall(name="create_price_chart")]),
    # === 복합 질문 ===
    Golden(input="쌀이랑 콩 가격 비교해줘", expected_tools=[ToolCall(name="search_price")]),
]

RAG_CASES = [
    # === 레시피 검색 (search) ===
    Golden(input="감자로 뭐 해먹을 수 있어?", expected_tools=[ToolCall(name="search")]),
    # === 영양 정보 (search) ===
    Golden(input="고구마 칼로리 얼마야?", expected_tools=[ToolCall(name="search")]),
    # === 식재료 추천 (search) ===
    Golden(input="오늘 저녁 싸게 해먹을 수 있는 메뉴 추천해줘", expected_tools=[ToolCall(name="search")]),
]

GENERAL_CASES = [
    # === 일반 대화 (도구 호출 불필요) ===
    Golden(input="안녕!", expected_tools=[]),
    # === 에지 케이스 ===
    Golden(input="존재하지않는품목 가격 알려줘", expected_tools=[ToolCall(name="search_price")]),
]

TEST_CASES = PRICE_CASES + RAG_CASES + GENERAL_CASES


# ============================================================
# 5. 평가 실행
# ============================================================

def run_and_build(goldens: list) -> list:
    """Golden 리스트를 에이전트 실행 후 LLMTestCase 리스트로 변환한다"""
    test_cases = []
    for golden in goldens:
        print(f"  실행 중: {golden.input}")
        output = run_agent(golden.input)
        tc = LLMTestCase(
            input=golden.input,
            actual_output=output,
            expected_tools=golden.expected_tools,
        )
        test_cases.append(tc)
    return test_cases


def main():
    """DeepEval 평가를 직접 실행 — 카테고리별 GEval 분리 적용"""
    print("=" * 60)
    print("DeepEval 에이전틱 평가 시작")
    print(f"테스트 케이스: {len(TEST_CASES)}개 (가격 {len(PRICE_CASES)} + RAG {len(RAG_CASES)} + 일반 {len(GENERAL_CASES)})")
    print("=" * 60)

    # 에이전트 실행
    print("\n[1단계] 에이전트 실행...")
    price_tc = run_and_build(PRICE_CASES)
    rag_tc = run_and_build(RAG_CASES)
    general_tc = run_and_build(GENERAL_CASES)

    # 가격 질문 → korean_quality_price 메트릭 적용
    print("\n[2단계] 가격 케이스 채점...")
    price_results = evaluate(
        test_cases=price_tc,
        metrics=[task_completion, korean_quality_price],
    )

    # RAG 통합검색 → korean_quality_rag 메트릭 적용
    print("\n[3단계] RAG 케이스 채점...")
    rag_results = evaluate(
        test_cases=rag_tc,
        metrics=[task_completion, korean_quality_rag],
    )

    # 일반 대화 → korean_quality_general 메트릭 적용
    print("\n[4단계] 일반/에지 케이스 채점...")
    general_results = evaluate(
        test_cases=general_tc,
        metrics=[task_completion, korean_quality_general],
    )

    print("\n평가 완료!")


if __name__ == "__main__":
    main()
