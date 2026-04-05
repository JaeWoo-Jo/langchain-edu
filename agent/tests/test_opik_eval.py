"""Opik 평가 테스트 - 데이터셋 생성 및 에이전트 평가 실행"""
import os
import asyncio
import json
import uuid

# Opik evaluate()가 async로 돌 때 asyncio.run() 중첩 허용
import nest_asyncio
nest_asyncio.apply()

# Opik 환경변수 설정
os.environ["OPIK_URL_OVERRIDE"] = "https://opik-edu.didim365.app/api"
os.environ["OPIK_PROJECT_NAME"] = "jw-project"
os.environ["OPIK_WORKSPACE"] = "default"

import opik
opik.configure(
    url="https://opik-edu.didim365.app/api",
    workspace="default",
    use_local=True,
)

from openai import OpenAI
from opik import Opik, evaluate
from opik.evaluation.metrics import base_metric, score_result
from typing import Any

# ============================================================
# 1. 100개 테스트 데이터셋 생성
# ============================================================

ITEMS = ["쌀", "찹쌀", "콩", "팥", "녹두", "고구마", "감자"]

def generate_dataset():
    """에이전트 도구 4종(검색/통합검색/비교/차트)에 맞춘 115개 테스트 케이스 생성"""
    dataset_items = []

    # === 가격 검색 질문 (search_price) - 35개 ===
    search_templates = [
        "{item} 가격 알려줘",
        "{item} 얼마야?",
        "{item} 시세 좀 알려줘",
        "오늘 {item} 가격이 어떻게 돼?",
        "{item} 현재 가격 좀 알려줄래?",
    ]
    for i, item in enumerate(ITEMS):
        for j, tmpl in enumerate(search_templates):
            dataset_items.append({
                "input": tmpl.format(item=item),
                "expected_tool": "search_price",
                "category": "가격검색",
                "item": item,
            })

    # === 가격 비교 질문 (compare_prices) - 35개 ===
    compare_templates = [
        "{item} 가격 비교해줘",
        "{item} 기간별 가격 변동 알려줘",
        "{item} 1주간 가격 추이 보여줘",
        "{item} 가격이 올랐어 내렸어?",
        "{item} 최근 가격 변동 테이블로 보여줘",
    ]
    for i, item in enumerate(ITEMS):
        for j, tmpl in enumerate(compare_templates):
            dataset_items.append({
                "input": tmpl.format(item=item),
                "expected_tool": "compare_prices",
                "category": "가격비교",
                "item": item,
            })

    # === 차트 질문 (create_price_chart) - 21개 ===
    chart_templates = [
        "{item} 가격 차트 보여줘",
        "{item} 가격 추이 그래프 만들어줘",
        "{item} 1년간 가격 변동 차트 보여줘",
    ]
    for i, item in enumerate(ITEMS):
        for j, tmpl in enumerate(chart_templates):
            dataset_items.append({
                "input": tmpl.format(item=item),
                "expected_tool": "create_price_chart",
                "category": "차트생성",
                "item": item,
            })

    # === 통합 검색 질문 (search) - 15개 ===
    # 레시피·영양·식재료 관련 질문 → search 서브에이전트 호출
    rag_questions = [
        {"input": "감자로 뭐 해먹을 수 있어?", "item": "감자"},
        {"input": "고구마 요리 추천해줘", "item": "고구마"},
        {"input": "콩으로 만들 수 있는 요리 알려줘", "item": "콩"},
        {"input": "감자 영양성분 알려줘", "item": "감자"},
        {"input": "고구마 칼로리 얼마야?", "item": "고구마"},
        {"input": "단백질 많은 식재료 뭐 있어?", "item": ""},
        {"input": "된장국 만드는 법 알려줘", "item": ""},
        {"input": "볶음밥 레시피 추천해줘", "item": ""},
        {"input": "자취생 간단 요리 추천", "item": ""},
        {"input": "감자 보관법 알려줘", "item": "감자"},
        {"input": "고구마 삶는 법", "item": "고구마"},
        {"input": "오늘 저녁 싸게 해먹을 수 있는 메뉴 추천해줘", "item": ""},
        {"input": "쌀이랑 감자 중에 탄수화물 뭐가 많아?", "item": "쌀,감자"},
        {"input": "다이어트할 때 뭐 먹으면 좋아?", "item": ""},
        {"input": "팥으로 디저트 만들 수 있어?", "item": "팥"},
    ]
    for q in rag_questions:
        dataset_items.append({
            "input": q["input"],
            "expected_tool": "search",
            "category": "통합검색",
            "item": q["item"],
        })

    # === 복합 질문 - 5개 ===
    complex_questions = [
        {"input": "쌀이랑 콩 가격 비교해줘", "expected_tool": "search_price", "category": "복합질문", "item": "쌀,콩"},
        {"input": "감자 가격 알려주고 차트도 보여줘", "expected_tool": "search_price", "category": "복합질문", "item": "감자"},
        {"input": "고구마 시세 확인하고 1년 추이 그래프도 그려줘", "expected_tool": "search_price", "category": "복합질문", "item": "고구마"},
        {"input": "찹쌀이랑 쌀 중에 뭐가 더 비싸?", "expected_tool": "search_price", "category": "복합질문", "item": "찹쌀,쌀"},
        {"input": "팥이랑 녹두 가격 비교 테이블이랑 차트 둘 다 보여줘", "expected_tool": "compare_prices", "category": "복합질문", "item": "팥,녹두"},
    ]
    dataset_items.extend(complex_questions)

    # === 일반 대화 / 에지케이스 - 4개 ===
    edge_cases = [
        {"input": "안녕!", "expected_tool": "none", "category": "일반대화", "item": ""},
        {"input": "자취할 때 절약 팁 알려줘", "expected_tool": "none", "category": "일반대화", "item": ""},
        {"input": "존재하지않는품목 가격 알려줘", "expected_tool": "search_price", "category": "에지케이스", "item": "존재하지않는품목"},
        {"input": "오늘 뭐 해먹지?", "expected_tool": "search", "category": "통합검색", "item": ""},
    ]
    dataset_items.extend(edge_cases)

    return dataset_items  # 115개


# ============================================================
# 2. 평가 메트릭 정의
# ============================================================

class ToolUsageMetric(base_metric.BaseMetric):
    """에이전트가 올바른 도구를 호출했는지 확인하는 메트릭"""
    name = "tool_usage_accuracy"

    ALL_TOOLS = ["search_price", "search", "compare_prices", "create_price_chart"]

    def score(self, output: str, expected_tool: str = "", **ignored_kwargs: Any):
        if expected_tool == "none":
            # 도구 호출 없이 직접 응답해야 하는 경우
            has_tool = any(f"tools:{t}" in output or f",{t}" in output for t in self.ALL_TOOLS)
            return score_result.ScoreResult(
                name=self.name,
                value=1.0 if not has_tool else 0.5,
                reason="도구 호출 불필요 케이스" if not has_tool else "도구를 호출했지만 불필요했을 수 있음"
            )

        if expected_tool in output:
            return score_result.ScoreResult(
                name=self.name, value=1.0,
                reason=f"올바른 도구 호출: {expected_tool}"
            )
        return score_result.ScoreResult(
            name=self.name, value=0.0,
            reason=f"기대 도구: {expected_tool}, 실제 출력에서 미발견"
        )


class ResponseQualityMetric(base_metric.BaseMetric):
    """응답의 기본 품질을 확인하는 메트릭"""
    name = "response_quality"

    def score(self, output: str, item: str = "", **ignored_kwargs: Any):
        if not output or output.strip() == "":
            return score_result.ScoreResult(
                name=self.name, value=0.0, reason="빈 응답"
            )

        # 에러 응답 체크
        if "오류가 발생했습니다" in output:
            return score_result.ScoreResult(
                name=self.name, value=0.0, reason="에러 응답"
            )

        score_val = 0.5  # 기본 점수

        # 한글 응답인지 체크
        korean_chars = sum(1 for c in output if '\uac00' <= c <= '\ud7a3')
        if korean_chars > 5:
            score_val += 0.25

        # 품목 관련 질문인 경우 가격 정보 포함 여부
        if item and item != "존재하지않는품목":
            if "원" in output:
                score_val += 0.25

        return score_result.ScoreResult(
            name=self.name, value=min(score_val, 1.0),
            reason=f"응답 길이: {len(output)}자, 한글포함: {korean_chars > 5}, 가격정보: {'원' in output}"
        )


class ResponseCompletenessMetric(base_metric.BaseMetric):
    """응답의 완결성을 확인하는 메트릭"""
    name = "response_completeness"

    def score(self, output: str, category: str = "", **ignored_kwargs: Any):
        if not output:
            return score_result.ScoreResult(
                name=self.name, value=0.0, reason="빈 응답"
            )

        score_val = 0.0

        if category == "가격검색":
            # 가격 정보가 포함되어야 함
            if "원" in output:
                score_val += 0.5
            if any(kw in output for kw in ["대비", "전", "가격"]):
                score_val += 0.5

        elif category == "가격비교":
            if "원" in output:
                score_val += 0.5
            if any(kw in output for kw in ["TABLE_DATA", "테이블", "비교", "올랐", "내렸", "변동"]):
                score_val += 0.5

        elif category == "차트생성":
            if any(kw in output for kw in ["CHART_DATA", "차트", "그래프", "추이"]):
                score_val += 0.5
            if "원" in output:
                score_val += 0.5

        elif category == "통합검색":
            # 레시피/영양/식재료 관련 응답
            if any(kw in output for kw in ["레시피", "요리", "만들", "재료", "조리", "영양", "칼로리"]):
                score_val += 0.5
            if len(output) > 50:
                score_val += 0.5

        elif category in ("일반대화", "복합질문", "에지케이스"):
            # 응답이 있으면 기본 점수
            score_val = 1.0 if len(output) > 10 else 0.5

        return score_result.ScoreResult(
            name=self.name, value=min(score_val, 1.0),
            reason=f"카테고리: {category}, 응답길이: {len(output)}자"
        )


class LLMJudgeMetric(base_metric.BaseMetric):
    """GPT-4o를 판사로 사용하여 답변을 4가지 기준으로 채점하는 메트릭 (LLM-as-a-judge)

    채점 기준:
        1. 정확성(accuracy)     - 사실에 기반한 정확한 정보인가?
        2. 유용성(usefulness)   - 사용자가 원하는 정보를 담고 있는가?
        3. 완결성(completeness) - 빠진 정보 없이 충분한가?
        4. 자연스러움(naturalness) - 한국어 표현이 자연스러운가?

    각 기준 1~5점, 평균을 0.0~1.0으로 정규화하여 Opik에 기록.
    """
    name = "llm_judge"

    SYSTEM_PROMPT = """당신은 AI 에이전트의 답변 품질을 평가하는 전문 평가자입니다.

아래 사용자 질문과 에이전트 답변을 보고, 4가지 기준 각각을 1~5점으로 채점하세요.

## 채점 기준
1. **정확성**: 질문에 대해 사실에 기반한 정확한 정보를 제공하는가?
2. **유용성**: 사용자가 실제로 원하는 정보를 잘 담고 있는가?
3. **완결성**: 답변이 충분히 완전한가? 빠진 정보는 없는가?
4. **한국어 자연스러움**: 한국어 표현이 자연스럽고 읽기 쉬운가?

## 점수 기준
- 5점: 매우 우수
- 4점: 우수
- 3점: 보통
- 2점: 미흡
- 1점: 매우 미흡

반드시 아래 JSON 형식으로만 응답하세요:
{"accuracy": <1-5>, "usefulness": <1-5>, "completeness": <1-5>, "naturalness": <1-5>, "reason": "<한줄 총평>"}"""

    def __init__(self):
        from app.core.config import settings
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def score(self, output: str, input: str = "", **ignored_kwargs: Any):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"## 사용자 질문\n{input}\n\n## 에이전트 답변\n{output}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            result = json.loads(response.choices[0].message.content)
            avg = (
                result["accuracy"]
                + result["usefulness"]
                + result["completeness"]
                + result["naturalness"]
            ) / 4
            normalized = avg / 5.0  # 1~5점 → 0.0~1.0

            reason = (
                f"정확성:{result['accuracy']} 유용성:{result['usefulness']} "
                f"완결성:{result['completeness']} 자연스러움:{result['naturalness']} "
                f"| {result['reason']}"
            )
            return score_result.ScoreResult(
                name=self.name, value=normalized, reason=reason,
            )
        except Exception as e:
            return score_result.ScoreResult(
                name=self.name, value=0.0, reason=f"LLM 채점 실패: {str(e)}",
            )


# ============================================================
# 3. 에이전트 호출 래퍼
# ============================================================

def call_agent(input_text: str, model_override: str = None) -> dict:
    """에이전트를 호출하고 전체 응답을 수집"""
    from app.services.agent_service import AgentService
    from app.core.config import settings

    # 모델 오버라이드
    if model_override:
        original_model = settings.OPENAI_MODEL
        settings.OPENAI_MODEL = model_override

    service = AgentService()
    thread_id = uuid.uuid4()

    tool_calls = []
    final_content = ""

    async def _run():
        nonlocal final_content, tool_calls
        async for chunk_str in service.process_query(input_text, thread_id):
            try:
                chunk = json.loads(chunk_str)
            except (json.JSONDecodeError, TypeError):
                continue
            step = chunk.get("step", "")
            if step == "model" and chunk.get("tool_calls"):
                tool_calls.extend(chunk["tool_calls"])
            if step == "tools" and chunk.get("name"):
                tool_calls.append(chunk["name"])
            if step == "done":
                final_content = chunk.get("content", "")

    asyncio.run(_run())

    # 모델 원복
    if model_override:
        settings.OPENAI_MODEL = original_model

    return {
        "output": final_content,
        "tool_calls_str": ",".join(tool_calls),
    }


# ============================================================
# 4. 메인 실행
# ============================================================

def main():
    client = Opik(
        host="https://opik-edu.didim365.app/api",
        project_name="jw-project",
        workspace="default",
    )

    # 데이터셋 생성
    print("=" * 60)
    print("1단계: 100개 테스트 데이터셋 생성")
    print("=" * 60)

    test_data = generate_dataset()
    print(f"생성된 테스트 케이스: {len(test_data)}개")

    # 카테고리별 분포
    from collections import Counter
    cat_dist = Counter(d["category"] for d in test_data)
    for cat, cnt in cat_dist.items():
        print(f"  {cat}: {cnt}개")

    dataset = client.get_or_create_dataset(name="jw-agent-eval-100")
    dataset.insert(test_data)
    print(f"\nOpik 데이터셋 'jw-agent-eval-100' 등록 완료")

    # 평가할 모델 목록
    models = os.environ.get("EVAL_MODELS", "gpt-4.1-mini").split(",")
    all_results = {}

    for model_name in models:
        model_name = model_name.strip()
        print(f"\n{'=' * 60}")
        print(f"2단계: 평가 실행 - 모델: {model_name}")
        print("=" * 60)

        def evaluation_task(item, _model=model_name):
            result = call_agent(item["input"], model_override=_model)
            return {
                "input": item["input"],  # LLM 판사가 원��� 질문을 볼 수 있도록 전달
                "output": f"[tools:{result['tool_calls_str']}] {result['output']}",
                "expected_tool": item.get("expected_tool", ""),
                "category": item.get("category", ""),
                "item": item.get("item", ""),
            }

        result = evaluate(
            dataset=dataset,
            task=evaluation_task,
            scoring_metrics=[
                ToolUsageMetric(),
                ResponseQualityMetric(),
                ResponseCompletenessMetric(),
                LLMJudgeMetric(),
            ],
            experiment_name=f"jw-agent-eval-{model_name}",
            project_name="jw-project",
        )

        scores = result.aggregate_evaluation_scores()
        all_results[model_name] = scores

        print(f"\n--- {model_name} 평가 결과 ---")
        for metric_name, stats in scores.aggregated_scores.items():
            print(f"  {metric_name}: mean={stats.mean:.4f}, std={stats.std:.4f}")

    # 최종 비교 결과
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("최종 모델 비교")
        print("=" * 60)
        print(f"{'메트릭':<30}", end="")
        for m in models:
            print(f"{m.strip():<15}", end="")
        print()
        print("-" * (30 + 15 * len(models)))

        metric_names = list(next(iter(all_results.values())).aggregated_scores.keys())
        for metric in metric_names:
            print(f"{metric:<30}", end="")
            for m in models:
                m = m.strip()
                stats = all_results[m].aggregated_scores[metric]
                print(f"{stats.mean:<15.4f}", end="")
            print()

    print(f"\n평가 완료! Opik 대시보드에서 상세 결과를 확인하세요:")
    print(f"  https://opik-edu.didim365.app")


if __name__ == "__main__":
    main()
