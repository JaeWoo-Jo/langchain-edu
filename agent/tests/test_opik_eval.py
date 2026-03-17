"""Opik 평가 테스트 - 100개 데이터셋 생성 및 에이전트 평가 실행"""
import os
import asyncio
import json
import uuid

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

from opik import Opik, evaluate
from opik.evaluation.metrics import base_metric, score_result
from typing import Any

# ============================================================
# 1. 100개 테스트 데이터셋 생성
# ============================================================

ITEMS = ["쌀", "찹쌀", "콩", "팥", "녹두", "고구마", "감자"]

def generate_dataset():
    """에이전트 도구 3종(검색/비교/차트)에 맞춘 100개 테스트 케이스 생성"""
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
        {"input": "오늘 뭐 해먹지?", "expected_tool": "none", "category": "일반대화", "item": ""},
    ]
    dataset_items.extend(edge_cases)

    return dataset_items[:100]  # 정확히 100개


# ============================================================
# 2. 평가 메트릭 정의
# ============================================================

class ToolUsageMetric(base_metric.BaseMetric):
    """에이전트가 올바른 도구를 호출했는지 확인하는 메트릭"""
    name = "tool_usage_accuracy"

    def score(self, output: str, expected_tool: str = "", **ignored_kwargs: Any):
        if expected_tool == "none":
            # 도구 호출 없이 직접 응답해야 하는 경우
            has_tool = any(t in output for t in ["search_price", "compare_prices", "create_price_chart"])
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

        elif category in ("일반대화", "복합질문", "에지케이스"):
            # 응답이 있으면 기본 점수
            score_val = 1.0 if len(output) > 10 else 0.5

        return score_result.ScoreResult(
            name=self.name, value=min(score_val, 1.0),
            reason=f"카테고리: {category}, 응답길이: {len(output)}자"
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

    all_chunks = []
    tool_calls = []
    final_content = ""

    async def _run():
        nonlocal final_content, tool_calls
        async for chunk_str in service.process_query(input_text, thread_id):
            all_chunks.append(chunk_str)
            try:
                chunk = json.loads(chunk_str)
                if chunk.get("step") == "model" and chunk.get("tool_calls"):
                    tool_calls.extend(chunk["tool_calls"])
                if chunk.get("step") == "done":
                    final_content = chunk.get("content", "")
            except json.JSONDecodeError:
                pass

    asyncio.run(_run())

    # 모델 원복
    if model_override:
        settings.OPENAI_MODEL = original_model

    return {
        "output": final_content,
        "tool_calls_str": ",".join(tool_calls),
        "all_chunks": json.dumps(all_chunks, ensure_ascii=False),
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
    models = os.environ.get("EVAL_MODELS", "gpt-4o").split(",")
    all_results = {}

    for model_name in models:
        model_name = model_name.strip()
        print(f"\n{'=' * 60}")
        print(f"2단계: 평가 실행 - 모델: {model_name}")
        print("=" * 60)

        def evaluation_task(item, _model=model_name):
            result = call_agent(item["input"], model_override=_model)
            return {
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
