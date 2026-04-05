"""2주차 금요일 과제: 성능 진단 리포트 자동 생성

Opik 서버에서 평가 데이터를 가져와 분석하고,
마크다운 리포트를 자동 생성한다.

사용법:
  cd agent/
  uv run python tests/generate_report.py

생성 결과:
  docs/week2-performance-report.md
"""
import os
import json
from datetime import datetime
from collections import defaultdict

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

from opik import Opik


# ============================================================
# 1. Opik에서 데이터 수집
# ============================================================

def fetch_traces(client: Opik, project_name: str, max_results: int = 200):
    """프로젝트의 트레이스를 가져온다"""
    try:
        traces = client.rest_client.traces.search_traces(
            project_name=project_name,
            max_results=max_results,
        )
        return traces
    except Exception as e:
        print(f"트레이스 조회 실패: {e}")
        return []


def fetch_experiment_results(client: Opik, dataset_name: str):
    """데이터셋의 실험 결과를 가져온다"""
    try:
        dataset = client.get_dataset(name=dataset_name)
        items = dataset.get_items()
        return items
    except Exception as e:
        print(f"데이터셋 조회 실패: {e}")
        return []


# ============================================================
# 2. 분석 함수
# ============================================================

def analyze_traces(traces) -> dict:
    """트레이스를 분석하여 통계를 생성한다"""
    stats = {
        "total": 0,
        "success": 0,
        "error": 0,
        "tool_calls": defaultdict(int),
        "avg_duration_ms": 0,
        "error_messages": [],
    }

    durations = []

    for trace in traces:
        stats["total"] += 1

        # 에러 여부 확인
        if hasattr(trace, "error") and trace.error:
            stats["error"] += 1
            stats["error_messages"].append(str(trace.error)[:100])
        else:
            stats["success"] += 1

        # 소요 시간
        if hasattr(trace, "duration") and trace.duration:
            durations.append(trace.duration)

        # 도구 호출 통계
        if hasattr(trace, "metadata") and trace.metadata:
            tool_calls = trace.metadata.get("tool_calls", [])
            for tool in tool_calls:
                stats["tool_calls"][tool] += 1

    if durations:
        stats["avg_duration_ms"] = sum(durations) / len(durations)

    return stats


def analyze_scores(traces) -> dict:
    """트레이스의 피드백 스코어를 분석한다"""
    scores_by_metric = defaultdict(list)

    for trace in traces:
        if hasattr(trace, "feedback_scores") and trace.feedback_scores:
            for score in trace.feedback_scores:
                name = score.get("name", score.name if hasattr(score, "name") else "unknown")
                value = score.get("value", score.value if hasattr(score, "value") else 0)
                scores_by_metric[name].append(value)

    # 메트릭별 통계 계산
    result = {}
    for metric, values in scores_by_metric.items():
        result[metric] = {
            "mean": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "count": len(values),
            "below_threshold": sum(1 for v in values if v < 0.7),
        }

    return result


def identify_weaknesses(score_stats: dict) -> list:
    """취약 지점을 식별한다"""
    weaknesses = []

    for metric, stats in score_stats.items():
        if stats["mean"] < 0.7:
            weaknesses.append({
                "metric": metric,
                "severity": "높음" if stats["mean"] < 0.5 else "중간",
                "mean": stats["mean"],
                "fail_rate": stats["below_threshold"] / stats["count"] * 100 if stats["count"] > 0 else 0,
                "recommendation": get_recommendation(metric, stats["mean"]),
            })

    # 심각도 순 정렬
    weaknesses.sort(key=lambda x: x["mean"])
    return weaknesses


def get_recommendation(metric: str, score: float) -> str:
    """메트릭별 개선 권고사항을 반환한다"""
    recommendations = {
        "tool_usage_accuracy": "프롬프트에 도구 선택 기준을 더 명확하게 명시하거나, few-shot 예시를 추가한다.",
        "response_quality": "시스템 프롬프트에 응답 형식 가이드라인을 강화한다.",
        "response_completeness": "도구 실행 결과를 응답에 더 충실하게 반영하도록 프롬프트를 개선한다.",
        "llm_judge": "전반적인 답변 품질 개선이 필요하다. 프롬프트 튜닝과 도구 출력 형식 개선을 병행한다.",
    }
    return recommendations.get(metric, "해당 메트릭의 채점 기준을 확인하고 에이전트 동작을 개선한다.")


# ============================================================
# 3. 리포트 생성
# ============================================================

def generate_report(trace_stats: dict, score_stats: dict, weaknesses: list, week: int = 2) -> str:
    """마크다운 리포트를 생성한다"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# {week}주차 에이전트 성능 진단 리포트

> 생성일시: {now}
> 프로젝트: jw-project
> Opik 대시보드: https://opik-edu.didim365.app

## 1. 개요

이 리포트는 Opik 트레이싱 데이터와 평가 메트릭 결과를 기반으로,
에이전트의 취약 지점을 분석하고 개선 우선순위를 도출한다.

### 평가 체계

| 파일 | 프레임워크 | 메트릭 유형 |
|------|-----------|------------|
| `test_opik_eval.py` | Opik | 규칙 기반 3종 + LLM-as-a-judge 1종 |
| `test_deepeval.py` | DeepEval | 에이전틱 표준 메트릭 3종 |

## 2. 트레이스 요약

| 항목 | 값 |
|------|-----|
| 전체 트레이스 수 | {trace_stats['total']} |
| 성공 | {trace_stats['success']} |
| 에러 | {trace_stats['error']} |
| 성공률 | {trace_stats['success'] / trace_stats['total'] * 100:.1f}% |
| 평균 응답 시간 | {trace_stats['avg_duration_ms']:.0f}ms |

### 도구 호출 분포

"""
    if trace_stats["tool_calls"]:
        report += "| 도구 | 호출 횟수 |\n|------|----------|\n"
        for tool, count in sorted(trace_stats["tool_calls"].items(), key=lambda x: -x[1]):
            report += f"| `{tool}` | {count} |\n"
    else:
        report += "_도구 호출 데이터가 트레이스 메타데이터에 기록되지 않았습니다._\n"

    report += f"""
### 에러 사례

"""
    if trace_stats["error_messages"]:
        for i, msg in enumerate(trace_stats["error_messages"][:5], 1):
            report += f"{i}. `{msg}`\n"
    else:
        report += "_에러 없음_\n"

    report += f"""
## 3. 메트릭별 성능

"""
    if score_stats:
        report += "| 메트릭 | 평균 | 최저 | 최고 | 평가 수 | 미달 수 (< 0.7) |\n"
        report += "|--------|------|------|------|---------|----------------|\n"
        for metric, stats in sorted(score_stats.items()):
            status = "🔴" if stats["mean"] < 0.5 else "🟡" if stats["mean"] < 0.7 else "🟢"
            report += (
                f"| {status} {metric} | {stats['mean']:.3f} | {stats['min']:.3f} | "
                f"{stats['max']:.3f} | {stats['count']} | {stats['below_threshold']} |\n"
            )
    else:
        report += "_아직 평가 데이터가 없습니다. `test_opik_eval.py`를 실행하세요._\n"

    report += f"""
## 4. 취약 지점 분석

"""
    if weaknesses:
        for i, w in enumerate(weaknesses, 1):
            report += f"""### 취약점 {i}: {w['metric']}

- **심각도**: {w['severity']}
- **평균 점수**: {w['mean']:.3f}
- **미달률**: {w['fail_rate']:.1f}%
- **개선 권고**: {w['recommendation']}

"""
    else:
        report += "_모든 메트릭이 임계값(0.7) 이상입니다. 취약 지점 없음._\n\n"

    report += f"""## 5. 개선 우선순위

"""
    if weaknesses:
        report += "| 우선순위 | 메트릭 | 심각도 | 현재 점수 | 목표 | 개선 방법 |\n"
        report += "|---------|--------|--------|----------|------|----------|\n"
        for i, w in enumerate(weaknesses, 1):
            report += f"| {i} | {w['metric']} | {w['severity']} | {w['mean']:.3f} | 0.70+ | {w['recommendation'][:30]}... |\n"
    else:
        report += "현재 모든 메트릭이 양호합니다. 다음 단계로 3주차 과제(LangGraph StateGraph 전환, 하이브리드 검색)를 진행하세요.\n"

    report += f"""
## 6. 다음 단계 제안

1. **즉시 조치**: 취약 지점으로 식별된 메트릭의 개선 권고사항을 적용한다
2. **프롬프트 튜닝**: 시스템 프롬프트(`app/agents/prompts.py`)를 개선한다
3. **도구 출력 개선**: 도구 반환값의 형식을 더 구조화한다
4. **다음 주차 연계**: 동일 평가를 재실행하여 개선 효과를 숫자로 확인한다

## 부록: 평가 실행 명령어

```bash
cd agent/

# Opik 평가 (115개 데이터셋)
uv run python -m tests.test_opik_eval

# DeepEval 평가 (13개 대표 케이스)
uv run deepeval test run tests/test_deepeval.py

# 리포트 생성 (주차 지정)
uv run python tests/generate_report.py --week {week}
```
"""

    return report


# ============================================================
# 4. 메인 실행
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="에이전트 성능 진단 리포트 생성")
    parser.add_argument("--week", type=int, default=3, help="리포트 주차 (기본: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"{args.week}주차 성능 진단 리포트 생성")
    print("=" * 60)

    client = Opik(
        host="https://opik-edu.didim365.app/api",
        project_name="jw-project",
        workspace="default",
    )

    # 트레이스 수집
    print("\n[1/4] 트레이스 수집 중...")
    traces = fetch_traces(client, "jw-project")
    print(f"  수집된 트레이스: {len(traces) if traces else 0}개")

    # 트레이스 분석
    print("[2/4] 트레이스 분석 중...")
    if traces:
        trace_stats = analyze_traces(traces)
        score_stats = analyze_scores(traces)
    else:
        print("  트레이스 없음 — 기본 템플릿으로 생성합니다.")
        trace_stats = {
            "total": 0, "success": 0, "error": 0,
            "tool_calls": {}, "avg_duration_ms": 0, "error_messages": [],
        }
        score_stats = {}

    # 취약점 식별
    print("[3/4] 취약 지점 식별 중...")
    weaknesses = identify_weaknesses(score_stats)
    if weaknesses:
        for w in weaknesses:
            print(f"  ⚠ {w['metric']}: {w['mean']:.3f} ({w['severity']})")
    else:
        print("  취약 지점 없음 (또는 데이터 부족)")

    # 리포트 생성
    print("[4/4] 리포트 생성 중...")
    report = generate_report(trace_stats, score_stats, weaknesses, week=args.week)

    output_path = os.path.join(os.path.dirname(__file__), "..", "docs", f"week{args.week}-performance-report.md")
    output_path = os.path.normpath(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n리포트 저장 완료: {output_path}")
    print(f"Opik 대시보드에서 상세 데이터 확인: https://opik-edu.didim365.app")


if __name__ == "__main__":
    main()
