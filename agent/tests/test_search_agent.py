"""search_agent 단위 테스트.

3-way 병렬 검색(match, multi_match, rag), 중복 제거, 포맷팅을 검증한다.
"""

import pytest
from app.agents.search_agent import (
    match_search,
    multi_match_search,
    rag_search,
    merge_results,
    format_results,
    _search_graph,
)


# ---------------------------------------------------------------------------
# merge_results 단위 테스트
# ---------------------------------------------------------------------------


def _price_hit(item_name: str, kind_name: str, doc_id: str, score: float = 1.0) -> dict:
    """가격 검색 결과 히트를 생성하는 헬퍼."""
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {
            "item_name": item_name,
            "kind_name": kind_name,
            "price_today": 1000,
            "price_1week_ago": 900,
            "price_1month_ago": 800,
            "unit": "100g",
        },
    }


def _rag_hit(doc_id: str, content: str, source_type: str = "recipe", score: float = 3.0) -> dict:
    """RAG 검색 결과 히트를 생성하는 헬퍼."""
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {
            "content": content,
            "metadata": {"source_type": source_type, "category": "테스트"},
        },
    }


class TestMergeResults:
    """merge_results 노드 테스트."""

    def test_가격_중복_제거(self):
        """같은 item_name + kind_name 조합은 하나만 남긴다."""
        state = {
            "query": "감자",
            "match_hits": [
                _price_hit("감자", "수미(노지)", "id1", score=2.0),
                _price_hit("감자", "수미(노지)", "id2", score=1.0),  # 중복
            ],
            "multi_hits": [
                _price_hit("감자", "수미(시설)", "id3"),
            ],
            "rag_hits": [],
        }
        result = merge_results(state)
        names = [h["_source"]["kind_name"] for h in result["merged_hits"]]
        assert len(result["merged_hits"]) == 2
        assert "수미(노지)" in names
        assert "수미(시설)" in names

    def test_RAG_중복_제거(self):
        """같은 _id의 RAG 문서는 하나만 남긴다."""
        state = {
            "query": "감자",
            "match_hits": [],
            "multi_hits": [],
            "rag_hits": [
                _rag_hit("rag1", "감자 볶음밥 레시피"),
                _rag_hit("rag1", "감자 볶음밥 레시피"),  # 중복
                _rag_hit("rag2", "된장국 레시피"),
            ],
        }
        result = merge_results(state)
        assert len(result["merged_hits"]) == 2

    def test_가격과_RAG_혼합(self):
        """가격 결과와 RAG 결과가 함께 병합된다."""
        state = {
            "query": "감자",
            "match_hits": [_price_hit("감자", "수미(노지)", "p1")],
            "multi_hits": [],
            "rag_hits": [_rag_hit("r1", "감자 요리")],
        }
        result = merge_results(state)
        assert len(result["merged_hits"]) == 2

    def test_TOP_K_제한(self):
        """결과가 TOP_K를 초과하지 않는다."""
        state = {
            "query": "감자",
            "match_hits": [_price_hit("감자", f"종류{i}", f"id{i}") for i in range(15)],
            "multi_hits": [],
            "rag_hits": [],
        }
        result = merge_results(state)
        from app.agents.tools._es_common import TOP_K
        assert len(result["merged_hits"]) <= TOP_K


# ---------------------------------------------------------------------------
# format_results 단위 테스트
# ---------------------------------------------------------------------------


class TestFormatResults:
    """format_results 노드 테스트."""

    def test_빈_결과(self):
        """결과가 없으면 안내 메시지를 반환한다."""
        state = {"query": "없는품목", "merged_hits": []}
        result = format_results(state)
        assert "찾을 수 없습니다" in result["result"]

    def test_가격_결과_포맷(self):
        """가격 결과에 가격 검색 결과 헤더가 포함된다."""
        state = {
            "query": "감자",
            "merged_hits": [_price_hit("감자", "수미(노지)", "p1")],
        }
        result = format_results(state)
        assert "가격 검색 결과" in result["result"]

    def test_RAG_결과_포맷(self):
        """RAG 결과에 관련 문서 헤더가 포함된다."""
        state = {
            "query": "감자",
            "merged_hits": [_rag_hit("r1", "감자 볶음밥 레시피")],
        }
        result = format_results(state)
        assert "관련 문서" in result["result"]


# ---------------------------------------------------------------------------
# 통합 테스트 (ES 연결 필요)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSearchGraphIntegration:
    """search_agent 그래프 통합 테스트. ES 연결 필요."""

    def test_감자_검색(self):
        """감자 검색 시 가격 + RAG 결과가 모두 나온다."""
        result = _search_graph.invoke({"query": "감자"})
        assert result["result"]
        assert "감자" in result["result"]

    def test_빈_검색어(self):
        """빈 검색어도 에러 없이 처리된다."""
        result = _search_graph.invoke({"query": ""})
        assert isinstance(result["result"], str)
