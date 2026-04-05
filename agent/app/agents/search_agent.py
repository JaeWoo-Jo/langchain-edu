"""LangGraph 기반 가격 검색 서브 에이전트.

Match + Multi-match 병렬 검색 → 병합 → 포맷 파이프라인을 StateGraph로 구성하고,
@tool로 래핑하여 price_agent에 subagent-as-tool 패턴으로 통합한다.
"""

from __future__ import annotations

from typing import TypedDict

from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

from app.agents.tools._es_common import (
    CONTENT_FIELD,
    RAG_CONTENT_FIELD,
    RAG_VECTOR_FIELD,
    TOP_K,
    format_price_hits,
    get_embeddings,
    get_es_client,
    get_price_index,
    get_rag_index,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class SearchState(TypedDict):
    query: str                # 검색 쿼리
    match_hits: list[dict]    # 정확 매칭 검색 결과
    multi_hits: list[dict]    # 멀티 매칭 검색 결과 (item_name + kind_name)
    rag_hits: list[dict]      # RAG 문서 검색 결과
    merged_hits: list[dict]   # 병합 + 중복 제거된 결과
    result: str               # 최종 포맷팅된 문자열


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def match_search(state: SearchState) -> dict:
    """item_name 정확 매칭 검색."""
    es = get_es_client()
    try:
        resp = es.search(
            index=get_price_index(),
            body={
                "query": {
                    "match": {
                        CONTENT_FIELD: {
                            "query": state["query"],
                            "operator": "or",
                        }
                    }
                },
                "size": TOP_K,
            },
        )
        hits = resp["hits"]["hits"]
    except Exception:
        hits = []
    return {"match_hits": hits}


def multi_match_search(state: SearchState) -> dict:
    """item_name + kind_name 멀티 필드 검색 (넓은 범위)."""
    es = get_es_client()
    try:
        resp = es.search(
            index=get_price_index(),
            body={
                "query": {
                    "multi_match": {
                        "query": state["query"],
                        "fields": ["item_name", "kind_name"],
                        "type": "best_fields",
                        "minimum_should_match": "75%",
                    }
                },
                "size": TOP_K,
            },
        )
        hits = resp["hits"]["hits"]
    except Exception:
        hits = []
    return {"multi_hits": hits}


def rag_search(state: SearchState) -> dict:
    """RAG 문서 인덱스(edu-price-info)에서 BM25 + kNN 하이브리드 검색."""
    es = get_es_client()
    try:
        query_vector = get_embeddings().embed_query(state["query"])
        resp = es.search(
            index=get_rag_index(),
            body={
                "query": {
                    "match": {
                        RAG_CONTENT_FIELD: {
                            "query": state["query"],
                            "operator": "or",
                        }
                    }
                },
                "knn": {
                    "field": RAG_VECTOR_FIELD,
                    "query_vector": query_vector,
                    "k": TOP_K,
                    "num_candidates": TOP_K * 10,
                },
                "size": TOP_K,
            },
        )
        hits = resp["hits"]["hits"]
    except Exception:
        hits = []
    return {"rag_hits": hits}


def merge_results(state: SearchState) -> dict:
    """Match + Multi-match + RAG 검색 결과를 병합하고 중복을 제거한다.

    가격 데이터는 item_name + kind_name 기준으로 중복 제거하여 같은 품목이
    날짜만 다르게 여러 건 나오는 것을 방지한다. RAG 문서는 _id 기준.
    """
    seen_prices: set[str] = set()
    seen_rag: set[str] = set()
    merged: list[dict] = []

    for hit in state["match_hits"] + state["multi_hits"]:
        source = hit.get("_source", {})
        key = f"{source.get('item_name', '')}_{source.get('kind_name', '')}"
        if key and key not in seen_prices:
            seen_prices.add(key)
            merged.append(hit)

    for hit in state["rag_hits"]:
        doc_id = hit.get("_id", "")
        if doc_id and doc_id not in seen_rag:
            seen_rag.add(doc_id)
            merged.append(hit)

    return {"merged_hits": merged[:TOP_K]}


def format_results(state: SearchState) -> dict:
    """병합된 결과를 읽기 좋은 문자열로 포맷팅한다."""
    hits = state["merged_hits"]
    query = state["query"]

    if not hits:
        return {"result": f"'{query}'에 대한 정보를 찾을 수 없습니다."}

    price_hits = []
    rag_hits = []
    for hit in hits:
        source = hit.get("_source", {})
        if "metadata" in source and source["metadata"].get("source_type"):
            rag_hits.append(hit)
        else:
            price_hits.append(hit)

    parts: list[str] = []

    if price_hits:
        formatted = format_price_hits(price_hits)
        parts.append(f"■ 가격 검색 결과 (상위 {len(price_hits)}건)\n\n{formatted}")

    if rag_hits:
        rag_lines: list[str] = []
        for i, hit in enumerate(rag_hits, 1):
            source = hit["_source"]
            score = hit.get("_score", 0)
            content = source.get("content", "")[:300].replace("\n", " ")
            meta = source.get("metadata", {})
            source_type = meta.get("source_type", "")
            rag_lines.append(f"[{i}] ({source_type}) score={score:.4f}\n{content}")
        parts.append(f"■ 관련 문서 (상위 {len(rag_hits)}건)\n\n" + "\n\n".join(rag_lines))

    return {"result": "\n\n".join(parts)}


# ---------------------------------------------------------------------------
# Graph 구성
# ---------------------------------------------------------------------------


def _build_search_graph():
    builder = StateGraph(SearchState)

    builder.add_node("match_search", match_search)
    builder.add_node("multi_match_search", multi_match_search)
    builder.add_node("rag_search", rag_search)
    builder.add_node("merge_results", merge_results)
    builder.add_node("format_results", format_results)

    # 3-way 병렬 fan-out
    builder.add_edge(START, "match_search")
    builder.add_edge(START, "multi_match_search")
    builder.add_edge(START, "rag_search")

    # fan-in: 셋 다 완료 후 merge_results
    builder.add_edge("match_search", "merge_results")
    builder.add_edge("multi_match_search", "merge_results")
    builder.add_edge("rag_search", "merge_results")

    builder.add_edge("merge_results", "format_results")
    builder.add_edge("format_results", END)

    return builder.compile()


# 모듈 로드 시 한 번만 컴파일 (싱글턴)
_search_graph = _build_search_graph()


# ---------------------------------------------------------------------------
# Tool 래핑 (subagent as tool)
# ---------------------------------------------------------------------------


@tool
def search(search_query: str) -> str:
    """가격 + 레시피·영양·식재료 문서를 함께 검색합니다.
    요리 추천이나 식재료 정보가 필요할 때 사용."""
    result = _search_graph.invoke({"query": search_query})
    return result["result"]
