"""Elasticsearch 검색 테스트 스크립트 (BM25 / Vector / Hybrid + Cohere ReRanker)"""

import argparse
import urllib3
import cohere

from index_mapping import get_es_client
from embedder import embed_texts
from config import ES_INDEX, COHERE_API_KEY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def bm25_search(es, query: str, top_k: int = 5) -> list[dict]:
    """BM25 텍스트 검색"""
    resp = es.search(
        index=ES_INDEX,
        body={
            "query": {"match": {"content": query}},
            "size": top_k,
        },
    )
    return resp["hits"]["hits"]


def vector_search(es, query: str, top_k: int = 5) -> list[dict]:
    """Vector(kNN) 검색"""
    query_vector = embed_texts([query])[0]
    resp = es.search(
        index=ES_INDEX,
        body={
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
            },
            "size": top_k,
        },
    )
    return resp["hits"]["hits"]


def hybrid_search(es, query: str, top_k: int = 5) -> list[dict]:
    """BM25 + Vector 하이브리드 검색 (score 합산 방식)"""
    query_vector = embed_texts([query])[0]
    resp = es.search(
        index=ES_INDEX,
        body={
            "query": {"match": {"content": query}},
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
            },
            "size": top_k,
        },
    )
    return resp["hits"]["hits"]


def rerank(query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
    """Cohere Rerank API로 검색 결과를 재정렬한다."""
    if not COHERE_API_KEY:
        print("  [경고] COHERE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return hits

    client = cohere.Client(api_key=COHERE_API_KEY)
    documents = [hit["_source"]["content"] for hit in hits]

    response = client.rerank(
        query=query,
        documents=documents,
        model="rerank-v3.5",
        top_n=top_k,
    )

    reranked_hits = []
    for result in response.results:
        hit = hits[result.index].copy()
        hit["_score"] = result.relevance_score
        reranked_hits.append(hit)

    return reranked_hits


def print_results(hits: list[dict], label: str):
    """검색 결과 출력"""
    print(f"\n{'='*60}")
    print(f"  {label} (총 {len(hits)}건)")
    print(f"{'='*60}")
    for i, hit in enumerate(hits, 1):
        source = hit["_source"]
        score = hit.get("_score", 0)
        meta = source.get("metadata", {})
        content = source["content"][:150].replace("\n", " ")
        print(f"\n  [{i}] score={score:.4f}")
        print(f"      source: {meta.get('source', '?')} (p{meta.get('page', '?')})")
        print(f"      {content}...")


def main():
    parser = argparse.ArgumentParser(description="Elasticsearch 검색 테스트")
    parser.add_argument("query", help="검색 쿼리")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid", "all"], default="all", help="검색 모드")
    parser.add_argument("--top-k", type=int, default=5, help="결과 수 (기본: 5)")
    parser.add_argument("--rerank", action="store_true", help="Cohere ReRanker 적용")
    args = parser.parse_args()

    es = get_es_client()
    print(f"쿼리: \"{args.query}\"")

    modes = ["bm25", "vector", "hybrid"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "bm25":
            hits = bm25_search(es, args.query, args.top_k)
            label = "BM25 검색"
        elif mode == "vector":
            hits = vector_search(es, args.query, args.top_k)
            label = "Vector 검색"
        else:
            hits = hybrid_search(es, args.query, args.top_k)
            label = "Hybrid 검색"

        print_results(hits, label)

        if args.rerank and hits:
            reranked = rerank(args.query, hits, args.top_k)
            print_results(reranked, f"{label} + Cohere ReRank")


if __name__ == "__main__":
    main()
