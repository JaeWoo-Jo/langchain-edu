"""가격 도구들이 공유하는 Elasticsearch 클라이언트 및 유틸리티.

모든 외부 클라이언트는 싱글턴 패턴으로 관리한다.
"""

from __future__ import annotations

from elasticsearch import Elasticsearch

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

CONTENT_FIELD = "item_name"
TOP_K = 10
RAG_CONTENT_FIELD = "content"

# ---------------------------------------------------------------------------
# 싱글턴 인스턴스
# ---------------------------------------------------------------------------

_es_client: Elasticsearch | None = None


def get_es_client() -> Elasticsearch:
    """가격 데이터 인덱스용 ES 클라이언트 (싱글턴)."""
    global _es_client
    if _es_client is None:
        from app.core.config import settings

        _es_client = Elasticsearch(
            hosts=[settings.ELASTICSEARCH_HOST],
            basic_auth=(settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD),
            verify_certs=settings.ELASTICSEARCH_VERIFY_CERTS,
        )
    return _es_client


def get_price_index() -> str:
    """가격 인덱스명을 반환한다."""
    from app.core.config import settings

    return settings.ELASTICSEARCH_INDEX_PRICES


def get_rag_index() -> str:
    """RAG 문서 인덱스명을 반환한다."""
    from app.core.config import settings

    return settings.ELASTICSEARCH_INDEX_RAG


def format_price_hits(hits: list[dict]) -> str:
    """Elasticsearch 가격 검색 결과를 읽기 좋은 문자열로 포맷팅한다."""
    if not hits:
        return ""
    results: list[str] = []
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        score = hit.get("_score", 0)
        today = src.get("price_today", 0)
        one_week = src.get("price_1week_ago", 0)
        one_month = src.get("price_1month_ago", 0)

        diff_week = today - one_week if one_week else 0
        diff_month = today - one_month if one_month else 0

        header = f"[{i}] score={score:.4f}"
        results.append(
            f"{header}\n"
            f"- {src.get('item_name', '')} ({src.get('kind_name', '')}): "
            f"오늘 {today:,}원/{src.get('unit', '')} "
            f"(1주전 대비 {diff_week:+,}원, 1개월전 대비 {diff_month:+,}원)"
        )
    return "\n\n".join(results)
