from elasticsearch import Elasticsearch
from app.core.config import settings

_client: Elasticsearch | None = None


def get_elasticsearch_client() -> Elasticsearch:
    """Elasticsearch 클라이언트 싱글턴을 반환합니다."""
    global _client
    if _client is None:
        _client = Elasticsearch(
            hosts=[settings.ELASTICSEARCH_HOST],
            basic_auth=(settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD),
            verify_certs=settings.ELASTICSEARCH_VERIFY_CERTS,
        )
    return _client
