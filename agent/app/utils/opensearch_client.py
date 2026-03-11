from opensearchpy import OpenSearch
from app.core.config import settings


def get_opensearch_client() -> OpenSearch:
    """OpenSearch 클라이언트를 생성하여 반환합니다."""
    client = OpenSearch(
        hosts=[settings.OPENSEARCH_HOST],
        http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
        verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False,
    )
    return client
