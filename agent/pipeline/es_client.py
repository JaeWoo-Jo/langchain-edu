from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.documents import Document
from config import ES_INDEX


def index_documents(
    es: Elasticsearch,
    chunks: list[Document],
    embeddings: list[list[float]],
) -> int:
    """청크와 임베딩을 Elasticsearch에 bulk 적재한다."""

    def _generate_actions():
        for chunk, vector in zip(chunks, embeddings):
            yield {
                "_index": ES_INDEX,
                "_source": {
                    "content": chunk.page_content,
                    "content_vector": vector,
                    "metadata": {
                        "source": chunk.metadata.get("source", ""),
                        "source_type": chunk.metadata.get("source_type", ""),
                        "category": chunk.metadata.get("category", ""),
                        "page": chunk.metadata.get("page", 0),
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                    },
                },
            }

    success, errors = bulk(es, _generate_actions(), raise_on_error=False)
    if errors:
        print(f"  적재 중 오류 {len(errors)}건 발생")
        for err in errors[:5]:
            print(f"    {err}")
    print(f"  ES 적재 완료: {success}건 성공")
    return success
