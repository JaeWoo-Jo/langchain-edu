from elasticsearch import Elasticsearch
from config import ES_URL, ES_USER, ES_PASSWORD, ES_INDEX, EMBEDDING_DIMENSION

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "standard",
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMENSION,
                "index": True,
                "similarity": "cosine",
            },
            "metadata": {
                "properties": {
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "source_type": {"type": "keyword"},
                    "category": {"type": "keyword"},
                }
            },
        }
    },
}


def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        ES_URL,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False,
    )


def create_index(es: Elasticsearch, delete_existing: bool = False) -> None:
    if es.indices.exists(index=ES_INDEX):
        if delete_existing:
            es.indices.delete(index=ES_INDEX)
            print(f"기존 인덱스 '{ES_INDEX}' 삭제 완료")
        else:
            print(f"인덱스 '{ES_INDEX}'가 이미 존재합니다.")
            return

    es.indices.create(index=ES_INDEX, body=INDEX_MAPPING)
    print(f"인덱스 '{ES_INDEX}' 생성 완료")


if __name__ == "__main__":
    es = get_es_client()
    create_index(es)
