import time

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

_client = OpenAI(api_key=OPENAI_API_KEY)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트를 임베딩 벡터로 변환한다. 배치 처리 + 재시도 로직."""
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = _client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                print(f"  임베딩 생성: {len(all_embeddings)}/{len(texts)}")
                break
            except RateLimitError:
                wait = RETRY_DELAY * attempt
                print(f"  Rate limit 초과, {wait}초 후 재시도 ({attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            except (APITimeoutError, APIError) as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"임베딩 생성 실패 (재시도 {MAX_RETRIES}회 초과): {e}")
                wait = RETRY_DELAY * attempt
                print(f"  API 오류, {wait}초 후 재시도 ({attempt}/{MAX_RETRIES}): {e}")
                time.sleep(wait)

    return all_embeddings
