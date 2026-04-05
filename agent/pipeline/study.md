# RAG 파이프라인 학습 가이드

> 이 프로젝트는 PDF/HWP 문서를 Elasticsearch에 적재하고, 다양한 검색 방식으로 질의하는 **RAG(Retrieval-Augmented Generation) 파이프라인**을 직접 구축하는 실습입니다.

---

## 목차

1. [RAG란 무엇인가?](#1-rag란-무엇인가)
2. [프로젝트 전체 구조](#2-프로젝트-전체-구조)
3. [환경 설정](#3-환경-설정)
4. [Step 1: 데이터 수집](#step-1-데이터-수집)
5. [Step 2: 문서 파싱 (Document Loading)](#step-2-문서-파싱-document-loading)
6. [Step 3: 텍스트 청킹 (Chunking)](#step-3-텍스트-청킹-chunking)
7. [Step 4: 임베딩 생성 (Embedding)](#step-4-임베딩-생성-embedding)
8. [Step 5: Elasticsearch 인덱싱](#step-5-elasticsearch-인덱싱)
9. [Step 6: 검색 (Retrieval)](#step-6-검색-retrieval)
10. [Step 7: Reranking](#step-7-reranking)
11. [실습 과제](#실습-과제)
12. [더 알아보기](#더-알아보기)

**별첨: 구성요소별 상세 가이드**

- [별첨 A. Elasticsearch](#별첨-a-elasticsearch)
- [별첨 B. OpenAI Embeddings API](#별첨-b-openai-embeddings-api)
- [별첨 C. LangChain](#별첨-c-langchain)
- [별첨 D. Cohere Rerank API](#별첨-d-cohere-rerank-api)
- [별첨 E. OpenAI Python SDK](#별첨-e-openai-python-sdk)
- [별첨 F. uv (패키지 매니저)](#별첨-f-uv-패키지-매니저)
- [별첨 G. BeautifulSoup (웹 크롤링)](#별첨-g-beautifulsoup-웹-크롤링)
- [별첨 H. python-dotenv (환경변수 관리)](#별첨-h-python-dotenv-환경변수-관리)

---

## 1. RAG란 무엇인가?

**RAG (Retrieval-Augmented Generation)** 은 LLM이 답변을 생성하기 전에, 외부 지식 저장소에서 관련 문서를 **검색(Retrieve)** 하여 프롬프트에 함께 제공하는 기법입니다.

### 왜 RAG가 필요한가?

| 문제 | RAG의 해결 |
|------|-----------|
| LLM은 학습 데이터 이후의 정보를 모른다 | 최신 문서를 검색하여 제공 |
| LLM이 없는 내용을 지어내는 환각(Hallucination) | 실제 문서 근거를 기반으로 답변 |
| 사내 문서, 비공개 데이터에 대한 답변 불가 | 자체 문서를 인덱싱하여 검색 가능 |

### RAG의 전체 흐름

```
[사용자 질문]
      │
      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Retriever  │────▶│  관련 문서 검색   │────▶│  LLM 생성   │
│ (검색 엔진)  │     │  (상위 K개 반환)  │     │ (답변 생성)  │
└─────────────┘     └──────────────────┘     └─────────────┘
                                                    │
                                                    ▼
                                              [최종 답변]
```

**이 프로젝트에서는 RAG의 핵심인 "Retriever" 부분을 구축합니다.**

---

## 2. 프로젝트 전체 구조

```
pipeline/
├── config.py           # 환경변수 로드 (ES, OpenAI, 청킹 파라미터)
├── download_pdfs.py    # Step 1: 데이터 수집 (웹 크롤링)
├── pdf_loader.py       # Step 2: PDF 파싱
├── hwp_loader.py       # Step 2: HWP 파싱
├── chunker.py          # Step 3: 텍스트 청킹
├── embedder.py         # Step 4: 임베딩 생성
├── index_mapping.py    # Step 5: ES 인덱스 매핑 정의 & 생성
├── es_client.py        # Step 5: ES에 문서 적재
├── main.py             # 파이프라인 통합 실행 (Step 2~5)
├── search.py           # Step 6-7: 검색 & Reranking
├── .env.example        # 환경변수 템플릿
├── pyproject.toml      # 의존성 관리 (uv)
└── data/               # 다운로드된 PDF/HWP 파일
```

### 데이터 흐름

```
PDF/HWP 파일
    │
    ▼
[문서 파싱] ─── pdf_loader.py / hwp_loader.py
    │             → Document(page_content, metadata)
    ▼
[텍스트 청킹] ── chunker.py
    │             → 500자 단위 분할 (100자 오버랩)
    ▼
[임베딩 생성] ── embedder.py
    │             → OpenAI text-embedding-3-small (1536차원)
    ▼
[ES 적재] ────── es_client.py + index_mapping.py
    │             → content + content_vector + metadata
    ▼
[검색] ───────── search.py
                  → BM25 / Vector / Hybrid / +Rerank
```

---

## 3. 환경 설정

### 3-1. Python 환경 구성

이 프로젝트는 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
# uv가 없다면 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

### 3-2. 환경변수 설정

`.env.example`을 복사하여 `.env`를 만들고, 실제 값을 입력합니다.

```bash
cp .env.example .env
```

```dotenv
# Elasticsearch
ES_URL=https://elasticsearch-edu.didim365.app
ES_USER=elastic
ES_PASSWORD=<비밀번호>
ES_INDEX=edu-medicine-info

# OpenAI (임베딩 생성에 사용)
OPENAI_API_KEY=<OpenAI API 키>

# Cohere (ReRanker에 사용, 선택)
COHERE_API_KEY=<Cohere API 키>
```

> **학습 포인트**: `config.py`를 열어 환경변수가 어떻게 로드되는지 확인해 보세요. `python-dotenv`는 `.env` 파일을 읽어 `os.getenv()`로 접근할 수 있게 합니다.

---

## Step 1: 데이터 수집

> **파일**: `download_pdfs.py`

RAG의 첫 단계는 검색 대상이 될 **원본 문서를 확보**하는 것입니다.

### 이 스크립트가 하는 일

1. 식약처(nedrug.mfds.go.kr)의 게시판 페이지를 크롤링
2. 각 게시물의 첨부파일(PDF, HWP) 정보를 추출
3. 파일을 `data/` 디렉토리에 다운로드

### 실행 방법

```bash
uv run python download_pdfs.py
```

### 핵심 개념

| 개념 | 설명 |
|------|------|
| **웹 크롤링** | `requests` + `BeautifulSoup`으로 HTML을 파싱하여 데이터 추출 |
| **게시판 페이지네이션** | 50건 단위로 여러 페이지를 순회하며 게시물 ID 수집 |
| **서버 부하 방지** | `time.sleep(0.3)`으로 요청 간 간격을 두어 서버에 무리를 주지 않음 |

### 생각해 볼 점

- 크롤링 시 `time.sleep()`을 넣는 이유는 무엇인가?
- 이미 다운로드된 파일을 다시 받지 않도록 어떻게 처리하고 있는가? (`download_file` 함수 참고)
- 실제 업무에서는 크롤링 대신 어떤 방식으로 문서를 수집할 수 있을까? (API, S3, DB 등)

---

## Step 2: 문서 파싱 (Document Loading)

> **파일**: `pdf_loader.py`, `hwp_loader.py`

수집한 파일에서 **텍스트를 추출**하는 단계입니다.

### PDF 파싱 (`pdf_loader.py`)

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(str(file_path))
documents = loader.load()  # 페이지별 Document 리스트 반환
```

- `PyPDFLoader`는 PDF를 **페이지 단위**로 분리하여 `Document` 객체를 생성합니다.
- 각 `Document`에는 `page_content`(텍스트)와 `metadata`(source, page 번호)가 포함됩니다.

### HWP 파싱 (`hwp_loader.py`)

```python
result = subprocess.run(["hwp5txt", str(file_path)], capture_output=True, text=True)
```

- HWP는 한국 고유 포맷이라 별도 도구(`pyhwp`)가 필요합니다.
- `hwp5txt` CLI로 텍스트를 추출하여 하나의 `Document`로 만듭니다.

### 핵심 개념: LangChain Document

```python
from langchain_core.documents import Document

doc = Document(
    page_content="추출된 텍스트 내용...",
    metadata={"source": "파일경로", "page": 0}
)
```

`Document`는 LangChain에서 텍스트와 메타데이터를 함께 관리하는 기본 단위입니다. 이후 청킹, 임베딩, 검색에서 모두 이 구조를 사용합니다.

### 생각해 볼 점

- PDF에서 추출한 텍스트가 원본과 다를 수 있는 경우는? (이미지 기반 PDF, 표 등)
- `metadata`에 source와 page를 저장하는 이유는 무엇인가?
- 다른 포맷(DOCX, HTML, 이미지 등)은 어떻게 처리할 수 있을까?

---

## Step 3: 텍스트 청킹 (Chunking)

> **파일**: `chunker.py`

문서 전체를 한 번에 임베딩하면 의미가 희석됩니다. **적절한 크기로 분할**하여 검색 정밀도를 높입니다.

### 코드 분석

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 한 청크의 최대 문자 수
    chunk_overlap=100,    # 청크 간 겹치는 문자 수
    length_function=len,
)
chunks = splitter.split_documents(documents)
```

### RecursiveCharacterTextSplitter의 동작 원리

이 splitter는 다음 구분자를 **순서대로** 시도하며 텍스트를 분할합니다:

```
"\n\n" → "\n" → " " → ""
```

1. 먼저 `\n\n`(빈 줄)로 나눠봄
2. 그래도 chunk_size를 초과하면 `\n`(줄바꿈)으로 나눔
3. 그래도 초과하면 공백으로 나눔
4. 최후에는 글자 단위로 나눔

### 왜 overlap이 필요한가?

```
원문: "고혈압 환자는 나트륨 섭취를 줄여야 합니다. 칼륨이 풍부한 음식을 권장합니다."

chunk_size=30, overlap=0일 때:
  청크1: "고혈압 환자는 나트륨 섭취를 줄여야 합니다."
  청크2: "칼륨이 풍부한 음식을 권장합니다."

chunk_size=30, overlap=10일 때:
  청크1: "고혈압 환자는 나트륨 섭취를 줄여야 합니다."
  청크2: "줄여야 합니다. 칼륨이 풍부한 음식을 권장합니다."
                ↑ 겹침 구간: 문맥 연결 유지
```

overlap이 있으면 청크 경계에서 **문맥이 끊기는 것을 방지**합니다.

### 생각해 볼 점

- `chunk_size`를 너무 크게 하면 어떤 문제가 생길까? 너무 작게 하면?
- `chunk_overlap`을 0으로 하면 어떤 정보가 손실될 수 있는가?
- 500자와 100 overlap은 어떤 기준으로 정해졌을까? 다른 값으로 실험해 보자.

### 실험해 보기

```bash
# 다양한 청크 크기로 파이프라인 실행
uv run python main.py --chunk-size 200 --chunk-overlap 50 --recreate-index
uv run python main.py --chunk-size 1000 --chunk-overlap 200 --recreate-index

# 이후 같은 쿼리로 검색하여 결과를 비교
uv run python search.py "고혈압 약"
```

---

## Step 4: 임베딩 생성 (Embedding)

> **파일**: `embedder.py`

텍스트를 **숫자 벡터(임베딩)** 로 변환하여 의미 기반 검색을 가능하게 합니다.

### 임베딩이란?

```
"고혈압 치료약"  →  [0.023, -0.118, 0.045, ..., 0.087]  (1536차원 벡터)
"혈압 낮추는 약" →  [0.021, -0.115, 0.042, ..., 0.089]  (유사한 벡터!)
"오늘 날씨"     →  [-0.156, 0.098, -0.203, ..., 0.012]  (다른 벡터)
```

- **의미가 비슷한 텍스트**는 벡터 공간에서 **가까운 위치**에 놓입니다.
- 이를 활용하면 정확한 키워드가 일치하지 않아도 의미 기반으로 검색할 수 있습니다.

### 코드 분석

```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
response = client.embeddings.create(
    input=batch,                      # 텍스트 리스트
    model="text-embedding-3-small",   # 임베딩 모델
)
embeddings = [item.embedding for item in response.data]
```

### 이 프로젝트의 설정

| 항목 | 값 | 설명 |
|------|-----|------|
| 모델 | `text-embedding-3-small` | OpenAI의 경량 임베딩 모델 |
| 차원 | 1536 | 벡터의 크기 (숫자 1536개로 표현) |
| 배치 크기 | 100 | API 호출 당 처리하는 텍스트 수 |
| 재시도 | 최대 3회 | Rate Limit, 타임아웃 시 자동 재시도 |

### 코사인 유사도

두 벡터가 얼마나 비슷한지 측정하는 방법입니다:

```
유사도 = cos(θ) = (A · B) / (|A| × |B|)

1.0  = 완전히 같은 의미
0.0  = 관련 없음
-1.0 = 반대 의미
```

이 프로젝트에서 Elasticsearch의 `similarity: "cosine"` 설정이 이를 사용합니다.

### 생각해 볼 점

- 왜 배치 단위로 처리하는가? 1건씩 보내면 안 되는가?
- Rate Limit 재시도 로직에서 `wait = RETRY_DELAY * attempt`인 이유는? (지수 백오프 vs 선형 백오프)
- `text-embedding-3-small`과 `text-embedding-3-large`의 차이는 무엇인가?

---

## Step 5: Elasticsearch 인덱싱

> **파일**: `index_mapping.py`, `es_client.py`

임베딩된 문서를 Elasticsearch에 저장합니다.

### 인덱스 매핑 (`index_mapping.py`)

```python
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "content": {                     # 원문 텍스트
                "type": "text",
                "analyzer": "standard",
            },
            "content_vector": {              # 임베딩 벡터
                "type": "dense_vector",
                "dims": 1536,
                "index": True,               # kNN 검색 인덱스 활성화
                "similarity": "cosine",
            },
            "metadata": {                    # 부가 정보
                "properties": {
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                }
            },
        }
    },
}
```

### 각 필드의 역할

| 필드 | 타입 | 용도 |
|------|------|------|
| `content` | `text` | BM25 키워드 검색 대상 |
| `content_vector` | `dense_vector` | Vector(kNN) 검색 대상 |
| `metadata.source` | `keyword` | 원본 파일 경로 (필터링용) |
| `metadata.page` | `integer` | 원본 페이지 번호 |
| `metadata.chunk_index` | `integer` | 청크 순서 |

### Bulk 적재 (`es_client.py`)

```python
from elasticsearch.helpers import bulk

def _generate_actions():
    for chunk, vector in zip(chunks, embeddings):
        yield {
            "_index": ES_INDEX,
            "_source": {
                "content": chunk.page_content,
                "content_vector": vector,
                "metadata": { ... },
            },
        }

success, errors = bulk(es, _generate_actions())
```

- `bulk` API는 여러 문서를 **한 번의 요청**으로 적재합니다.
- 제너레이터(`yield`)를 사용하여 메모리 효율적으로 처리합니다.

### 생각해 볼 점

- `dense_vector`에 `"index": True`를 설정하는 이유는?
- `content`에 `"analyzer": "standard"`를 사용하면 한국어 검색에 어떤 한계가 있을까?
- `keyword` 타입과 `text` 타입의 차이는 무엇인가?

---

## Step 6: 검색 (Retrieval)

> **파일**: `search.py`

이 프로젝트는 3가지 검색 방식을 지원합니다.

### 6-1. BM25 검색 (키워드 기반)

```python
body = {
    "query": {"match": {"content": query}},
    "size": top_k,
}
```

- 전통적인 **키워드 매칭** 방식
- 질의에 포함된 단어가 문서에 얼마나 자주 등장하는지를 기반으로 점수를 매김
- 장점: 정확한 용어 검색에 강함 (예: "아세트아미노펜")
- 단점: 동의어, 유사 표현 검색에 약함

### 6-2. Vector 검색 (의미 기반)

```python
body = {
    "knn": {
        "field": "content_vector",
        "query_vector": query_vector,   # 질의를 임베딩
        "k": top_k,
        "num_candidates": top_k * 10,
    },
    "size": top_k,
}
```

- 질의를 임베딩하여 **벡터 공간에서 가까운 문서**를 찾음
- 장점: "해열제" 검색 시 "열 내리는 약" 포함 문서도 찾음
- 단점: 정확한 고유명사 검색에는 BM25보다 약할 수 있음

### 6-3. Hybrid 검색 (BM25 + Vector)

```python
body = {
    "query": {"match": {"content": query}},    # BM25
    "knn": {                                     # Vector
        "field": "content_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": top_k * 10,
    },
    "size": top_k,
}
```

- BM25와 Vector 검색을 **동시에** 수행
- Elasticsearch가 내부적으로 두 점수를 합산하여 최종 랭킹 결정
- **일반적으로 가장 좋은 성능**을 보여줌

### 검색 방식 비교

| 방식 | 원리 | 장점 | 단점 |
|------|------|------|------|
| BM25 | 키워드 빈도 기반 | 정확한 용어 매칭 | 동의어 검색 약함 |
| Vector | 의미적 유사도 | 유사 표현 검색 가능 | 고유명사 약함 |
| Hybrid | 두 방식 결합 | 균형 잡힌 성능 | 점수 스케일 차이 문제 |

### 실행 방법

```bash
# 전체 검색 모드 비교
uv run python search.py "고혈압 약"

# 특정 모드만 실행
uv run python search.py "안약 사용법" --mode bm25
uv run python search.py "안약 사용법" --mode vector
uv run python search.py "안약 사용법" --mode hybrid
```

### 실험해 보기

같은 쿼리로 3가지 모드의 결과를 비교하고, 다음을 관찰해 보세요:

1. **정확한 의약품 이름**으로 검색: BM25와 Vector 중 어느 쪽이 더 정확한가?
2. **일상적인 표현**으로 검색 (예: "열 내리는 약"): Vector가 더 나은 결과를 보여주는가?
3. **Hybrid**는 어떤 경우에 가장 좋은 결과를 내는가?

---

## Step 7: Reranking

> **파일**: `search.py`의 `rerank()` 함수

### Reranking이란?

1차 검색(Retrieval)에서 가져온 결과를 **더 정교한 모델로 재정렬**하는 기법입니다.

```
[질의] ──▶ [1차 검색: 후보 K개] ──▶ [ReRanker: 관련도 재평가] ──▶ [최종 결과]
             (빠르지만 대략적)        (느리지만 정밀)
```

### 왜 ReRanker가 필요한가?

- 1차 검색(BM25, Vector)은 **빠르지만 대략적**인 관련도 점수를 사용
- ReRanker는 질의와 문서를 **함께 분석**하여 더 정밀한 관련도를 판단
- Cross-Encoder 방식: 질의-문서 쌍을 동시에 입력하여 관련도 점수 산출

### 코드 분석

```python
import cohere

client = cohere.Client(api_key=COHERE_API_KEY)
response = client.rerank(
    query=query,
    documents=documents,       # 1차 검색 결과의 텍스트 리스트
    model="rerank-v3.5",
    top_n=top_k,               # 상위 N개만 반환
)
```

### 실행 방법

```bash
# Hybrid 검색 + ReRank
uv run python search.py "고혈압 약 부작용" --mode hybrid --rerank
```

### 실험해 보기

```bash
# ReRank 없이
uv run python search.py "변비약 복용법" --mode hybrid

# ReRank 적용
uv run python search.py "변비약 복용법" --mode hybrid --rerank
```

두 결과의 **순서**가 어떻게 달라지는지 비교해 보세요. 특히 상위 1~2건의 관련도가 개선되는지 확인합니다.

---

## 실습 과제

### 과제 1: 파이프라인 직접 실행하기

1. `.env` 파일을 설정한다.
2. `uv run python download_pdfs.py`로 데이터를 다운로드한다.
3. `uv run python main.py`로 전체 파이프라인을 실행한다.
4. `uv run python search.py "고혈압" --mode all`로 검색 결과를 확인한다.

### 과제 2: 청크 파라미터 실험

서로 다른 `chunk_size`와 `chunk_overlap` 설정으로 파이프라인을 돌리고, 검색 품질이 어떻게 달라지는지 비교해 보세요.

| 실험 | chunk_size | chunk_overlap | 관찰할 점 |
|------|-----------|---------------|-----------|
| A | 200 | 50 | 청크 수 증가, 세밀한 검색 |
| B | 500 | 100 | 기본값 |
| C | 1000 | 200 | 청크 수 감소, 넓은 문맥 |

```bash
uv run python main.py --chunk-size 200 --chunk-overlap 50 --recreate-index
uv run python search.py "해열제 사용법"
```

### 과제 3: 검색 방식 비교 리포트

5개의 다양한 쿼리를 선정하여 BM25 / Vector / Hybrid / Hybrid+Rerank 4가지 방식의 검색 결과를 비교하는 리포트를 작성해 보세요.

**리포트 포함 내용:**
- 각 쿼리별 상위 3건의 결과 (점수, 내용 요약)
- 어떤 검색 방식이 어떤 유형의 쿼리에 강한지 분석
- 자신의 결론 및 추천 방식

### 과제 4 (심화): 새로운 데이터 소스 추가

`download_pdfs.py`를 참고하여, 다른 웹사이트에서 문서를 수집하는 스크립트를 작성하고 파이프라인에 적재해 보세요.

### 과제 5 (심화): 한국어 분석기 적용

현재 `content` 필드는 `standard` 분석기를 사용합니다. Elasticsearch의 한국어 분석기(nori)를 적용하면 BM25 검색 성능이 어떻게 변하는지 실험해 보세요.

```python
# index_mapping.py에서 분석기를 변경
"content": {
    "type": "text",
    "analyzer": "nori",
}
```

---

## 더 알아보기

### 핵심 용어 정리

| 용어 | 설명 |
|------|------|
| **RAG** | Retrieval-Augmented Generation. 검색으로 보강된 생성 |
| **Embedding** | 텍스트를 고차원 벡터로 변환한 것 |
| **Chunking** | 긴 문서를 검색에 적합한 크기로 분할하는 것 |
| **BM25** | 키워드 빈도 기반의 전통적 검색 알고리즘 |
| **kNN** | k-Nearest Neighbors. 벡터 공간에서 가장 가까운 k개를 찾는 검색 |
| **Hybrid Search** | 키워드 + 벡터 검색을 결합한 방식 |
| **ReRanker** | 1차 검색 결과를 더 정밀하게 재정렬하는 모델 |
| **Dense Vector** | 임베딩 벡터를 저장하는 Elasticsearch 필드 타입 |
| **Bulk API** | 여러 문서를 한 번에 적재하는 Elasticsearch API |

### 참고 자료

- [LangChain Text Splitters 문서](https://python.langchain.com/docs/concepts/text_splitters/)
- [OpenAI Embeddings 가이드](https://platform.openai.com/docs/guides/embeddings)
- [Elasticsearch Dense Vector 검색](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
- [Cohere Rerank 문서](https://docs.cohere.com/docs/rerank)
- [RAG 개념 설명 (LangChain)](https://python.langchain.com/docs/concepts/rag/)

---
---

# 별첨: 구성요소별 상세 가이드

---

## 별첨 A. Elasticsearch

> 이 프로젝트에서 **문서 저장소 겸 검색 엔진** 역할을 합니다.

### A-1. Elasticsearch란?

Elasticsearch는 Apache Lucene 기반의 **분산 검색/분석 엔진**입니다. 원래 텍스트 검색(BM25)에 강점이 있었으나, 최근 버전에서는 `dense_vector` 타입을 지원하여 **벡터 검색(kNN)** 도 가능합니다.

```
┌─────────────────────────────────────────┐
│              Elasticsearch              │
│                                         │
│  ┌─────────┐  ┌─────────┐              │
│  │ Index A │  │ Index B │  ...          │
│  │(문서들)  │  │(문서들)  │              │
│  └─────────┘  └─────────┘              │
│                                         │
│  검색 방식:                              │
│  • BM25 (텍스트 매칭)                    │
│  • kNN  (벡터 유사도)                    │
│  • Hybrid (두 가지 결합)                 │
└─────────────────────────────────────────┘
```

### A-2. 핵심 개념

| 개념 | RDBMS 대응 | 설명 |
|------|-----------|------|
| **Index** | Database/Table | 문서들의 논리적 모음 |
| **Document** | Row | JSON 형태의 개별 데이터 단위 |
| **Field** | Column | 문서 내 개별 속성 |
| **Mapping** | Schema | 필드의 타입과 분석 방식 정의 |
| **Analyzer** | - | 텍스트를 토큰으로 분리하는 방식 |
| **Shard** | Partition | 인덱스를 분산 저장하는 단위 |

### A-3. 이 프로젝트에서의 사용법

#### 클라이언트 연결

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(
    "https://elasticsearch-edu.didim365.app",
    basic_auth=("elastic", "비밀번호"),
    verify_certs=False,           # self-signed 인증서 무시
)

# 연결 확인
print(es.info()["version"]["number"])
```

#### 인덱스 생성

```python
# 매핑을 정의하여 인덱스 생성
es.indices.create(index="my-index", body={
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
})
```

#### 문서 적재 (단건 / Bulk)

```python
# 단건 적재
es.index(index="my-index", body={"content": "안녕하세요", "content_vector": [0.1, ...]})

# Bulk 적재 (대량 데이터에 권장)
from elasticsearch.helpers import bulk

actions = [
    {"_index": "my-index", "_source": {"content": "문서1", "content_vector": [...]}}
    {"_index": "my-index", "_source": {"content": "문서2", "content_vector": [...]}}
]
bulk(es, actions)
```

#### 검색

```python
# BM25 검색
es.search(index="my-index", body={"query": {"match": {"content": "검색어"}}})

# Vector 검색 (kNN)
es.search(index="my-index", body={
    "knn": {
        "field": "content_vector",
        "query_vector": [0.1, 0.2, ...],
        "k": 5,
        "num_candidates": 50,
    }
})

# 인덱스 정보 확인
es.indices.get(index="my-index")
es.count(index="my-index")
```

### A-4. 필드 타입 이해

| 타입 | 용도 | 검색 방식 |
|------|------|-----------|
| `text` | 전문 검색용 텍스트 (분석기가 토큰으로 분리) | `match`, `multi_match` |
| `keyword` | 정확한 값 매칭 (분석기 적용 안 됨) | `term`, `terms`, 필터링/집계 |
| `dense_vector` | 임베딩 벡터 저장 | `knn` |
| `integer` / `float` | 숫자 | `range`, 정렬, 집계 |

### A-5. Analyzer 이해

Analyzer는 텍스트를 **토큰(검색 가능한 단위)** 으로 변환하는 과정입니다.

```
입력: "고혈압 환자의 치료 가이드"

[standard analyzer]  → ["고혈압", "환자의", "치료", "가이드"]
[nori analyzer]      → ["고혈압", "환자", "의", "치료", "가이드"]
                          ↑ 한국어 형태소 분석으로 조사 분리
```

`standard` 분석기는 공백/구두점 기반으로 분리하므로 "환자의"와 "환자"를 다른 토큰으로 인식합니다. 한국어에서는 `nori` 분석기가 더 적합합니다.

### A-6. 심화 학습 링크

| 주제 | 링크 |
|------|------|
| Elasticsearch 공식 가이드 (시작하기) | https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started.html |
| Python 클라이언트 문서 | https://elasticsearch-py.readthedocs.io/en/stable/ |
| Dense Vector 필드 타입 | https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html |
| kNN 검색 가이드 | https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html |
| Hybrid 검색 (BM25 + kNN) | https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#_combine_approximate_knn_with_other_features |
| Nori 한국어 분석기 | https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-nori.html |
| Mapping 정의 가이드 | https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html |
| Bulk API 가이드 | https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html |
| BM25 알고리즘 설명 | https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables |
| Elasticsearch Labs (AI Search) | https://www.elastic.co/search-labs |

---

## 별첨 B. OpenAI Embeddings API

> 이 프로젝트에서 **텍스트를 벡터로 변환**하는 데 사용합니다.

### B-1. Embeddings API 개요

OpenAI Embeddings API는 텍스트를 **고정 크기의 실수 벡터**로 변환합니다. 의미가 유사한 텍스트는 벡터 공간에서 가까이 위치합니다.

```
"감기 치료법"       → [0.023, -0.118, 0.045, ..., 0.087]  ─┐
"감기에 걸렸을 때"   → [0.021, -0.115, 0.042, ..., 0.089]  ─┤ 가까움
                                                           │
"오늘 주가 동향"     → [-0.156, 0.098, -0.203, ..., 0.012] ─┘ 멀리 떨어짐
```

### B-2. 사용 가능한 모델

| 모델 | 차원 | 가격 (1M 토큰) | 특징 |
|------|------|---------------|------|
| `text-embedding-3-small` | 1536 | $0.02 | 경량, 비용 효율적 (이 프로젝트에서 사용) |
| `text-embedding-3-large` | 3072 | $0.13 | 더 높은 정밀도 |
| `text-embedding-ada-002` | 1536 | $0.10 | 이전 세대 (호환성 목적) |

### B-3. 기본 사용법

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")

# 단일 텍스트 임베딩
response = client.embeddings.create(
    input="고혈압 약의 부작용",
    model="text-embedding-3-small",
)
vector = response.data[0].embedding  # [0.023, -0.118, ...]
print(f"벡터 차원: {len(vector)}")   # 1536

# 배치 임베딩 (여러 텍스트를 한 번에)
response = client.embeddings.create(
    input=["텍스트1", "텍스트2", "텍스트3"],
    model="text-embedding-3-small",
)
vectors = [item.embedding for item in response.data]
```

### B-4. 유사도 계산 예시

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 두 텍스트의 유사도 계산
response = client.embeddings.create(
    input=["감기 치료법", "감기에 걸렸을 때 대처법"],
    model="text-embedding-3-small",
)
sim = cosine_similarity(response.data[0].embedding, response.data[1].embedding)
print(f"유사도: {sim:.4f}")  # 약 0.85~0.95 (높은 유사도)
```

### B-5. API 사용 시 주의사항

| 항목 | 내용 |
|------|------|
| **토큰 제한** | `text-embedding-3-small`은 입력 최대 8,191 토큰 |
| **Rate Limit** | 분당 요청 수 / 토큰 수 제한 있음 (Tier별 상이) |
| **비용** | 토큰 기반 과금, 대량 처리 시 배치가 효율적 |
| **재현성** | 같은 입력에 대해 항상 같은 벡터 반환 (결정적) |

### B-6. 심화 학습 링크

| 주제 | 링크 |
|------|------|
| OpenAI Embeddings 공식 가이드 | https://platform.openai.com/docs/guides/embeddings |
| API 레퍼런스 | https://platform.openai.com/docs/api-reference/embeddings |
| 모델 비교 및 벤치마크 | https://platform.openai.com/docs/guides/embeddings#embedding-models |
| 토큰 카운팅 (tiktoken) | https://github.com/openai/tiktoken |
| 임베딩 활용 Cookbook | https://cookbook.openai.com/examples/get_embeddings_from_dataset |
| OpenAI Python SDK | https://github.com/openai/openai-python |
| Rate Limits 가이드 | https://platform.openai.com/docs/guides/rate-limits |

---

## 별첨 C. LangChain

> 이 프로젝트에서 **문서 로딩, 데이터 모델(Document), 텍스트 분할**에 사용합니다.

### C-1. LangChain이란?

LangChain은 LLM 기반 애플리케이션 개발을 위한 프레임워크입니다. 이 프로젝트에서는 LangChain의 전체 체인이 아닌, **유틸리티 모듈**만 사용합니다.

### C-2. 이 프로젝트에서 사용하는 LangChain 구성요소

```
langchain-core          → Document 클래스
langchain-text-splitters → RecursiveCharacterTextSplitter
langchain-community     → PyPDFLoader
```

### C-3. Document 클래스

LangChain의 데이터 전달 기본 단위입니다.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="문서의 텍스트 내용",
    metadata={
        "source": "data/example.pdf",
        "page": 0,
        "chunk_index": 3,
    },
)

# 접근
print(doc.page_content)        # 텍스트
print(doc.metadata["source"])  # 메타데이터
```

### C-4. Document Loaders (문서 로더)

다양한 형식의 파일을 `Document` 리스트로 변환합니다.

```python
# PDF
from langchain_community.document_loaders import PyPDFLoader
docs = PyPDFLoader("file.pdf").load()

# 다른 로더 예시 (이 프로젝트에서는 미사용)
from langchain_community.document_loaders import TextLoader          # .txt
from langchain_community.document_loaders import UnstructuredWordDocumentLoader  # .docx
from langchain_community.document_loaders import CSVLoader           # .csv
from langchain_community.document_loaders import WebBaseLoader       # 웹페이지
```

### C-5. Text Splitters (텍스트 분할기)

| 분할기 | 특징 | 적합한 상황 |
|--------|------|------------|
| `RecursiveCharacterTextSplitter` | 구분자를 순서대로 시도 (`\n\n` → `\n` → ` ` → `""`) | **범용 (이 프로젝트에서 사용)** |
| `CharacterTextSplitter` | 단일 구분자로 분할 | 단순한 구조의 텍스트 |
| `TokenTextSplitter` | 토큰 수 기반 분할 | LLM 토큰 제한 맞출 때 |
| `MarkdownTextSplitter` | Markdown 헤더 기반 분할 | Markdown 문서 |
| `HTMLHeaderTextSplitter` | HTML 태그 기반 분할 | 웹페이지 |
| `SemanticChunker` | 의미적 유사도 기반 분할 | 의미 단위 보존이 중요할 때 |

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],  # 기본값 (커스텀 가능)
)

# Document 리스트 분할
chunks = splitter.split_documents(documents)

# 텍스트 직접 분할
texts = splitter.split_text("긴 텍스트...")
```

### C-6. 심화 학습 링크

| 주제 | 링크 |
|------|------|
| LangChain 공식 문서 | https://python.langchain.com/docs/introduction/ |
| Document Loaders 목록 | https://python.langchain.com/docs/integrations/document_loaders/ |
| Text Splitters 개념 | https://python.langchain.com/docs/concepts/text_splitters/ |
| RecursiveCharacterTextSplitter API | https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html |
| RAG 튜토리얼 (LangChain) | https://python.langchain.com/docs/tutorials/rag/ |
| SemanticChunker (의미 기반 분할) | https://python.langchain.com/docs/how_to/semantic-chunker/ |
| LangChain GitHub | https://github.com/langchain-ai/langchain |

---

## 별첨 D. Cohere Rerank API

> 이 프로젝트에서 **검색 결과를 재정렬(Reranking)** 하는 데 사용합니다.

### D-1. Reranking이란?

검색 시스템은 보통 **2단계 구조**로 설계됩니다:

```
Stage 1 (Retriever)                  Stage 2 (ReRanker)
┌──────────────────┐                ┌──────────────────┐
│ 전체 문서에서     │   상위 K개     │ 질의-문서 쌍을    │   최종 N개
│ 빠르게 후보 추출  │ ────────────▶ │ 정밀하게 재평가   │ ──────────▶ 결과
│ (BM25, kNN)      │                │ (Cross-Encoder)  │
└──────────────────┘                └──────────────────┘
  속도: 빠름 (ms)                     속도: 느림 (수십~수백ms)
  정밀도: 보통                        정밀도: 높음
```

- **Bi-Encoder** (Stage 1): 질의와 문서를 **각각** 임베딩하여 비교 → 빠르지만 대략적
- **Cross-Encoder** (Stage 2): 질의와 문서를 **함께** 입력하여 관련도 직접 산출 → 느리지만 정밀

### D-2. Cohere Rerank 모델

| 모델 | 특징 |
|------|------|
| `rerank-v3.5` | 최신 모델, 다국어 지원 (이 프로젝트에서 사용) |
| `rerank-english-v3.0` | 영어 특화 |
| `rerank-multilingual-v3.0` | 다국어 지원 (이전 버전) |

### D-3. 기본 사용법

```python
import cohere

client = cohere.Client(api_key="your-api-key")

# Rerank 호출
response = client.rerank(
    query="고혈압 약의 부작용",
    documents=[
        "고혈압 치료제는 두통, 어지러움 등의 부작용이 있을 수 있습니다.",
        "고혈압은 성인의 약 30%에게 영향을 미치는 질환입니다.",
        "안약 사용 시 눈이 따가울 수 있습니다.",
    ],
    model="rerank-v3.5",
    top_n=2,    # 상위 2개만 반환
)

for result in response.results:
    print(f"  index={result.index}, score={result.relevance_score:.4f}")
    # index=0, score=0.9823  ← 가장 관련 높음
    # index=1, score=0.4512
```

### D-4. API 응답 구조

```python
response.results = [
    RerankResult(
        index=0,                    # 원본 documents 리스트에서의 위치
        relevance_score=0.9823,     # 0~1 관련도 점수
    ),
    RerankResult(
        index=1,
        relevance_score=0.4512,
    ),
]
```

- `index`: 입력 documents 리스트에서의 원래 인덱스
- `relevance_score`: 질의와의 관련도 (0에 가까우면 무관, 1에 가까우면 매우 관련)

### D-5. 가입 및 API 키 발급

1. https://dashboard.cohere.com/welcome/register 에서 가입
2. 대시보드 → **API Keys** → Trial key 생성
3. Trial key: 무료, 분당 10회 제한
4. Production key: 종량제 ($2 / 1,000 검색)

### D-6. 심화 학습 링크

| 주제 | 링크 |
|------|------|
| Cohere Rerank 공식 문서 | https://docs.cohere.com/docs/rerank |
| Rerank API 레퍼런스 | https://docs.cohere.com/reference/rerank |
| Rerank 모델 소개 (v3.5) | https://docs.cohere.com/docs/rerank-overview |
| Rerank Best Practices | https://docs.cohere.com/docs/reranking-best-practices |
| Cohere Python SDK | https://github.com/cohere-ai/cohere-python |
| Cohere Dashboard | https://dashboard.cohere.com |
| Bi-Encoder vs Cross-Encoder 비교 (SBERT) | https://www.sbert.net/examples/applications/cross-encoder/README.html |

---

## 별첨 E. OpenAI Python SDK

> 이 프로젝트에서 **Embeddings API 호출** 에 사용합니다.

### E-1. 설치 및 초기화

```bash
pip install openai
# 또는
uv add openai
```

```python
from openai import OpenAI

# API 키로 초기화
client = OpenAI(api_key="sk-...")

# 환경변수 OPENAI_API_KEY가 설정되어 있으면 자동 인식
client = OpenAI()
```

### E-2. 주요 API

```python
# 1. Embeddings (이 프로젝트에서 사용)
response = client.embeddings.create(input="텍스트", model="text-embedding-3-small")
vector = response.data[0].embedding

# 2. Chat Completions (RAG의 Generation 단계에서 사용 가능)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "검색 결과를 바탕으로 답변하세요."},
        {"role": "user", "content": "질문: 고혈압 약의 부작용은?\n\n참고 문서:\n{검색 결과}"},
    ],
)
answer = response.choices[0].message.content
```

### E-3. 에러 처리

```python
from openai import APIError, RateLimitError, APITimeoutError

try:
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
except RateLimitError:
    # 요청 한도 초과 → 잠시 후 재시도
    pass
except APITimeoutError:
    # 요청 시간 초과 → 재시도 또는 배치 크기 줄이기
    pass
except APIError as e:
    # 기타 API 오류
    print(f"API 오류: {e}")
```

### E-4. 심화 학습 링크

| 주제 | 링크 |
|------|------|
| OpenAI Python SDK GitHub | https://github.com/openai/openai-python |
| API 레퍼런스 (전체) | https://platform.openai.com/docs/api-reference |
| Chat Completions 가이드 | https://platform.openai.com/docs/guides/text-generation |
| 에러 처리 가이드 | https://platform.openai.com/docs/guides/error-codes |
| Cookbook (활용 예제 모음) | https://cookbook.openai.com |

---

## 별첨 F. uv (패키지 매니저)

> 이 프로젝트의 **Python 환경 및 의존성 관리**에 사용합니다.

### F-1. uv란?

`uv`는 Rust로 작성된 **초고속 Python 패키지 매니저**입니다. `pip`, `venv`, `pip-tools`의 기능을 하나로 통합하며, 기존 도구 대비 10~100배 빠릅니다.

### F-2. 설치

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrew
brew install uv

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### F-3. 주요 명령어

```bash
# 프로젝트 초기화
uv init my-project
cd my-project

# 의존성 설치 (pyproject.toml 기반)
uv sync

# 패키지 추가
uv add openai elasticsearch langchain-community

# 패키지 제거
uv remove 패키지명

# 스크립트 실행 (가상환경 자동 활성화)
uv run python main.py
uv run python search.py "검색어"

# Python 버전 관리
uv python install 3.14
uv python pin 3.14
```

### F-4. 프로젝트 파일 구조

```
pipeline/
├── pyproject.toml     # 프로젝트 설정 & 의존성 선언
├── uv.lock            # 정확한 버전 고정 (자동 생성, 커밋 대상)
├── .python-version    # Python 버전 지정
└── .venv/             # 가상환경 (자동 생성)
```

**`pyproject.toml`** (이 프로젝트의 예):
```toml
[project]
name = "pipeline"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "elasticsearch>=9.3.0",
    "openai>=2.29.0",
    "langchain-community>=0.4.1",
    ...
]
```

### F-5. pip과의 비교

| 기능 | pip | uv |
|------|-----|-----|
| 설치 속도 | 보통 | **10~100배 빠름** |
| Lock 파일 | 없음 (requirements.txt 수동) | `uv.lock` 자동 생성 |
| 가상환경 관리 | `python -m venv` 별도 실행 | `uv sync`가 자동 생성 |
| 스크립트 실행 | `source .venv/bin/activate` 후 실행 | `uv run python ...` |
| Python 버전 관리 | pyenv 등 별도 도구 필요 | `uv python install` 내장 |

### F-6. 심화 학습 링크

| 주제 | 링크 |
|------|------|
| uv 공식 문서 | https://docs.astral.sh/uv/ |
| uv 시작하기 | https://docs.astral.sh/uv/getting-started/ |
| uv 프로젝트 관리 | https://docs.astral.sh/uv/concepts/projects/ |
| pyproject.toml 표준 | https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ |
| uv GitHub | https://github.com/astral-sh/uv |

---

## 별첨 G. BeautifulSoup (웹 크롤링)

> 이 프로젝트에서 **식약처 웹사이트에서 첨부파일을 수집**하는 데 사용합니다.

### G-1. BeautifulSoup이란?

HTML/XML 문서를 파싱하여 원하는 데이터를 추출하는 Python 라이브러리입니다. `requests`로 웹페이지를 가져오고, BeautifulSoup으로 내용을 파싱하는 조합이 일반적입니다.

### G-2. 기본 사용법

```python
import requests
from bs4 import BeautifulSoup

# 웹페이지 가져오기
response = requests.get("https://example.com")
soup = BeautifulSoup(response.text, "lxml")

# 태그 찾기
title = soup.find("title").text                   # 첫 번째 <title> 태그
links = soup.find_all("a")                        # 모든 <a> 태그
links = soup.find_all("a", class_="download")     # class="download"인 <a> 태그

# CSS 선택자
items = soup.select("div.content > p")            # CSS 선택자 사용
```

### G-3. 이 프로젝트에서의 활용 (`download_pdfs.py`)

```python
# 게시판에서 게시물 ID 추출
soup = BeautifulSoup(resp.text, "lxml")
for a in soup.find_all("a", onclick=re.compile("moveDetail")):
    match = re.search(r"moveDetail\(\s*\d+\s*,\s*(\d+)\s*\)", a.get("onclick", ""))
    if match:
        post_id = int(match.group(1))
```

### G-4. 크롤링 시 주의사항

| 항목 | 설명 |
|------|------|
| **robots.txt** | 크롤링 허용 범위를 확인 (`https://도메인/robots.txt`) |
| **요청 간격** | `time.sleep()`으로 서버에 부하를 주지 않도록 |
| **User-Agent** | 봇 차단 방지를 위해 적절한 User-Agent 설정 |
| **법적 고려** | 수집 데이터의 저작권 및 이용약관 확인 |

### G-5. 심화 학습 링크

| 주제 | 링크 |
|------|------|
| BeautifulSoup 공식 문서 | https://www.crummy.com/software/BeautifulSoup/bs4/doc/ |
| Requests 라이브러리 | https://docs.python-requests.org/en/latest/ |
| 웹 크롤링 튜토리얼 (Real Python) | https://realpython.com/beautiful-soup-web-scraper-python/ |

---

## 별첨 H. python-dotenv (환경변수 관리)

> 이 프로젝트에서 **API 키, 접속 정보 등 민감한 설정을 관리**하는 데 사용합니다.

### H-1. 왜 환경변수를 사용하는가?

```python
# ❌ 나쁜 예: 코드에 직접 하드코딩
api_key = "sk-abc123..."    # Git에 커밋되면 유출 위험!

# ✅ 좋은 예: 환경변수로 관리
api_key = os.getenv("OPENAI_API_KEY")
```

- API 키를 코드에 직접 넣으면 **Git 커밋 시 유출** 위험
- 환경(개발/운영)에 따라 다른 설정을 쉽게 전환 가능
- `.env` 파일은 `.gitignore`에 추가하여 버전 관리에서 제외

### H-2. 사용법

```bash
# .env 파일 작성
OPENAI_API_KEY=sk-abc123...
ES_URL=https://localhost:9200
DEBUG=true
```

```python
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드

api_key = os.getenv("OPENAI_API_KEY")       # 값 가져오기
debug = os.getenv("DEBUG", "false")          # 기본값 지정 가능
```

### H-3. 프로젝트에서의 패턴

```
.env.example   ← Git에 커밋 (템플릿, 실제 값 없음)
.env           ← Git에서 제외 (실제 비밀 값 포함)
.gitignore     ← .env를 제외하는 설정
```

### H-4. 심화 학습 링크

| 주제 | 링크 |
|------|------|
| python-dotenv 공식 문서 | https://saurabh-kumar.com/python-dotenv/ |
| python-dotenv GitHub | https://github.com/theskumar/python-dotenv |
| 12 Factor App (환경변수 원칙) | https://12factor.net/config |
