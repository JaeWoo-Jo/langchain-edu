# Data 수집 파이프라인

## 1. Elasticsearch 적재 파이프라인

### 연결 설정
| 항목 | 값 |
|------|-----|
| URL | `https://elasticsearch-edu.didim365.app` |
| User | `elastic` |
| Password | `FJl79PA7mMIJajxB1OHgdLEe` |
| Index | `edu-medicine-info` |
| Content Field | `content` |

### 아키텍처
```
PDF/HWP 파일 → 파싱 → 텍스트 청킹 → 임베딩 생성 → Elasticsearch 적재
          (PyPDFLoader/pyhwp) (RecursiveCharacter) (OpenAI)    (bulk API)
```

### 기능
- BM25 / Vector 하이브리드 검색
- Cohere Rerank v3.5 기반 ReRanker

### 구현 현황

| 단계 | 파일 | 상태 | 설명 |
|------|------|------|------|
| 프로젝트 설정 | `pyproject.toml` | ✅ 완료 | uv 기반, 의존성 설치 완료 |
| 환경 설정 | `config.py`, `.env.example` | ✅ 완료 | ES/OpenAI/청킹 파라미터 관리 |
| 인덱스 매핑 | `index_mapping.py` | ✅ 완료 | content(text) + content_vector(dense_vector) + metadata |
| 첨부파일 다운로드 | `download_pdfs.py` | ✅ 완료 | nedrug.mfds.go.kr 게시판 21, 124 전체 첨부파일 (26개: PDF 19 + HWP 7) |
| PDF 파싱 | `pdf_loader.py` | ✅ 완료 | PyPDFLoader로 페이지별 Document 로드 |
| HWP 파싱 | `hwp_loader.py` | ✅ 완료 | pyhwp(hwp5txt)로 텍스트 추출 |
| 텍스트 청킹 | `chunker.py` | ✅ 완료 | RecursiveCharacterTextSplitter (500자, 100 overlap) |
| 임베딩 생성 | `embedder.py` | ✅ 완료 | OpenAI text-embedding-3-small, 100건 배치, 재시도 로직 |
| ES 적재 | `es_client.py` | ✅ 완료 | bulk API 대량 적재 |
| 메인 파이프라인 | `main.py` | ✅ 완료 | CLI, 파일별 처리, 개별 파일 지정, 에러 핸들링 |
| 검색 테스트 | `search.py` | ✅ 완료 | BM25/Vector/Hybrid 검색 + Cohere ReRank v3.5 |

### 사용법
```bash
# .env 설정 후
uv run python main.py                                    # data/ 전체 처리 (PDF+HWP)
uv run python main.py data/bbs_124/some.hwp              # 개별 파일 지정
uv run python main.py data/bbs_21/a.pdf data/bbs_124/b.hwp  # 여러 파일 지정
uv run python main.py --chunk-size 1000                  # 청크 크기 조정
uv run python main.py --recreate-index                   # 인덱스 재생성

# 검색 테스트
uv run python search.py "고혈압 약"                       # 전체 모드 (BM25+Vector+Hybrid)
uv run python search.py "갱년기 치료" --mode bm25         # BM25만
uv run python search.py "안약 사용법" --mode hybrid --rerank  # Hybrid + ReRanker
uv run python search.py "변비약" --top-k 3                # 상위 3건
```

### 데이터 소스
| 게시판 | URL | 파일 수 | 내역 |
|--------|-----|---------|------|
| 질환별 정보 (21) | `https://nedrug.mfds.go.kr/bbs/21` | 3개 | PDF 3 |
| 가정상비약 (124) | `https://nedrug.mfds.go.kr/bbs/124` | 23개 | PDF 16 + HWP 7 |

저장 위치: `data/bbs_21/`, `data/bbs_124/`

### Cohere ReRanker 설정 가이드

#### 1. Cohere 가입
1. https://dashboard.cohere.com/welcome/register 접속
2. 이메일 또는 Google/GitHub 계정으로 회원가입
3. 가입 완료 후 대시보드 진입

#### 2. API 키 발급
1. 대시보드 좌측 메뉴 **API Keys** 클릭
2. **Trial key** (무료) 또는 **Production key** (유료) 생성
3. Trial key: 무료, rate limit 있음 (분당 10회), 테스트/개발용
4. Production key: 종량제 과금, Rerank v3.5 기준 $2/1,000 검색

#### 3. 환경변수 설정
```bash
# .env 파일에 추가
COHERE_API_KEY=your_cohere_api_key_here
```

#### 4. 사용
```bash
uv run python search.py "고혈압 약 부작용" --mode hybrid --rerank
```

