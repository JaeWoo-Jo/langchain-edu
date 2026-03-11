# 자취고수 - 생필품 가격 에이전트

LangChain 기반 AI 에이전트 교육용 프로젝트입니다. 공공데이터 포털의 참가격 생필품 가격정보를 OpenSearch에 적재하고, "자취고수" 페르소나의 에이전트가 가격 조회, 비교, 차트 생성까지 해주는 챗봇입니다.

## 데모

```
사용자: 쌀 가격 알려줘
자취고수: 야 쌀 20kg에 62,756원이야. 저번주보다 1,018원 올랐고,
         10kg짜리는 36,467원인데 한 달 전보다 내려갔어.
         지금 사려면 10kg짜리가 이득일지도 ㅋㅋ

사용자: 감자 가격 추이 보여줘
자취고수: [차트와 함께] 감자 1년간 39원 올랐어. 지금 안정적인 편이야.
```

## 기술 스택

| 구분 | 기술 |
|---|---|
| 백엔드 | FastAPI, LangChain v1.0, LangGraph, ChatOpenAI |
| 프론트엔드 | React 19, Vite, TypeScript, Jotai, TanStack Query |
| 데이터 | OpenSearch, KAMIS 공공데이터 API |
| 패키지 관리 | uv (백엔드), pnpm (프론트엔드) |

## 프로젝트 구조

```
├── agent/                  # 백엔드 (FastAPI + LangChain)
│   ├── app/
│   │   ├── agents/         # 에이전트 정의, 도구, 프롬프트
│   │   ├── api/routes/     # API 엔드포인트 (채팅, 스레드)
│   │   ├── services/       # 비즈니스 로직 (에이전트 서비스)
│   │   ├── core/           # 설정 (환경 변수)
│   │   └── utils/          # 유틸리티 (OpenSearch 클라이언트)
│   └── scripts/            # 데이터 수집 스크립트
├── ui/                     # 프론트엔드 (React + Vite)
│   └── src/
│       ├── components/     # 차트, 테이블, 코드 에디터 등
│       ├── hooks/          # useChat, useHistory
│       ├── pages/          # InitPage, ChatPage
│       └── services/       # API 서비스 (SSE 스트리밍)
└── docs/                   # 문서
```

## 빠른 시작

### 사전 요구사항

- Python 3.11~3.13
- Node.js 18+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python 패키지 관리)
- [pnpm](https://pnpm.io/ko/installation) (Node 패키지 관리)
- OpenSearch 접속 정보
- OpenAI API 키

### 1. 환경 변수 설정

```bash
# 백엔드
cp agent/env.sample agent/.env
# agent/.env 편집: OPENAI_API_KEY, OPENSEARCH_*, PUBLIC_DATA_API_KEY 입력

# 프론트엔드
cp ui/env.sample ui/.env
```

### 2. 의존성 설치

```bash
# 백엔드
cd agent && uv sync

# 프론트엔드
cd ui && pnpm install
```

### 3. 데이터 수집

```bash
cd agent
uv run python scripts/collect_prices.py --date 2026-03-11
```

### 4. 서버 실행

```bash
# 백엔드 (터미널 1)
cd agent
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 프론트엔드 (터미널 2)
cd ui
pnpm dev
```

http://localhost:5173 에서 접속

## 에이전트 도구

| 도구 | 기능 | UI 연결 |
|---|---|---|
| `search_price` | 품목별 최신 가격 조회 | 텍스트 답변 |
| `compare_prices` | 기간별 가격 비교 | GridViewer 테이블 |
| `create_price_chart` | 가격 추이 차트 | ChartViewer 차트 |

## 교육 목적

이 프로젝트의 핵심 학습 포인트:

1. **LangChain 에이전트 구성** — `ChatOpenAI` + `@tool` + `create_react_agent` 조립
2. **도구(Tool) 정의** — `@tool` 데코레이터로 외부 데이터소스(OpenSearch) 연동
3. **시스템 프롬프트** — 페르소나 설정으로 에이전트 성격 제어
4. **SSE 스트리밍** — 도구 호출 → 실행 → 최종 답변의 실시간 전달
5. **공공데이터 활용** — KAMIS API로 실제 가격 데이터 수집 및 적재
