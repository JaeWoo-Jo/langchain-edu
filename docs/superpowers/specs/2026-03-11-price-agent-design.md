# 생필품 가격 에이전트 설계 문서

## 개요

공공데이터 포털의 참가격 생필품 가격정보를 OpenSearch에 적재하고, LangChain 에이전트가 이를 활용하여 자취생에게 가격 정보와 생활 조언을 제공하는 시스템.

## 페르소나: 자취고수

자취 경험이 풍부한 선배가 후배 자취생에게 장보기 꿀팁을 알려주는 캐릭터.

- **톤**: 반말, 친근, 실용적
- **역할**: 가격 정보 안내 + 저렴한 품목 기반 요리/생활 추천
- **예시**:
  - "야 계란 지금 사면 비싸, 다음주에 사"
  - "요즘 양파가 개싸졌어ㅋㅋ 계란이랑 같이 사서 계란볶음밥 해먹어"
  - "이마트가 500원 싸긴 한데 교통비 생각하면 걍 가까운 데서 사"

## 시스템 구조

```
[수집 모듈 — 1회성 또는 주기적 실행]
  공공데이터 포털 참가격 API → OpenSearch (prices-daily-goods 인덱스)

[에이전트 — 사용자 요청 시 실행]
  사용자 질문 → LangChain 에이전트 (자취고수 페르소나)
                  ├── search_price       → 품목 가격 검색 → 텍스트 답변
                  ├── compare_prices     → 기간별 가격 비교 → 테이블 (GridViewer)
                  └── create_price_chart → 가격 추이 차트 → 차트 (ChartViewer)
```

## 데이터 수집 모듈

### 파일: `agent/scripts/collect_prices.py`

공공데이터 포털 참가격 생필품 가격정보 API를 호출하여 OpenSearch에 적재하는 독립 스크립트.

### 공공데이터 API 정보

- **API 키**: 환경 변수 `PUBLIC_DATA_API_KEY`에서 로드
- **인증 방식**: 서비스 키 (URL 파라미터)

### OpenSearch 인덱스

- **인덱스명**: `prices-daily-goods`
- **문서 구조** (API 응답에 따라 조정):

```json
{
  "item_name": "계란(30개)",
  "item_code": "1234",
  "category": "농산물",
  "price": 5980,
  "unit": "30개",
  "market_name": "하나로마트 양재점",
  "market_type": "대형마트",
  "region": "서울특별시 서초구",
  "date": "2026-03-11",
  "collected_at": "2026-03-11T10:00:00"
}
```

### OpenSearch 접속 정보

- **Host**: `https://bigdata04.didim365.co:9201`
- **인증**: admin / (env에서 로드)
- **SSL 검증**: 비활성화

## 에이전트 도구 설계

### 도구 1: search_price

- **목적**: 품목명으로 최신 가격 검색
- **입력**: `item_name: str` (예: "계란", "우유")
- **동작**: OpenSearch에서 해당 품목의 최신 가격 데이터 조회
- **출력**: 품목명, 가격, 판매처, 지역 정보를 텍스트로 반환
- **UI 연결**: 텍스트 답변

### 도구 2: compare_prices

- **목적**: 품목의 기간별 가격 변동 비교
- **입력**: `item_name: str`, `period: str` (기본값: "1주")
- **동작**: OpenSearch 날짜 범위 쿼리 + 집계
- **출력**: 날짜별 가격 데이터를 테이블 형식으로 반환
- **UI 연결**: `metadata.data` → GridViewer 테이블 표시

### 도구 3: create_price_chart

- **목적**: 품목의 가격 추이 차트 생성
- **입력**: `item_name: str`, `period: str` (기본값: "1개월")
- **동작**: OpenSearch 집계 → Highcharts 형식 변환
- **출력**: Highcharts JSON 구조 반환
- **UI 연결**: `metadata.chart` → ChartViewer 차트 표시

## 에이전트 구성

### _create_agent() 변경

```python
def _create_agent(self, thread_id):
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from app.core.config import settings
    from app.agents.prompts import SYSTEM_PROMPT
    from app.agents.tools import search_price, compare_prices, create_price_chart

    llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
    )

    tools = [search_price, compare_prices, create_price_chart]

    self.agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
```

### 시스템 프롬프트 핵심 지시

- 자취고수 페르소나로 반말 사용
- 가격 정보 요청 시 도구로 데이터 조회 후 답변
- 저녁/요리 추천 시 현재 저렴한 품목을 기반으로 추천
- 가격 비교 요청 시 테이블 데이터 반환
- 추이 확인 요청 시 차트 데이터 반환
- 최종 응답은 반드시 ChatResponse 형식으로 반환

## 수정/생성 파일 목록

| 파일 | 작업 | 설명 |
|---|---|---|
| `agent/scripts/collect_prices.py` | 신규 | 공공데이터 API → OpenSearch 적재 스크립트 |
| `agent/app/agents/tools.py` | 신규 | @tool 도구 3개 (search_price, compare_prices, create_price_chart) |
| `agent/app/agents/dummy.py` | 삭제 | mock 에이전트 제거 |
| `agent/app/agents/prompts.py` | 수정 | 자취고수 페르소나 시스템 프롬프트 |
| `agent/app/services/agent_service.py` | 수정 | _create_agent()에서 실제 에이전트 생성 |
| `agent/.env` | 수정 | PUBLIC_DATA_API_KEY, OPENSEARCH_* 접속정보 추가 |
| `agent/app/core/config.py` | 수정 | 신규 환경 변수 설정 추가 |
| `agent/pyproject.toml` | 수정 | opensearch-py 의존성 추가 |

## 환경 변수 추가

```env
# 공공데이터 포털
PUBLIC_DATA_API_KEY=...

# OpenSearch
OPENSEARCH_HOST=https://bigdata04.didim365.co:9201
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=...
OPENSEARCH_VERIFY_CERTS=false
```

## 사용 시나리오

### 시나리오 1: 단순 가격 조회
```
사용자: "계란 얼마야?"
에이전트: [search_price("계란") 호출]
         "야 계란 지금 30개에 5,980원이야.
          하나로마트가 제일 싼 편이고, 이마트는 6,200원 정도 하더라."
```

### 시나리오 2: 가격 비교
```
사용자: "우유 가격 지난주랑 비교해줘"
에이전트: [compare_prices("우유", "1주") 호출]
         "우유 지난주 대비 200원 올랐어ㅠ
          근데 1+1 행사하는 데 있으니까 그거 노려봐"
         → metadata.data에 테이블 데이터 포함
```

### 시나리오 3: 저녁 추천
```
사용자: "오늘 저녁 뭐 해먹지?"
에이전트: [search_price로 여러 품목 가격 조회]
         "야 요즘 양파가 개싸졌어ㅋㅋ 계란이랑 같이 사서
          계란볶음밥 해먹어. 재료비 3천원이면 2끼 해결됨 ㅇㅇ"
```

### 시나리오 4: 가격 추이
```
사용자: "라면 가격 추이 보여줘"
에이전트: [create_price_chart("라면", "1개월") 호출]
         "라면 한 달 가격 추이 뽑아봤어.
          월초에 좀 올랐다가 지금은 안정된 편이야."
         → metadata.chart에 차트 데이터 포함
```
