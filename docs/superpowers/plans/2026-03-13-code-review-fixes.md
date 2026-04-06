# 코드 리뷰 이슈 수정 구현 계획

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GitHub 이슈 #1의 코드 리뷰 지적사항 중 실제 문제(1, 2, 4번)를 수정하여 대화 연속성 및 리소스 효율성 확보

**Architecture:** `AgentService`를 모듈 레벨 싱글턴으로 변경하고, `InMemorySaver`와 agent를 인스턴스 생명주기 동안 유지. OpenSearch 클라이언트도 싱글턴으로 변경. `chat.py`에서 매 요청마다 새 인스턴스를 만들지 않도록 수정.

**Tech Stack:** Python, FastAPI, LangChain/LangGraph, OpenSearch

---

## 파일 구조

| 파일 | 변경 | 역할 |
|------|------|------|
| `agent/app/utils/opensearch_client.py` | 수정 | 싱글턴 패턴 적용 |
| `agent/app/services/agent_service.py` | 수정 | checkpointer/agent 캐싱 |
| `agent/app/api/routes/chat.py` | 수정 | 모듈 레벨 `AgentService` 사용 |
| `agent/tests/test_agent_service.py` | 생성 | AgentService 단위 테스트 |
| `agent/tests/test_opensearch_client.py` | 생성 | OpenSearch 싱글턴 테스트 |

---

## Chunk 1: OpenSearch 클라이언트 싱글턴

### Task 1: OpenSearch 클라이언트 싱글턴 적용

**Files:**
- Modify: `agent/app/utils/opensearch_client.py`
- Create: `agent/tests/test_opensearch_client.py`

- [ ] **Step 1: 싱글턴 테스트 작성**

`agent/tests/test_opensearch_client.py` 생성:

```python
from unittest.mock import patch, MagicMock


def test_get_opensearch_client_returns_singleton():
    """get_opensearch_client()가 항상 같은 인스턴스를 반환하는지 확인"""
    import app.utils.opensearch_client as mod
    mod._client = None  # 캐시 초기화

    with patch.object(mod, "OpenSearch") as mock_cls:
        mock_cls.return_value = MagicMock()
        client1 = mod.get_opensearch_client()
        client2 = mod.get_opensearch_client()
        assert client1 is client2
        mock_cls.assert_called_once()  # 생성자가 한 번만 호출됨
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd agent && uv run pytest tests/test_opensearch_client.py -v
```

예상: FAIL — `_client` 속성 없음 또는 `mock_cls`가 두 번 호출됨

- [ ] **Step 3: 싱글턴 구현**

`agent/app/utils/opensearch_client.py`를 다음으로 수정:

```python
from opensearchpy import OpenSearch
from app.core.config import settings

_client: OpenSearch | None = None


def get_opensearch_client() -> OpenSearch:
    """OpenSearch 클라이언트 싱글턴을 반환합니다."""
    global _client
    if _client is None:
        _client = OpenSearch(
            hosts=[settings.OPENSEARCH_HOST],
            http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
            verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
            ssl_show_warn=False,
        )
    return _client
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd agent && uv run pytest tests/test_opensearch_client.py -v
```

예상: PASS

- [ ] **Step 5: 커밋**

```bash
git add agent/app/utils/opensearch_client.py agent/tests/test_opensearch_client.py
git commit -m "fix: OpenSearch 클라이언트 싱글턴 패턴 적용"
```

---

## Chunk 2: AgentService 캐싱 및 싱글턴

### Task 2: AgentService의 checkpointer/agent 캐싱

**Files:**
- Modify: `agent/app/services/agent_service.py:14-51`
- Create: `agent/tests/test_agent_service.py`

- [ ] **Step 1: AgentService 캐싱 테스트 작성**

`agent/tests/test_agent_service.py` 생성:

```python
import os
import pytest
from unittest.mock import patch, MagicMock

# settings 로딩을 위한 환경변수 설정 (테스트 환경에서 .env가 없을 수 있음)
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY 환경변수 필요"
)


def test_agent_created_once():
    """_create_agent()가 여러 번 호출되어도 agent를 한 번만 생성하는지 확인"""
    with patch("app.services.agent_service.ChatOpenAI"), \
         patch("app.services.agent_service.create_react_agent") as mock_create, \
         patch("app.services.agent_service.InMemorySaver"):

        mock_create.return_value = MagicMock()

        from app.services.agent_service import AgentService
        service = AgentService()
        service._create_agent()
        service._create_agent()

        mock_create.assert_called_once()


def test_checkpointer_persists():
    """checkpointer가 인스턴스 수명 동안 유지되는지 확인"""
    with patch("app.services.agent_service.ChatOpenAI"), \
         patch("app.services.agent_service.create_react_agent"), \
         patch("app.services.agent_service.InMemorySaver") as mock_saver:

        mock_saver.return_value = MagicMock()

        from app.services.agent_service import AgentService
        service = AgentService()
        service._create_agent()
        service._create_agent()

        mock_saver.assert_called_once()  # checkpointer도 한 번만 생성
```

**참고:** import를 메서드 내부에 유지하면 테스트에서 patch가 동작하지 않으므로, 구현 시 LangChain 관련 import는 모듈 최상위로 이동해야 함. 단, `settings`와 `tools`의 import 체인으로 인해 `.env` 없는 환경에서는 import 자체가 실패할 수 있으므로 `skipif`로 보호.

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd agent && uv run pytest tests/test_agent_service.py -v
```

예상: FAIL — `mock_create`/`mock_saver`가 두 번 호출됨 (캐싱 없으므로)

- [ ] **Step 3: AgentService 캐싱 구현**

`agent/app/services/agent_service.py` 수정 — `__init__`과 `_create_agent`만 변경:

```python
class AgentService:
    def __init__(self):
        self.agent = None
        self._checkpointer = InMemorySaver()
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    def _create_agent(self):
        """LangChain 에이전트 생성 (한 번만 생성 후 캐싱)"""
        if self.agent is not None:
            return

        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
        )

        tools = [search_price, compare_prices, create_price_chart]

        self.agent = create_react_agent(
            llm,
            tools,
            prompt=system_prompt,
            checkpointer=self._checkpointer,
        )
```

추가 변경사항:

1. LangChain 관련 import를 모듈 최상위로 이동:
```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from app.core.config import settings
from app.agents.prompts import system_prompt
from app.agents.tools import search_price, compare_prices, create_price_chart
```

2. `process_query`의 호출부 수정 (line 51):
```python
# 변경 전
self._create_agent(thread_id=thread_id)
# 변경 후
self._create_agent()
```

3. `process_query` 시작 시 progress_queue 초기화 추가 (동시 요청 안전):
```python
async def process_query(self, user_messages: str, thread_id: uuid.UUID):
    try:
        self.progress_queue = asyncio.Queue()  # 요청별 큐 초기화
        self._create_agent()
        ...
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd agent && uv run pytest tests/test_agent_service.py -v
```

예상: PASS

- [ ] **Step 5: 커밋**

```bash
git add agent/app/services/agent_service.py agent/tests/test_agent_service.py
git commit -m "fix: AgentService checkpointer/agent 캐싱으로 대화 연속성 확보"
```

### Task 3: chat.py에서 AgentService 싱글턴 사용

**Files:**
- Modify: `agent/app/api/routes/chat.py:9-33`

- [ ] **Step 1: chat.py 수정**

`agent/app/api/routes/chat.py`에서 `AgentService()`를 모듈 레벨로 이동:

```python
chat_router = APIRouter()
_agent_service = AgentService()
```

`event_generator()` 내부의 `agent_service = AgentService()`(line 33)를 삭제하고, `_agent_service`로 변경:

```python
async def event_generator():
    try:
        yield f'data: {{"step": "model", "tool_calls": ["Planning"]}}\n\n'
        async for chunk in _agent_service.process_query(
            user_messages=request.message,
            thread_id=thread_id
        ):
            yield f"data: {chunk}\n\n"
```

- [ ] **Step 2: 기존 테스트로 회귀 검증**

```bash
cd agent && uv run pytest tests/test_main.py -v
```

예상: PASS (기본 엔드포인트 테스트 통과)

- [ ] **Step 3: 커밋**

```bash
git add agent/app/api/routes/chat.py
git commit -m "fix: AgentService를 모듈 레벨 싱글턴으로 변경"
```

---

## 수정하지 않는 항목

| 이슈 | 사유 |
|------|------|
| #3 인덱스명 하드코딩 | 버그 아닌 개선사항. 현재 인덱스가 하나뿐이고 변경 계획 없음 |
| #5 테스트 부재 | `tests/`에 테스트 파일 존재. 다만 `test_v8_scenarios.py`는 현재 에이전트와 무관한 시나리오 |
