import asyncio
import contextlib
from datetime import datetime
import json
from typing import Optional
import uuid

from app.utils.logger import log_execution, custom_logger

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from app.core.config import settings
from app.agents.prompts import system_prompt
from app.agents.tools import search_price, compare_prices, create_price_chart

from opik.integrations.langchain import OpikTracer, track_langgraph


class AgentService:
    def __init__(self):
        self.agent = None
        self._checkpointer = InMemorySaver()
        self.progress_queue: asyncio.Queue = asyncio.Queue()
        self._opik_configured = False

    def _configure_opik(self):
        """Opik 설정 (한 번만 실행)"""
        if self._opik_configured:
            return
        import opik
        opik_settings = settings.OPIK
        if opik_settings and opik_settings.URL_OVERRIDE:
            opik.configure(
                url=opik_settings.URL_OVERRIDE,
                workspace=opik_settings.WORKSPACE or "default",
                use_local=True,
            )
            custom_logger.info(f"Opik 설정 완료: {opik_settings.URL_OVERRIDE}, 프로젝트: {opik_settings.PROJECT}")
        self._opik_configured = True

    def _create_agent(self):
        """LangChain 에이전트 생성 (한 번만 생성 후 캐싱)"""
        if self.agent is not None:
            return

        self._configure_opik()

        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
        )

        tools = [search_price, compare_prices, create_price_chart]

        compiled_agent = create_react_agent(
            llm,
            tools,
            prompt=system_prompt,
            checkpointer=self._checkpointer,
        )

        # Opik 트레이싱 래핑
        opik_settings = settings.OPIK
        if opik_settings and opik_settings.PROJECT:
            opik_tracer = OpikTracer(
                project_name=opik_settings.PROJECT,
                tags=["langgraph", "react-agent"],
            )
            self.agent = track_langgraph(compiled_agent, opik_tracer)
            custom_logger.info("Opik track_langgraph 래핑 완료")
        else:
            self.agent = compiled_agent

    # 실제 대화 로직
    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """LangChain Messages 형식의 쿼리를 처리하고 AIMessage 형식으로 반환합니다."""
        try:
            # 에이전트 초기화 (한 번만)
            self.progress_queue = asyncio.Queue()  # 요청별 큐 초기화
            self._create_agent()

            custom_logger.info(f"사용자 메시지: {user_messages}")

            # IMP: LangGraph 에이전트에 사용자의 메시지를 HumanMessage 형태로 전달하고, 
            # thread_id를 통해 대화 문맥(Context)을 유지하며 비동기 스트리밍(astream)으로 실행하는 구현.
            agent_stream = self.agent.astream(
                {"messages": [HumanMessage(content=user_messages)]},
                config={"configurable": {"thread_id": str(thread_id)}},
                stream_mode="updates",
            )

            agent_iterator = agent_stream.__aiter__()
            agent_task = asyncio.create_task(agent_iterator.__anext__())
            progress_task = asyncio.create_task(self.progress_queue.get())

            while True:
                pending = {agent_task}
                if progress_task is not None:
                    pending.add(progress_task)

                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                if progress_task in done:
                    try:
                        progress_event = progress_task.result()
                        yield json.dumps(progress_event, ensure_ascii=False)
                        progress_task = asyncio.create_task(self.progress_queue.get())
                    except asyncio.CancelledError:
                        progress_task = None
                    except Exception as e:
                        # progress_task에서 예외 발생 시 로그만 남기고 계속 진행
                        custom_logger.error(f"Error in progress_task: {e}")
                        progress_task = None

                if agent_task in done:
                    try:
                        chunk = agent_task.result()
                    except StopAsyncIteration:
                        agent_task = None
                        break
                    except Exception as e:
                        # Task에서 발생한 예외 처리
                        custom_logger.error(f"Error in agent_task: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        agent_task = None
                        # 에러를 스트리밍으로 전송
                        error_response = {
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                        yield json.dumps(error_response, ensure_ascii=False)
                        break

                    custom_logger.info(f"에이전트 청크: {chunk}")
                    try:
                        for step, event in chunk.items():
                            if not event or not (step in ["model", "agent", "tools"]):
                                continue
                            messages = event.get("messages", [])
                            if len(messages) == 0:
                                continue
                            message = messages[0]
                            if step in ("model", "agent"):
                                tool_calls = message.tool_calls
                                if not tool_calls:
                                    # 도구 호출 없음 = 최종 답변
                                    content = message.content or ""
                                    metadata = self._parse_metadata(content)
                                    yield f'{{"step": "done", "message_id": {json.dumps(str(uuid.uuid4()))}, "role": "assistant", "content": {json.dumps(content, ensure_ascii=False)}, "metadata": {json.dumps(metadata, ensure_ascii=False)}, "created_at": "{datetime.utcnow().isoformat()}"}}'
                                    continue
                                tool = tool_calls[0]
                                if tool.get("name") == "ChatResponse":
                                    args = tool.get("args", {})
                                    metadata = args.get("metadata")
                                    custom_logger.info("========================================")
                                    custom_logger.info(args)
                                    yield f'{{"step": "done", "message_id": {json.dumps(args.get("message_id"))}, "role": "assistant", "content": {json.dumps(args.get("content"), ensure_ascii=False)}, "metadata": {json.dumps(self._handle_metadata(metadata), ensure_ascii=False)}, "created_at": "{datetime.utcnow().isoformat()}"}}'
                                else:
                                    yield f'{{"step": "model", "tool_calls": {json.dumps([tool["name"] for tool in tool_calls])}}}'
                            if step == "tools":
                                yield f'{{"step": "tools", "name": {json.dumps(message.name)}, "content": {message.content}}}'
                    except Exception as e:
                        # 청크 처리 중 예외 발생
                        custom_logger.error(f"Error processing chunk: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        error_response = {
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": "데이터 처리 중 오류가 발생했습니다.",
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                        yield json.dumps(error_response, ensure_ascii=False)
                        break

                    agent_task = asyncio.create_task(agent_iterator.__anext__())

            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            while not self.progress_queue.empty():
                try:
                    remaining = self.progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                yield json.dumps(remaining, ensure_ascii=False)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            custom_logger.error(f"Error in process_query: {e}")
            custom_logger.error(error_trace)
            
            error_content = f"처리 중 오류가 발생했습니다. 다시 시도해주세요."
            error_metadata = {}
            
            # 에러 응답을 스트리밍으로 전송 (HTTPException 대신)
            error_response = {
                "step": "done",
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": error_content,
                "metadata": error_metadata,
                "created_at": datetime.utcnow().isoformat(),
                "error": str(e) if not isinstance(e, GraphRecursionError) else None
            }
            yield json.dumps(error_response, ensure_ascii=False)

    @log_execution
    def _handle_metadata(self, metadata) -> dict:
        custom_logger.info("========================================")
        custom_logger.info(metadata)
        result = {}
        if metadata:
            for k, v in metadata.items():
                result[k] = v
        return result

    def _parse_metadata(self, content: str) -> dict:
        """에이전트 응답에서 [TABLE_DATA]...[/TABLE_DATA], [CHART_DATA]...[/CHART_DATA] 태그를 파싱합니다."""
        import re
        metadata = {}

        table_match = re.search(r'\[TABLE_DATA\](.*?)\[/TABLE_DATA\]', content, re.DOTALL)
        if table_match:
            try:
                metadata["data"] = json.loads(table_match.group(1))
            except json.JSONDecodeError:
                pass

        chart_match = re.search(r'\[CHART_DATA\](.*?)\[/CHART_DATA\]', content, re.DOTALL)
        if chart_match:
            try:
                metadata["chart"] = json.loads(chart_match.group(1))
            except json.JSONDecodeError:
                pass

        return metadata
