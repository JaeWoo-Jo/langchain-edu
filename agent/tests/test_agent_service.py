from unittest.mock import patch, MagicMock


def test_agent_created():
    """_create_agent()가 에이전트를 정상 생성하는지 확인"""
    with patch("langchain_openai.ChatOpenAI"), \
         patch("app.core.config.settings") as mock_settings:

        mock_settings.OPENAI_MODEL = "gpt-4o"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPIK = None
        mock_settings.DEEPAGENT_RECURSION_LIMIT = 40

        from app.services.agent_service import AgentService
        service = AgentService()
        service.checkpointer = MagicMock()

        with patch("app.agents.price_agent.create_deep_agent") as mock_create:
            mock_create.return_value = MagicMock()
            service._create_agent()
            mock_create.assert_called_once()
