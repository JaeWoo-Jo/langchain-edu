from unittest.mock import patch, MagicMock


def test_get_elasticsearch_client_returns_singleton():
    """get_elasticsearch_client()가 항상 같은 인스턴스를 반환하는지 확인"""
    import app.utils.opensearch_client as mod
    mod._client = None  # 캐시 초기화

    with patch.object(mod, "Elasticsearch") as mock_cls:
        mock_cls.return_value = MagicMock()
        client1 = mod.get_elasticsearch_client()
        client2 = mod.get_elasticsearch_client()
        assert client1 is client2
        mock_cls.assert_called_once()  # 생성자가 한 번만 호출됨
