from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class OpikSettings(BaseSettings):
    """Opik configuration."""

    URL_OVERRIDE: str | None = Field(default=None, description="Opik base URL")
    # Optional if you are using Opik Cloud:
    API_KEY: str | None = Field(default=None, description="opik cloud api key here")
    WORKSPACE: str | None = Field(default=None, description="your workspace name")
    PROJECT: str | None = Field(default=None, description="your project name")


class Settings(BaseSettings):
    # API 설정
    API_V1_PREFIX: str

    CORS_ORIGINS: List[str] = ["*"]
    
    # IMP: LangChain 객체 및 LLM 연동에 사용되는 필수 설정값(API Key 등)
    # LangChain 설정
    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    
    COHERE_API_KEY: str | None = None

    # Elasticsearch 설정
    ELASTICSEARCH_HOST: str = "https://elasticsearch-edu.didim365.app"
    ELASTICSEARCH_USERNAME: str = "elastic"
    ELASTICSEARCH_PASSWORD: str = "elastic"
    ELASTICSEARCH_VERIFY_CERTS: bool = True
    ELASTICSEARCH_INDEX_PRICES: str = "prices-daily-goods"
    ELASTICSEARCH_INDEX_RAG: str = "edu-price-info"
    ELASTICSEARCH_INDEX_NUTRITION: str = "nutrition-info"

    # 공공데이터 포털
    PUBLIC_DATA_API_KEY: str = ""

    # DeepAgents 설정
    DEEPAGENT_RECURSION_LIMIT: int = 40

    OPIK: OpikSettings | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
        extra="ignore",
    )

settings = Settings()

