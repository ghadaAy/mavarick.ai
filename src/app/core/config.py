import logging
from typing import Literal

from pydantic import (
    computed_field,
)
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVER_STATE: Literal["dev", "production"]
    DOCKER: bool
    PRODUCTION: bool = False
    SPLIT_CHUNK_SIZE: int = 500
    SPLIT_OVERLAP_SIZE: int = 100
    LLMSHERPA_TIMEOUT: int = 60
    LLM_MODEL: str = "llama3.1:8b-instruct-q5_0"
    RAG_TOP_K: int = 15
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDINGS_MODEL_NAME: str = "nomic-embed-text"
    VECTOR_DB_TABLE_NAME: str = "mavarick"
    VECTOR_DB_DB_NAME: str = "vectordb"
    EMBEDDINGS_DIMENSION: int = 768
    MAX_LLM_RETRIES: int = 2

    COLLECTION_NAME: str = "mavarick"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_DATABASE_URL(self) -> str:
        return f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}"

    @computed_field
    @property
    def LLM_SHERPA_URI(self) -> str:
        if self.DOCKER:
            return "http://nlm-ingestor:5001/api/parseDocument?renderFormat=all"
        return "http://localhost:5001/api/parseDocument?renderFormat=all"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ASYNC_DATABASE_URI(self) -> str:
        return f"postgresql+asyncpg://{self.BASE_DATABASE_URL}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def SYNC_DATABASE_URI(self) -> str:
        return f"postgresql+psycopg://{self.BASE_DATABASE_URL}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def LOG_LEVEL(self) -> logging.INFO | logging.DEBUG:
        if self.PRODUCTION:
            return logging.INFO
        return logging.DEBUG

    @computed_field
    @property
    def MILVUS_CONNECTION_URL(self) -> str:
        if self.DOCKER:
            return "http://milvus-standalone:19530"
        return "http://localhost:19530"


settings = Settings(_env_file="./src/.env")
