"""Environment configuration for application settings."""
import logging
from typing import Literal

from pydantic import (
    computed_field,
)
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App configuration settings."""

    SERVER_STATE: Literal["dev", "production"]
    DOCKER: bool
    PRODUCTION: bool = False
    SPLIT_CHUNK_SIZE: int = 500
    SPLIT_OVERLAP_SIZE: int = 100
    LLMSHERPA_TIMEOUT: int = 60
    LLM_MODEL: str = "gemma2:2b"
    RAG_TOP_K: int = 14
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDINGS_MODEL_NAME: str = "nomic-embed-text"
    VECTOR_DB_TABLE_NAME: str = "mavarick"
    VECTOR_DB_DB_NAME: str = "vectordb"
    EMBEDDINGS_DIMENSION: int = 768
    MAX_LLM_RETRIES: int = 2

    COLLECTION_NAME: str = "mavarick"
    PROJECT_NAME:str="Mavarick"
    API_PREFIX: str = "/api/v1"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_DATABASE_URL(self) -> str:
        """Constructs the base URL for the PostgreSQL database connection."""
        return f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}"

    @computed_field
    @property
    def LLM_SHERPA_URI(self) -> str:
        """Determines the LLMSherpa API URI based on whether Docker is used."""
        if self.DOCKER:
            return "http://nlm-ingestor:5001/api/parseDocument?renderFormat=all"
        return "http://localhost:5001/api/parseDocument?renderFormat=all"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def LOG_LEVEL(self) -> logging.INFO | logging.DEBUG:
        """Sets log level based on production status."""
        if self.PRODUCTION:
            return logging.INFO
        return logging.DEBUG

    @computed_field
    @property
    def MILVUS_CONNECTION_URL(self) -> str:
        """Provides the Milvus connection URL based on Docker environment."""
        if self.DOCKER:
            return "http://milvus-standalone:19530"
        return "http://localhost:19530"


settings = Settings(_env_file="./.env")
