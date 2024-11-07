import logging
from typing import Annotated, Literal

from pydantic import (
    StringConstraints,
    computed_field,
)
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    POSTGRES_SERVER: Annotated[
        str, StringConstraints(strip_whitespace=True, strict=True, min_length=3)
    ]
    POSTGRES_PORT: int = 5432

    POSTGRES_USER: Annotated[
        str, StringConstraints(strip_whitespace=True, strict=True, min_length=3)
    ]

    POSTGRES_PASSWORD: Annotated[
        str, StringConstraints(strip_whitespace=True, strict=True, min_length=15)
    ]
    SERVER_STATE: Literal["dev", "production"]
    DOCKER: bool
    PRODUCTION: bool = False
    SPLIT_CHUNK_SIZE: int = 500
    SPLIT_OVERLAP_SIZE: int = 100
    LLMSHERPA_TIMEOUT: int = 60
    LLM_MODEL: str = "gemma2:2b"  # llama3.1:8b-instruct-q5_0"
    RAG_TOP_K: int = 15
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDINGS_MODEL_NAME: str = "nomic-embed-text"
    VECTOR_DB_TABLE_NAME: str = "mavarick"
    VECTOR_DB_DB_NAME: str = "vectordb"
    EMBEDDINGS_DIMENSION: int = 768
    LIGHT_RAG_WOKING_DIR: str = "light_rag_dir"

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


settings = Settings(_env_file="./.env")
