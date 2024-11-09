"""Environment configuration for application settings."""

import os
from pathlib import Path
from typing import Literal

from pydantic import (
    AnyHttpUrl,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal[
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
]
LogFormat = Literal["console", "json"]


class Settings(BaseSettings):
    """App configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    ENV: Literal["dev", "production"]
    SPLIT_CHUNK_SIZE: int = 500
    SPLIT_OVERLAP_SIZE: int = 100
    LLMSHERPA_TIMEOUT: int = 60
    OLLAMA_LLM_MODEL: str
    OLLAMA_EMBEDDINGS_MODEL_NAME: str
    RAG_TOP_K: int = 14
    OLLAMA_HOST: AnyHttpUrl
    VECTOR_DB_TABLE_NAME: str = "mavarick"
    VECTOR_DB_DB_NAME: str = "vectordb"
    EMBEDDINGS_DIMENSION: int = 768
    MAX_LLM_RETRIES: int = 2

    COLLECTION_NAME: str = "mavarick"
    PROJECT_NAME: str = "Mavarick"
    API_PREFIX: str = "/api/v1"

    MILVUS_CONNECTION_URL: AnyHttpUrl
    LOG_LEVEL: LogLevel
    LOG_FORMAT: LogFormat
    TEST_FILE_NAME: str
    APP_PATH: Path = Path(__file__).parent.parent

    @property
    def TEST_FILE_PATH(self) -> str:
        """
        Returns:
            str: Absolute path to the testfile.

        """
        path = self.APP_PATH / "files" / self.TEST_FILE_NAME
        assert os.path.exists(
            path
        ), f"please ensure the test file {self.TEST_FILE_NAME} is placed under ROOT/src/app/files"
        return str(path)

    LLM_SHERPA_HOST: AnyHttpUrl

    @property
    def LLM_SHERPA_URI(self) -> str:
        """Determines the LLMSherpa endpoint based on the host used."""
        return f"{self.LLM_SHERPA_HOST}/api/parseDocument?renderFormat=all"


settings = Settings()  # pyright: ignore[reportCallIssue]
