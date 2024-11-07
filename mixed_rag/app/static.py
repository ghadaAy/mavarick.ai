from enum import Enum
from typing import Any, Generic, TypeVar

from llama_index.embeddings.ollama import OllamaEmbedding

from app.core.config import settings

T_ENUM = TypeVar("T_ENUM", bound=Enum)
T_VALUE = TypeVar("T_VALUE", bound=Any)


class EnumMixin(Generic[T_VALUE]):
    @classmethod
    def by_value(cls: type[T_ENUM], value: T_VALUE) -> T_ENUM | None:  # type: ignore[misc]
        for member in cls:
            if member.value == value:
                return member
        return None

    @classmethod
    def members(cls: type[T_ENUM]) -> list[T_ENUM]:  # type: ignore[misc]
        return list(cls)

    @classmethod
    def values(cls: type[T_ENUM]) -> list[T_VALUE]:  # type: ignore[misc]
        return [member.value for member in cls]

    @classmethod
    def choices(cls: type[T_ENUM]) -> list[tuple[T_VALUE, T_VALUE]]:  # type: ignore[misc]
        return [(member.value, member.value) for member in cls]


ollama_embedding = OllamaEmbedding(
    model_name=settings.OLLAMA_EMBEDDINGS_MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
    ollama_additional_kwargs={"mirostat": 0},
)
