import asyncio  # noqa: D100
from itertools import chain

from langchain_text_splitters import RecursiveCharacterTextSplitter
from result import Ok, Result
from structlog import get_logger

from app.core.config import settings
from app.core.custom_error import AppError
from app.indexers.abc import IStrategy


class TokenSizeSplitting(IStrategy):
    """strategy to split a file contents by token size."""

    def __init__(
        self,
        chunks: list[str],
        chunk_size: int = settings.SPLIT_CHUNK_SIZE,
        chunk_overlap: int = settings.SPLIT_OVERLAP_SIZE,
    ) -> None:
        """
        chunks (list[str]): list of text chunks.

        Args:
            chunks (list[str]): chunks to normalize.
            chunk_size (int, optional): target chunk size. Defaults to settings.SPLIT_CHUNK_SIZE.
            chunk_overlap (int, optional): maximum overlap between chunks. Defaults to settings.SPLIT_OVERLAP_SIZE.

        """
        self.chunks = chunks
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.logger = get_logger("token splitter")

    async def execute(self) -> Result[list[str], AppError]:
        """
        split a file by token size.

        Returns:
            Result[list[str], AppError]: Ok of list of chunks or Error

        """
        coros = [
            asyncio.to_thread(self.splitter.split_text, chunk) for chunk in self.chunks
        ]
        result = await asyncio.gather(*coros)
        return Ok(list(chain.from_iterable(result)))
