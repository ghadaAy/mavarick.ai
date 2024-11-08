import asyncio
import os
from typing import Any

import aiofiles
import magic
import msgspec
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llmsherpa.readers import LayoutPDFReader
from result import Ok, Result

from app.core.config import settings
from app.core.custom_error import CustomError
from app.core.custom_logger import logger
from app.file_utils.file_loaders import (
    read_pdf,
)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=settings.SPLIT_CHUNK_SIZE,
    chunk_overlap=settings.SPLIT_OVERLAP_SIZE,
)
mime = magic.Magic(mime=True)

file_reader = LayoutPDFReader(settings.LLM_SHERPA_URI)
Metadata = dict[str, Any] | None


class Chunk(msgspec.Struct):
    file_title: str
    content: str
    metadata: Metadata = {}


async def llmsherpa_split_file(
    file_bytes: bytes, file_name: str, metadata: Metadata = None
) -> Result[list[Chunk], CustomError]:
    """
    Function to split file from bytes into a list of Chunk.

    Args:
        file_bytes (bytes): file content
        file_name (str): the file name as string
        metadata (Metadata, optional): metadata to be passed with the file to be sent to the
        hybrid vector store. Defaults to None.

    Returns:
        Result[list[Chunk], CustomError]: Ok of list of chunks or Error

    """
    try:
        await logger.ainfo(event="File Splitting", step="llm sherpa", status="Started")
        doc = await asyncio.to_thread(file_reader.read_pdf, file_name, file_bytes)
        chunks = [
            Chunk(
                file_title=file_name, content=chunk.to_context_text(), metadata=metadata
            )
            for chunk in doc.chunks()
        ]
        await logger.ainfo(
            event="File Splitting", step="llm sherpa", status="Succeeded"
        )
        return Ok(chunks)
    except Exception as e:
        await logger.aexception(
            event="File Splitting", step="llm sherpa", status="Failed", exception=e
        )

        return CustomError("FileSplitError", exc=e)


async def split_by_token_size(
    text: str, file_name: str, metadata: Metadata = None
) -> list[Chunk]:
    """
    split a file by token size.

    Args:
        text (str): file content as string
        file_name (str): the file name
        metadata (Metadata, optional): metadata to be passed with the file to be sent to the
        hybrid vector store. Defaults to None.

    Returns:
        Result[list[Chunk], CustomError]: Ok of list of chunks or Error

    """
    await logger.ainfo(event="File Splitting", step="by token size", status="Started")
    texts = await asyncio.to_thread(text_splitter.split_text, text)
    chunks = [
        Chunk(file_title=file_name, content=chunk, metadata=metadata) for chunk in texts
    ]
    await logger.ainfo(event="File Splitting", step="by token size", status="Succeeded")

    return chunks


def calculate_tokens(chunk: str) -> Result[int, CustomError]:
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return Ok(len(encoding.encode(chunk)))
    except Exception as e:
        return CustomError("Chunk size calculation", exc=e)


async def split_file(
    file_bytes: bytes, file_name: str, metadata: Metadata = None
) -> Result[list[Chunk], CustomError]:
    list_chunks: list[Chunk] = []
    try:
        async with asyncio.timeout(settings.LLMSHERPA_TIMEOUT):
            chunks = await llmsherpa_split_file(
                file_bytes=file_bytes, file_name=file_name, metadata=metadata
            )
        if chunks.is_err():
            msg = "llm sherpa failed"
            raise ValueError(msg)  # noqa: TRY301
        for chunk in chunks.unwrap():
            if calculate_tokens(chunk.content).unwrap() > settings.SPLIT_CHUNK_SIZE:
                split_chunks = await split_by_token_size(
                    text=chunk.content, file_name=file_name, metadata=metadata
                )
            else:
                split_chunks = [chunk]
            list_chunks.extend(split_chunks)
        return Ok(list_chunks)
    except (TimeoutError, ValueError) as e:
        await logger.aexception(
            event="File Splitting", step="llm sherpa", status="Failed", exception=e
        )
        text = read_pdf(file_stream=file_bytes)
        if text.is_err():
            return text
        list_split_chunks = await split_by_token_size(
            text=text.unwrap(), file_name=file_name, metadata=metadata
        )

    return Ok(list_split_chunks)


async def split_mavarick_file() -> list[str] | None:
    file_name = "Scope3_Calculation_Guidance_0.pdf"
    async with aiofiles.open(os.path.join("app", "files", file_name), mode="rb") as f:
        contents = await f.read()

    chunks = await split_file(file_bytes=contents, file_name=file_name, metadata=None)
    if chunks.is_err():
        await logger.aerror(step="by token size", status="Started")
        return None
    docs = chunks_to_texts(chunks.unwrap())
    return docs


def chunks_to_texts(chunks: list[Chunk]) -> list[str]:
    return [chunk.content for chunk in chunks]
