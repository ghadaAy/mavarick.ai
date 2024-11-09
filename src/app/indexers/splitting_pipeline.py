"""Text splitter module that contains different methods to split a file's content."""

from __future__ import annotations

from result import Result

from app.core.custom_error import AppError
from app.indexers.readers import FitzFileReader, SherpaReader
from app.indexers.splitters import TokenSizeSplitting
from app.telemetry import get_logger

logger = get_logger()
type Chunk = str


async def split_file(file_path: str) -> Result[list[Chunk], AppError]:
    """
    Splits a file into standard sized chunks.

    Args:
        file_path (str): path to the target file.

    Returns:
        Result[list[Chunk], CustomError]: List of chunks or an error.

    """
    result = await SherpaReader(file_path).execute()
    if result.is_err():
        # default to a simple file read using fitz
        result = await FitzFileReader(file_path).execute()
    if result.is_err():
        # both approaches failed
        return result.unwrap_err()
    return await TokenSizeSplitting(result.unwrap()).execute()
