"""File loaders module that will contain different types of files loaders in the future. Currently, only contains pdf reader."""
import io
from concurrent.futures import Future, ThreadPoolExecutor

import fitz  # type: ignore[import-untyped] # PyMuPDF
from result import Ok, Result

from app.core.custom_error import CustomError
from app.core.custom_logger import logger

type ClientDocument = str
type StreamFile = io.BytesIO


def read_pdf(file_stream: StreamFile) -> Result[ClientDocument, CustomError]:
    """
    Reads and extracts text from a PDF file.

    Args:
        file_stream (StreamFile): Input PDF file stream.

    Returns:
        Result[ClientDocument, CustomError]: Extracted document text or an error.

    """
    try:
        with fitz.open(stream=file_stream) as document:

            def extract_page_content(page_number: int) -> str:
                logger.info(f"reading file page {page_number}", status="Succeeded")
                return document[page_number].get_text()  # pyright: ignore[reportAttributeAccessIssue]

            futures: list[Future[str]] = []
            with ThreadPoolExecutor(5) as executor:
                for i in range(len(document)):
                    futures.append(executor.submit(extract_page_content, i))

            texts = [future.result() for future in futures]
            if not texts:
                logger.error("reading file", status="Failed", reason="No text found")
                return CustomError(message="Error reading file")
            logger.info("reading file", status="Succeeded")
            return Ok("".join(texts))
    except Exception as e:
        logger.error("reading file", status="Failed")
        return CustomError(message="File could not be read", exc=e)
