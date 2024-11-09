import asyncio  # noqa: D100
from concurrent.futures import ThreadPoolExecutor

import fitz  # type: ignore[import-untyped]
from llmsherpa.readers import LayoutPDFReader  # type: ignore[import-untyped]
from result import Ok, Result
from structlog import get_logger

from app.core.config import settings
from app.core.custom_error import AppError
from app.indexers.abc import IFileReader


class FitzFileReader(IFileReader):
    """A strategy that uses fitz to read pdf files."""

    def __init__(self, file_path: str) -> None:
        """
        Args:
            file_path (str): path to the pdf file.

        """
        super().__init__(file_path)
        self.logger = get_logger("Fitz reader")

    async def execute(self) -> Result[list[str], AppError]:
        """
        Reads and extracts text from a PDF file.

        Args:
            file_path (str): Input PDF file stream.

        Returns:
            Result[ClientDocument, CustomError]: Extracted document text or an error.

        """
        return await asyncio.to_thread(self._read)

    def _read(self) -> Result[list[str], AppError]:
        try:
            with fitz.open(filename=self.file_path) as document:

                def extract_page_content(page_number: int) -> str:
                    self.logger.info(
                        f"reading file page {page_number}", status="Succeeded"
                    )
                    return document[page_number].get_text()  # pyright: ignore[reportAttributeAccessIssue]

                with ThreadPoolExecutor(5) as executor:
                    futures = [
                        executor.submit(extract_page_content, i)
                        for i in range(len(document))
                    ]

                texts = [future.result() for future in futures]
                if not texts:
                    self.logger.error(
                        "reading file", status="Failed", reason="No text found"
                    )
                    return AppError(message="Error reading file")
                self.logger.info("reading file", status="Succeeded")
                # we don't care about pages
                return Ok(["".join(texts)])
        except Exception as e:
            self.logger.error("reading file", status="Failed")
            return AppError(message="File could not be read", exc=e)


class SherpaReader(IFileReader):
    """
    strategy to read and split a file using LLMSherpa.

    Notes:
      unfortunately sherpa couples both reading and splitting
      and as we have a downstream splitting normalizer we treat it as reader
      for our purposes.

    """

    def __init__(self, file_path: str, reader: LayoutPDFReader | None = None) -> None:  # type: ignore[no-any-unimported]
        """
        Args:
            file_path (str): the pdf file
            reader (LayoutPDFReader | None, optional): sherpa compatible reader. Defaults to None.
            constructed if not provided.

        """
        super().__init__(file_path)
        self.reader = reader or LayoutPDFReader(settings.LLM_SHERPA_URI)
        self.logger = get_logger("sherpa splitter")

    async def execute(self) -> Result[list[str], AppError]:
        """
        Function to split file from bytes into a list of Chunk.

        Args:
            file_name (str): the file name as string
        Returns:
            Result[list[str], CustomError]: Ok of list of chunks or Error

        """
        try:
            async with asyncio.timeout(settings.LLMSHERPA_TIMEOUT):
                await self.logger.ainfo(
                    event="File Splitting", step="llm sherpa", status="Started"
                )
                doc = await asyncio.to_thread(self.reader.read_pdf, self.file_path)
                chunks: list[str] = [chunk.to_context_text() for chunk in doc.chunks()]
                await self.logger.ainfo(
                    event="File Splitting", step="llm sherpa", status="Succeeded"
                )
                return Ok(chunks)
        except TimeoutError as e:
            await self.logger.aexception(
                event="File Splitting",
                step="llm sherpa",
                status="timed_out",
                exception=e,
            )

            return AppError("SherpaTimeOut", exc=e)
        except Exception as e:
            await self.logger.aexception(
                event="File Splitting", step="llm sherpa", status="Failed", exception=e
            )
            return AppError("FileSplitError", exc=e)
