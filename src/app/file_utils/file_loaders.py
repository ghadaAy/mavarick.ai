import csv
import io
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum
from io import BytesIO

import fitz  # type: ignore[import-untyped] # PyMuPDF
import pandas as pd
from app.core.custom_error import CustomError
from app.core.custom_logger import logger
from app.static import EnumMixin
from docx import Document
from result import Ok, Result

type FileExtension = str
type ClientDocument = str
type StreamFile = io.BytesIO


class AllowedMimeFileTypes(EnumMixin[str], StrEnum):
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    PDF = "application/pdf"
    TXT = "text/plain"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    XLS = "application/vnd.ms-excel"
    CSV = "text/csv"
    MARKDOWN = "text/markdown"


def is_probably_csv(file_stream: BytesIO, delimiter: str = ",") -> bool:
    """
    Checks if a text/plain BytesIO object is likely a CSV file by analyzing its content.

    Args:
        file_stream (BytesIO): In-memory bytes stream to analyze.
        delimiter (str): Delimiter to use for checking (default is comma).

    Returns:
        bool: True if the content matches a CSV-like structure, False otherwise.

    """
    try:
        # Decode the bytes to a string assuming UTF-8 encoding
        text_content = file_stream.getvalue().decode("utf-8")

        # Split the content into lines and consider a small sample for analysis
        sample_lines = text_content.splitlines()[:5]  # Take the first 5 lines
        if not sample_lines:
            return False
        # Check if all lines have a consistent number of columns
        csv_reader = csv.reader(sample_lines, delimiter=delimiter)

        first_row = next(csv_reader)
        num_columns = len([column for column in first_row if column])
        if num_columns == 0:
            return False  # No columns detected
        for row in csv_reader:
            # Skip empty rows
            if not row:
                continue  # Skip any empty rows

            if len(row) != num_columns:
                return False  # If a row has a different number of columns, it's not consistent

        return True  # All lines have a consistent number of columns
    except (UnicodeDecodeError, StopIteration):
        return False


def read_docx(file_stream: StreamFile) -> Result[ClientDocument, CustomError]:
    try:
        document = Document(file_stream)
        full_text = [paragraph.text for paragraph in document.paragraphs]

        return Ok("\n".join(full_text))

    except Exception as e:
        logger.error("reading file", status="Failed")
        return CustomError(message="File could not be read", exc=e)


def read_pdf(file_stream: StreamFile) -> Result[ClientDocument, CustomError]:
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


def read_txt_and_markdown(
    file_stream: StreamFile,
) -> Result[ClientDocument, CustomError]:
    try:
        text_content = file_stream.getvalue().decode("utf-8")
        return Ok(text_content)
    except Exception as e:
        logger.error("reading file", status="Failed")
        return CustomError(message="File could not be read", exc=e)


def read_xlsx_csv(
    file_stream: StreamFile, mime_type: AllowedMimeFileTypes
) -> Result[ClientDocument, CustomError]:
    try:
        match mime_type:
            case AllowedMimeFileTypes.XLSX | AllowedMimeFileTypes.XLS:
                df = pd.read_excel(file_stream)
            case AllowedMimeFileTypes.CSV:
                df = pd.read_csv(file_stream)
            case _:
                logger.error("unsuppoted file type by read_xlsx_csv", status="Failed")
                return CustomError("unsuppoted file type by read_xlsx_csv")
        data = df.to_dict(orient="records")
        structured_data = "\n".join(
            [
                f"Row {i+1}:\n"
                + "\n".join([f"{key}: {value}" for key, value in row.items()])
                for i, row in enumerate(data)
            ]
        )
        return Ok(structured_data)
    except Exception as e:
        logger.error("reading file", status="Failed", exc_info=e)
        return CustomError(message="File could not be read", exc=e)


def read_file_bytes_as_text(
    file_bytes: bytes, file_type: AllowedMimeFileTypes
) -> Result[ClientDocument, CustomError]:
    file_stream = io.BytesIO(file_bytes)
    match file_type:
        case AllowedMimeFileTypes.PDF:
            return read_pdf(file_stream)
        case AllowedMimeFileTypes.DOCX:
            return read_docx(file_stream)
        case AllowedMimeFileTypes.TXT | AllowedMimeFileTypes.MARKDOWN:
            csv_flag = is_probably_csv(file_stream)
            if csv_flag:
                file_type = AllowedMimeFileTypes.CSV
                return read_xlsx_csv(file_stream, mime_type=file_type)
            return read_txt_and_markdown(file_stream)
        case (
            AllowedMimeFileTypes.CSV
            | AllowedMimeFileTypes.XLSX
            | AllowedMimeFileTypes.XLS
        ):
            return read_xlsx_csv(file_stream, mime_type=file_type)
