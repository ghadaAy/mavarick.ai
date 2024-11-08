"""Main file that runs the fastapi app."""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, status

from app.core.config import settings
from app.file_utils.splitter import split_mavarick_file
from app.indexers.milvus_hybrid import hybrid
from app.rag.flow import answer_user


@asynccontextmanager
async def life_span(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """
    Initializes and tears down resources for the app lifespan.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Indicates the application lifespan context.

    """
    texts=await split_mavarick_file()
    if not texts:
        msg = "No texts were retrieved from split_maverick_file."
        raise ValueError(msg)
    hybrid.connect().fit_bm25(texts).field_schema().set_collection().set_indexes().insert().set_retriever()
    yield
    hybrid.disconnect()

app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url=f"{settings.API_PREFIX}/docs",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=life_span,
)

@app.post(
    "/get_answer",
    status_code=status.HTTP_200_OK,
)
async def answer_user_query(user_query:str) -> str:
    """
    Endpoint to get an answer for a user's query.

    Args:
        user_query (str): The user's question.

    Returns:
        str: The generated answer.

    """
    return await answer_user(user_query)
