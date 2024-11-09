"""Main file that runs the fastapi app."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import nltk  # type:ignore[import-untyped]
import uvicorn
from fastapi import FastAPI, status

from app.core.config import settings
from app.indexers.milvus_hybrid import milvus
from app.indexers.splitting_pipeline import split_file
from app.rag.flow import answer_user
from app.telemetry import get_tracer
from app.telemetry.instrumentation import configure_instrumentation
from app.utils import async_client, pull_model

tracer = get_tracer("startup")


@asynccontextmanager
async def life_span(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """
    Initializes and tears down resources for the app lifespan.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Indicates the application lifespan context.

    """
    await async_client.__aenter__()
    try:
        with tracer.start_span("startup") as span:
            span.add_event("startup.downloading_punkt")
            nltk.data.path = [settings.APP_PATH]
            nltk.download(["punkt_tab", "stopwords"], download_dir=settings.APP_PATH)
            for model in (
                settings.OLLAMA_EMBEDDINGS_MODEL_NAME,
                settings.OLLAMA_LLM_MODEL,
            ):
                span.add_event("startup.ollama.pull", attributes={"model": model})
                result = await pull_model(model)
                assert result.is_ok(), f"Failed to load ollama model {model}, ensure ollama is up and running"
            span.add_event("startup.test_file.splitting")
            chunks = await split_file(file_path=settings.TEST_FILE_PATH)
            assert chunks.is_ok(), f"No texts were retrieved from split_maverick_file. error: {chunks.err()}"
            span.add_event("startup.milvus.setup")
            (
                milvus.connect()
                .fit_bm25(chunks.unwrap())
                .field_schema()
                .set_collection()
                .set_indexes()
                .insert()
                .set_retriever()
            )
            yield
    finally:
        with tracer.start_span("shutdown") as span:
            span.add_event("disconnecting from milvus")
            milvus.disconnect()
            span.add_event("closing httpx connections")
            await async_client.aclose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url=f"{settings.API_PREFIX}/docs",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=life_span,
)
configure_instrumentation(app)


@app.get("/up", status_code=status.HTTP_200_OK, include_in_schema=False)
async def is_up() -> None:
    """
    Endpoint to check if the backend is up and ready.

    Returns:
        str: The generated answer.

    """
    return


@app.post(
    "/get_answer",
    status_code=status.HTTP_200_OK,
)
async def answer_user_query(user_query: str) -> str:
    """
    Endpoint to get an answer for a user's query.

    Args:
        user_query (str): The user's question.

    Returns:
        str: The generated answer.

    """
    return await answer_user(user_query)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")  # noqa: S104
