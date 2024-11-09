"""General app utils."""

import httpx
from result import Ok, Result
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from app.core.config import settings
from app.core.custom_error import AppError
from app.telemetry import get_logger, get_tracer

async_client = httpx.AsyncClient()
tracer = get_tracer("utils")


@tracer.start_as_current_span("ollama.model.pull")
async def pull_model(
    model_name: str,
    client: httpx.AsyncClient = async_client,
    server_url: str = settings.OLLAMA_HOST.unicode_string(),
) -> Result[None, AppError]:
    """
    Pulls a specified model from the Ollama server.

    Args:
        model_name (str): The name of the model to pull.
        client(httpx.AsyncClient): async http client to use, defaults to global client.
        server_url (str): The base URL of the Ollama server. Defaults to env defined OLLAMA_HOST".

    Returns:
        Optional[dict]: JSON response from the server if successful; None otherwise.

    """
    logger = get_logger("ollama")
    url = f"{server_url}api/pull"
    payload = {"model": model_name}

    try:
        async with client.stream(
            method="POST", url=url, json=payload, timeout=None
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            with (
                logging_redirect_tqdm(),
                tqdm(
                    total=total, unit="B", unit_scale=True, desc=f"pulling {model_name}"
                ) as pbar,
            ):
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    pbar.update(len(chunk))

            logger.info("Model pulled", model_name=model_name)
        return Ok(None)
    except httpx.HTTPError as e:
        logger.exception("Failed to pull model", model_name=model_name, exc_info=e)
        return AppError("Failed to load model", exc=e)
