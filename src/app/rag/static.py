"""The llm model that will be used throughout the project."""

from langchain_ollama.llms import OllamaLLM

from app.core.config import settings

llm = OllamaLLM(
    model=settings.OLLAMA_LLM_MODEL,
    num_thread=4,
    top_p=0.43,
    temperature=0,
    base_url=settings.OLLAMA_HOST.unicode_string(),
)
