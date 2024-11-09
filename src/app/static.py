"""The models that will be used throughout the project."""
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from app.core.config import settings

ollama_embedding = OllamaEmbeddings(
    model=settings.OLLAMA_EMBEDDINGS_MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
)

llm = OllamaLLM(model=settings.LLM_MODEL, num_thread=4, top_p=0.43, temperature=0, verbose=False)
