"""Milvus vector db that supports hybrid search."""

from enum import StrEnum
from itertools import chain
from typing import Self

from langchain_core.documents.base import Document as LangchainDocument
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_ollama import OllamaEmbeddings
from opentelemetry import trace
from pymilvus import (  # type:ignore[import-untyped]
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility,
)

from app.core.config import settings
from app.telemetry import get_tracer

tracer = get_tracer("milvus")


class FieldNames(StrEnum):
    """tracing fields."""

    PK_FIELD = "doc_id"
    DENSE_FIELD = "dense_vector"
    SPARSE_FIELD = "sparse_vector"
    TEXT_FIELD = "text"
    DEFAULT_CONNECTION = "default"


class MilvusHybrid:
    """Class for managing hybrid search operations with Milvus."""

    def __init__(self, dense_embedding_func: OllamaEmbeddings | None = None) -> None:
        """Initializes MilvusHybrid with dense embedding function."""
        self.dense_embedding_func = dense_embedding_func or OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDINGS_MODEL_NAME,
            base_url=settings.OLLAMA_HOST.unicode_string(),
        )

    @tracer.start_as_current_span("MilvusHybrid.fit_bm25")
    def fit_bm25(self, texts: list[str]) -> Self:
        """Fits BM25 sparse embedding on a list of texts."""
        self.texts = texts
        self.sparse_embedding_func = BM25SparseEmbedding(corpus=texts)
        return self

    @tracer.start_as_current_span("MilvusHybrid.field_schema")
    def field_schema(self) -> Self:
        """Defines schema fields for Milvus collection."""
        self.fields = [
            FieldSchema(
                name=FieldNames.PK_FIELD,
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100,
            ),
            FieldSchema(
                name=FieldNames.DENSE_FIELD,
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.EMBEDDINGS_DIMENSION,
            ),
            FieldSchema(
                name=FieldNames.SPARSE_FIELD, dtype=DataType.SPARSE_FLOAT_VECTOR
            ),
            FieldSchema(
                name=FieldNames.TEXT_FIELD, dtype=DataType.VARCHAR, max_length=65_535
            ),
        ]
        return self

    @tracer.start_as_current_span("MilvusHybrid.set_collection")
    def set_collection(self) -> Self:
        """Creates or retrieves a Milvus collection."""
        if not utility.has_collection(settings.COLLECTION_NAME):
            schema = CollectionSchema(fields=self.fields, enable_dynamic_field=False)
            self.collection = Collection(
                name=settings.COLLECTION_NAME, schema=schema, consistency_level="Strong"
            )
        else:
            self.collection = Collection(settings.COLLECTION_NAME)
        return self

    @tracer.start_as_current_span("MilvusHybrid.set_indexes")
    def set_indexes(self) -> Self:
        """Sets indexes on the dense and sparse fields in the Milvus collection."""
        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        self.collection.create_index(FieldNames.DENSE_FIELD, dense_index)
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        self.collection.create_index(FieldNames.SPARSE_FIELD, sparse_index)
        self.collection.flush()
        return self

    @tracer.start_as_current_span("MilvusHybrid.connect")
    def connect(self) -> Self:
        """Connects to the Milvus server."""
        connections.connect(uri=settings.MILVUS_CONNECTION_URL.unicode_string())
        return self

    @tracer.start_as_current_span("MilvusHybrid.insert")
    def insert(self) -> Self:
        """Inserts documents into the Milvus collection."""
        entities = []
        total = len(self.texts)
        total_digits = len(str(total))
        for idx, text in enumerate(self.texts, start=1):
            with tracer.start_span("MilvusHybrid.insert.embeddings.chunk") as span:
                entity = {
                    FieldNames.DENSE_FIELD: self.dense_embedding_func.embed_documents(
                        [text]
                    )[0],
                    FieldNames.SPARSE_FIELD: self.sparse_embedding_func.embed_documents(
                        [text]
                    )[0],
                    FieldNames.TEXT_FIELD: text,
                }
                entities.append(entity)
                span.add_event(
                    "MilvusHybrid.insert.embeddings.created",
                    attributes={"progress": f"{str(idx).zfill(total_digits)}/{total}"},
                )
        with trace.get_current_span() as span:
            self.collection.insert(entities)
            span.add_event("MilvusHybrid.insert.embeddings.inserted")
            self.collection.load()
            span.add_event("MilvusHybrid.insert.embeddings.loaded")
        return self

    @tracer.start_as_current_span("MilvusHybrid.set_retriever")
    def set_retriever(self) -> None:
        """Sets up the hybrid search retriever for the collection."""
        sparse_search_params = {"metric_type": "IP"}
        dense_search_params = {"metric_type": "IP", "params": {}}
        self.retriever = MilvusCollectionHybridSearchRetriever(
            collection=self.collection,
            rerank=WeightedRanker(0.5, 0.5),
            anns_fields=[FieldNames.DENSE_FIELD, FieldNames.SPARSE_FIELD],
            field_embeddings=[self.dense_embedding_func, self.sparse_embedding_func],
            field_search_params=[dense_search_params, sparse_search_params],
            top_k=settings.RAG_TOP_K,
            text_field=FieldNames.TEXT_FIELD,
        )

    @tracer.start_as_current_span("MilvusHybrid.disconnect")
    def disconnect(self) -> None:
        """Disconnects from the Milvus server."""
        connections.disconnect(FieldNames.DEFAULT_CONNECTION)

    @tracer.start_as_current_span("MilvusHybrid.retrieve")
    async def retrieve(self, query: str) -> list[LangchainDocument]:
        """Retrieves relevant documents for a given query."""
        return await self.retriever.aget_relevant_documents(query=query)

    @tracer.start_as_current_span("MilvusHybrid.batch_retrieve")
    async def batch_retrieve(self, queries: list[str]) -> list[LangchainDocument]:
        """Retrieves relevant documents for a list of queries."""
        nested_docs = await self.retriever.abatch(queries)
        filter_ = set()
        result: list[LangchainDocument] = []
        for doc in chain.from_iterable(nested_docs):
            if doc.page_content in filter_:
                continue
            filter_.add(doc.page_content)
            result.append(doc)

        return result


milvus = MilvusHybrid()
