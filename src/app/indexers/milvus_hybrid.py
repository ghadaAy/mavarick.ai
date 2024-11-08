from typing import Self

from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_ollama import OllamaEmbeddings
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility,
)

from app.core.config import settings


class MilvusHybrid:
    pk_field = "doc_id"
    dense_field = "dense_vector"
    sparse_field = "sparse_vector"
    text_field = "text"
    def __init__(self,
                 dense_embedding_func = OllamaEmbeddings(model=settings.OLLAMA_EMBEDDINGS_MODEL_NAME)
                 ) -> None:
        self.dense_embedding_func=dense_embedding_func

    def fit_bm25(self, texts:list[str]) -> Self:
        self.texts=texts
        self.sparse_embedding_func= BM25SparseEmbedding(corpus=texts)
        return self

    def field_schema(self) -> Self:
        self.fields = [
            FieldSchema(
                name=self.pk_field,
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100,
            ),
            FieldSchema(name=self.dense_field, dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDINGS_DIMENSION
                        ),
            FieldSchema(name=self.sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=65_535),
        ]
        return self

    def set_collection(self) -> Self:
        if not utility.has_collection(settings.COLLECTION_NAME):

            schema = CollectionSchema(fields=self.fields, enable_dynamic_field=False)
            self.collection= Collection(
                name=settings.COLLECTION_NAME, schema=schema, consistency_level="Strong"
            )
        else:
            self.collection=Collection(settings.COLLECTION_NAME)
        return self
    def set_indexes(self) -> Self:
        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        self.collection.create_index("dense_vector", dense_index)
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        self.collection.create_index("sparse_vector", sparse_index)
        self.collection.flush()
        return self
    def connect(self) -> Self:
        connections.connect(uri=settings.MILVUS_CONNECTION_URL)
        return self

    def insert(self) -> Self:
        entities = []
        for text in self.texts:
            entity = {
                self.dense_field: self.dense_embedding_func.embed_documents([text])[0],
                self.sparse_field: self.sparse_embedding_func.embed_documents([text])[0],
                self.text_field: text,
            }
            entities.append(entity)
        self.collection.insert(entities)
        self.collection.load()
        return self

    def set_retriever(self) -> None:
        sparse_search_params = {"metric_type": "IP"}
        dense_search_params = {"metric_type": "IP", "params": {}}
        self.retriever = MilvusCollectionHybridSearchRetriever(
            collection=self.collection,
            rerank=WeightedRanker(0.5, 0.5),
            anns_fields=[self.dense_field, self.sparse_field],
            field_embeddings=[self.dense_embedding_func, self.sparse_embedding_func],
            field_search_params=[dense_search_params, sparse_search_params],
            top_k=settings.RAG_TOP_K,
            text_field=self.text_field,
        )

hybrid = MilvusHybrid()
