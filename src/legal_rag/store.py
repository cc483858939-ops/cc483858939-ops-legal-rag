from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from legal_rag.bm25 import BM25Index
from legal_rag.embeddings import DenseEmbedder, HashingEmbedder, cosine_similarity
from legal_rag.metadata_filter import MetadataFilters, build_qdrant_filter, matches_metadata_filters
from legal_rag.schema import DocumentChunk


class LegalVectorStore(Protocol):
    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Index chunks."""

    def dense_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Run dense retrieval."""

    def bm25_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Run sparse BM25 retrieval."""


@dataclass
class InMemoryLegalStore:
    embedder: DenseEmbedder = field(default_factory=HashingEmbedder)
    chunks: list[DocumentChunk] = field(default_factory=list)
    vectors: dict[str, list[float]] = field(default_factory=dict)
    bm25: BM25Index = field(default_factory=BM25Index)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        embeddings = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
        self.vectors = {
            chunk.chunk_id: embedding
            for chunk, embedding in zip(self.chunks, embeddings, strict=True)
        }
        self.bm25.fit(self.chunks)

    def dense_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        query_vector = self.embedder.embed_query(query)
        chunks = [chunk for chunk in self.chunks if matches_metadata_filters(chunk, filters)]
        scored = [
            (chunk, cosine_similarity(query_vector, self.vectors.get(chunk.chunk_id, [])))
            for chunk in chunks
        ]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def bm25_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        if filters is None or filters.is_empty:
            return self.bm25.search(query, top_k=top_k)
        filtered_ids = {
            chunk.chunk_id for chunk in self.chunks if matches_metadata_filters(chunk, filters)
        }
        return [
            (chunk, score)
            for chunk, score in self.bm25.search(query, top_k=max(top_k, len(self.chunks)))
            if chunk.chunk_id in filtered_ids
        ][:top_k]


class QdrantLegalStore:
    """Qdrant store with dense and Qdrant/bm25 sparse named vectors.

    This implementation is intentionally thin. It keeps the production path wired to Qdrant while
    preserving an in-memory store for deterministic tests.
    """

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        embedder: DenseEmbedder,
        sparse_model: str = "Qdrant/bm25",
        api_key: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedder = embedder
        self.sparse_model = sparse_model
        self.models = models
        self._sparse_embedder = None

    def ensure_collection(self) -> None:
        models = self.models
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.embedder.dimensions,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        from qdrant_client.http import models

        self.ensure_collection()
        dense_vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        sparse_vectors = self._embed_sparse([chunk.text for chunk in chunks])

        points: list[models.PointStruct] = []
        for chunk, dense_vector, sparse_vector in zip(
            chunks,
            dense_vectors,
            sparse_vectors,
            strict=True,
        ):
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector={
                        "dense": dense_vector,
                        "bm25": models.SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                    },
                    payload=chunk.model_dump(mode="json"),
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def dense_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        query_vector = self.embedder.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            query_filter=build_qdrant_filter(filters),
            limit=top_k,
            with_payload=True,
        )
        points = getattr(results, "points", results)
        return [
            (DocumentChunk.model_validate(item.payload), float(item.score))
            for item in points
            if item.payload
        ]

    def bm25_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        from qdrant_client.http import models

        sparse_query = self._embed_sparse([query])[0]
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=sparse_query.indices.tolist(),
                values=sparse_query.values.tolist(),
            ),
            using="bm25",
            query_filter=build_qdrant_filter(filters),
            limit=top_k,
            with_payload=True,
        )
        points = getattr(results, "points", results)
        return [
            (DocumentChunk.model_validate(item.payload), float(item.score))
            for item in points
            if item.payload
        ]

    def _embed_sparse(self, texts: list[str]):
        if self._sparse_embedder is None:
            from fastembed import SparseTextEmbedding

            self._sparse_embedder = SparseTextEmbedding(model_name=self.sparse_model)
        return list(self._sparse_embedder.embed(texts))
