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
    """Qdrant store with dense vectors plus selectable sparse retrieval.

    This implementation is intentionally thin. It keeps the production path wired to Qdrant while
    preserving an in-memory store for deterministic tests.
    """

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        embedder: DenseEmbedder,
        sparse_backend: str = "bm25_jieba",
        sparse_model: str = "Qdrant/bm25",
        api_key: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedder = embedder
        self.sparse_backend = _normalize_sparse_backend(sparse_backend)
        self.sparse_model = sparse_model
        self.models = models
        self._sparse_embedder = None
        self.bm25 = BM25Index()
        self._bm25_loaded_from_qdrant = False

    def ensure_collection(self) -> None:
        models = self.models
        if self.client.collection_exists(self.collection_name):
            return
        kwargs = {
            "collection_name": self.collection_name,
            "vectors_config": {
                "dense": models.VectorParams(
                    size=self.embedder.dimensions,
                    distance=models.Distance.COSINE,
                )
            },
        }
        if self.sparse_backend == "qdrant_bm25":
            kwargs["sparse_vectors_config"] = {
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        self.client.create_collection(**kwargs)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        from qdrant_client.http import models

        self.ensure_collection()
        self.bm25.fit(chunks)
        self._bm25_loaded_from_qdrant = True
        dense_vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        sparse_vectors = (
            self._embed_sparse([chunk.text for chunk in chunks])
            if self.sparse_backend == "qdrant_bm25"
            else [None for _ in chunks]
        )

        points: list[models.PointStruct] = []
        for chunk, dense_vector, sparse_vector in zip(
            chunks,
            dense_vectors,
            sparse_vectors,
            strict=True,
        ):
            vector = {"dense": dense_vector}
            if sparse_vector is not None:
                vector["bm25"] = models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                )
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector,
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
        if self.sparse_backend == "bm25_jieba":
            self._ensure_local_bm25_index()
            return self._local_bm25_search(query, top_k=top_k, filters=filters)

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

    def _local_bm25_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: MetadataFilters | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        if filters is None or filters.is_empty:
            return self.bm25.search(query, top_k=top_k)
        filtered_ids = {
            chunk.chunk_id for chunk in self.bm25.chunks if matches_metadata_filters(chunk, filters)
        }
        return [
            (chunk, score)
            for chunk, score in self.bm25.search(query, top_k=max(top_k, len(self.bm25.chunks)))
            if chunk.chunk_id in filtered_ids
        ][:top_k]

    def _ensure_local_bm25_index(self) -> None:
        if self._bm25_loaded_from_qdrant:
            return
        self.bm25.fit(self._scroll_chunks())
        self._bm25_loaded_from_qdrant = True

    def _scroll_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if isinstance(result, tuple):
                points, offset = result
            else:
                points = getattr(result, "points", result)
                offset = getattr(result, "next_page_offset", None)
            for point in points:
                payload = getattr(point, "payload", None)
                if payload:
                    chunks.append(DocumentChunk.model_validate(payload))
            if offset is None:
                break
        return chunks

    def _embed_sparse(self, texts: list[str]):
        if self._sparse_embedder is None:
            from fastembed import SparseTextEmbedding

            self._sparse_embedder = SparseTextEmbedding(model_name=self.sparse_model)
        return list(self._sparse_embedder.embed(texts))


def _normalize_sparse_backend(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "bm25_jieba", "local_bm25", "jieba"}:
        return "bm25_jieba"
    if normalized in {"qdrant_bm25", "qdrant", "fastembed"}:
        return "qdrant_bm25"
    raise ValueError(f"Unsupported SPARSE_BACKEND={value!r}")
