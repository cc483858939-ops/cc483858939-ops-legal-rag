from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from legal_rag.embeddings import HashingEmbedder
from legal_rag.metadata_filter import MetadataFilters
from legal_rag.schema import DocumentChunk
from legal_rag.store import QdrantLegalStore


@dataclass
class FakeSparseEmbedding:
    indices: np.ndarray
    values: np.ndarray


class FakeQdrantClient:
    def __init__(self) -> None:
        self.upserted_points = []
        self.last_query_filter = None
        self.scrolled = False

    def upsert(self, *, collection_name, points):
        self.collection_name = collection_name
        self.upserted_points = points

    def query_points(self, **kwargs):
        self.last_query_filter = kwargs.get("query_filter")
        return type("Response", (), {"points": []})()

    def scroll(self, **kwargs):
        self.scrolled = True
        return [], None


def test_qdrant_adapter_defaults_to_dense_only_with_local_bm25() -> None:
    store = object.__new__(QdrantLegalStore)
    store.collection_name = "legal_test"
    store.client = FakeQdrantClient()
    store.embedder = HashingEmbedder(dimensions=16)
    store.sparse_backend = "bm25_jieba"
    store.bm25 = type("FakeBM25", (), {"fit": lambda self, chunks: None})()
    store._bm25_loaded_from_qdrant = False
    store.ensure_collection = lambda: None

    chunk = DocumentChunk(
        chunk_id="chunk-1",
        source_id="usc-5-553",
        doc_type="statute",
        title="Rule Making",
        citation="5 U.S.C. § 553",
        text="Federal Register notice of proposed rule making.",
    )

    store.upsert_chunks([chunk])

    point = store.client.upserted_points[0]
    assert store.client.collection_name == "legal_test"
    assert "dense" in point.vector
    assert "bm25" not in point.vector
    assert point.payload["citation"] == "5 U.S.C. § 553"


def test_qdrant_adapter_writes_sparse_vectors_when_qdrant_bm25_enabled() -> None:
    store = object.__new__(QdrantLegalStore)
    store.collection_name = "legal_test"
    store.client = FakeQdrantClient()
    store.embedder = HashingEmbedder(dimensions=16)
    store.sparse_backend = "qdrant_bm25"
    store.bm25 = type("FakeBM25", (), {"fit": lambda self, chunks: None})()
    store._bm25_loaded_from_qdrant = False
    store.ensure_collection = lambda: None
    store._embed_sparse = lambda texts: [
        FakeSparseEmbedding(indices=np.array([11, 22]), values=np.array([1.5, 2.5]))
        for _ in texts
    ]

    chunk = DocumentChunk(
        chunk_id="chunk-1",
        source_id="usc-5-553",
        doc_type="statute",
        title="Rule Making",
        citation="5 U.S.C. § 553",
        text="Federal Register notice of proposed rule making.",
    )

    store.upsert_chunks([chunk])

    point = store.client.upserted_points[0]
    assert "dense" in point.vector
    assert "bm25" in point.vector


def test_qdrant_adapter_passes_payload_filter_to_dense_search() -> None:
    store = object.__new__(QdrantLegalStore)
    store.collection_name = "legal_test"
    store.client = FakeQdrantClient()
    store.embedder = HashingEmbedder(dimensions=16)

    store.dense_search(
        "Chevron",
        top_k=5,
        filters=MetadataFilters(
            values={"doc_type": ("case",)},
            date_from=date(2020, 1, 1),
        ),
    )

    payload = store.client.last_query_filter.model_dump(mode="json")
    assert payload["must"][0]["key"] == "doc_type"
    assert payload["must"][0]["match"]["value"] == "case"
    assert payload["must"][1]["key"] == "date"
    assert payload["must"][1]["range"]["gte"] == "2020-01-01T00:00:00"


def test_qdrant_local_bm25_rebuilds_from_payload_scroll() -> None:
    chunk = DocumentChunk(
        chunk_id="personal-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="炫神最喜欢的歌是打火机。",
    )

    class ScrollClient(FakeQdrantClient):
        def scroll(self, **kwargs):
            self.scrolled = True
            point = type("Point", (), {"payload": chunk.model_dump(mode="json")})()
            return [point], None

    store = object.__new__(QdrantLegalStore)
    store.collection_name = "legal_test"
    store.client = ScrollClient()
    store.embedder = HashingEmbedder(dimensions=16)
    store.sparse_backend = "bm25_jieba"
    from legal_rag.bm25 import BM25Index

    store.bm25 = BM25Index()
    store._bm25_loaded_from_qdrant = False

    results = store.bm25_search("炫神最喜欢什么歌", top_k=8)

    assert store.client.scrolled is True
    assert results
    assert results[0][0].text == "炫神最喜欢的歌是打火机。"
    assert results[0][1] > 0
