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

    def upsert(self, *, collection_name, points):
        self.collection_name = collection_name
        self.upserted_points = points

    def query_points(self, **kwargs):
        self.last_query_filter = kwargs.get("query_filter")
        return type("Response", (), {"points": []})()


def test_qdrant_adapter_writes_dense_and_bm25_named_vectors() -> None:
    store = object.__new__(QdrantLegalStore)
    store.collection_name = "legal_test"
    store.client = FakeQdrantClient()
    store.embedder = HashingEmbedder(dimensions=16)
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
    assert store.client.collection_name == "legal_test"
    assert "dense" in point.vector
    assert "bm25" in point.vector
    assert point.payload["citation"] == "5 U.S.C. § 553"


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
