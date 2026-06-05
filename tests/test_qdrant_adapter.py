from __future__ import annotations

from personal_rag.embeddings import HashingEmbedder
from personal_rag.schema import DocumentChunk
from personal_rag.store import QdrantPersonalStore


class FakeClient:
    def __init__(self) -> None:
        self.collection_name = None
        self.points = []

    def upsert(self, *, collection_name, points):
        self.collection_name = collection_name
        self.points = points


class FakeBM25:
    def fit(self, chunks):
        self.chunks = chunks


def test_qdrant_upsert_payload_uses_source_ref(monkeypatch) -> None:
    store = object.__new__(QdrantPersonalStore)
    store.client = FakeClient()
    store.collection_name = "personal_kb_test"
    store.embedder = HashingEmbedder(dimensions=8)
    store.sparse_backend = "bm25_jieba"
    store.bm25 = FakeBM25()
    store._bm25_loaded_from_qdrant = False

    monkeypatch.setattr(store, "ensure_collection", lambda: None)

    chunk = DocumentChunk(
        chunk_id="chunk-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        source_ref="Personal Test Note v1",
        text="炫神最喜欢的歌是打火机。",
    )

    store.upsert_chunks([chunk])

    assert store.client.collection_name == "personal_kb_test"
    payload = store.client.points[0].payload
    assert payload["source_ref"] == "Personal Test Note v1"
