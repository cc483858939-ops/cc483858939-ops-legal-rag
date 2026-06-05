from __future__ import annotations

import pytest

from personal_rag.config import Settings
from personal_rag.factory import build_embedder, build_retriever
from personal_rag.metadata_filter import MetadataFilters
from personal_rag.schema import DocumentChunk, RetrievalHit
from personal_rag.store import InMemoryPersonalStore


def make_store() -> InMemoryPersonalStore:
    store = InMemoryPersonalStore()
    store.upsert_chunks(
        [
            DocumentChunk(
                chunk_id="song",
                source_id="personal-test-note",
                doc_type="note",
                title="Personal Test Note",
                source_ref="Personal Test Note v1",
                text="炫神最喜欢的歌是打火机，因为歌词里有吉隆坡的天气。",
            ),
            DocumentChunk(
                chunk_id="memory",
                source_id="memory-note",
                doc_type="note",
                title="Memory Note",
                source_ref="Memory Note v1",
                text="长期事实记忆适合保存稳定偏好和个人事实。",
            ),
        ]
    )
    return store


def test_hybrid_retrieval_finds_personal_note() -> None:
    retriever = build_retriever(Settings(reranker_model=None), make_store())

    hits = retriever.retrieve("打火机是什么", top_k=3, mode="hybrid")

    assert hits
    assert hits[0].source_ref == "Personal Test Note v1"


def test_retrieval_respects_metadata_filters() -> None:
    retriever = build_retriever(Settings(reranker_model=None), make_store())

    hits = retriever.retrieve(
        "记忆",
        top_k=3,
        mode="hybrid",
        filters=MetadataFilters(values={"source_ref": ("Memory Note v1",)}),
    )

    assert hits
    assert {hit.source_ref for hit in hits} == {"Memory Note v1"}


def test_source_ref_boost_records_explanation() -> None:
    from personal_rag.retriever import apply_query_source_ref_boost

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Other",
        source_ref="ShowMaker Note",
        text="Some text",
        fusion_score=0.1,
    )

    boosted = apply_query_source_ref_boost("ShowMaker", [hit])[0]

    assert boosted.rank_explanation["source_ref_overlap"] == ["showmaker"]
    assert boosted.rank_explanation["source_ref_boost"] > 0


def test_build_embedder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_BACKEND"):
        build_embedder(Settings(embedding_backend="unknown"), offline=False)
