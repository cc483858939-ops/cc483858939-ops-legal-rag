from __future__ import annotations

from personal_rag.fusion import FusionConfig, weighted_rrf
from personal_rag.schema import DocumentChunk


def chunk(chunk_id: str, source_ref: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_ref,
        doc_type="note",
        title=source_ref,
        source_ref=source_ref,
        text=f"{source_ref} text",
    )


def test_weighted_rrf_merges_dense_and_bm25_results() -> None:
    first = chunk("1", "Personal Test Note v1")
    second = chunk("2", "Memory Note v1")

    hits = weighted_rrf(
        [(first, 0.9), (second, 0.5)],
        [(second, 2.0), (first, 1.0)],
        config=FusionConfig(dense_weight=0.5, bm25_weight=0.5, rrf_k=60),
        limit=2,
    )

    assert {hit.source_ref for hit in hits} == {"Personal Test Note v1", "Memory Note v1"}
    assert all(hit.fusion_score > 0 for hit in hits)
