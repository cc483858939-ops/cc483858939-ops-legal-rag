from __future__ import annotations

from legal_rag.fusion import FusionConfig, weighted_rrf
from legal_rag.schema import DocumentChunk


def chunk(chunk_id: str, citation: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=chunk_id,
        doc_type="statute",
        title=citation,
        citation=citation,
        text=f"{citation} text",
    )


def test_weighted_rrf_deduplicates_and_combines_scores() -> None:
    first = chunk("a", "5 U.S.C. § 553")
    second = chunk("b", "17 U.S.C. § 107")

    hits = weighted_rrf(
        dense_results=[(first, 0.9), (second, 0.4)],
        bm25_results=[(second, 12.0), (first, 3.0)],
        config=FusionConfig(dense_weight=0.55, bm25_weight=0.45, rrf_k=60),
        limit=10,
    )

    assert {hit.chunk_id for hit in hits} == {"a", "b"}
    assert all(hit.fusion_score > 0 for hit in hits)
    assert hits[0].rank_explanation["dense_weight"] == 0.55


def test_weighted_rrf_can_bias_bm25() -> None:
    first = chunk("a", "dense")
    second = chunk("b", "bm25")

    hits = weighted_rrf(
        dense_results=[(first, 0.99), (second, 0.01)],
        bm25_results=[(second, 99.0), (first, 0.1)],
        config=FusionConfig(dense_weight=0.1, bm25_weight=0.9, rrf_k=1),
        limit=2,
    )

    assert hits[0].chunk_id == "b"
