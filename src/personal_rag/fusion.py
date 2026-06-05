from __future__ import annotations

from dataclasses import dataclass

from personal_rag.schema import DocumentChunk, RetrievalHit


@dataclass(frozen=True)
class FusionConfig:
    dense_weight: float = 0.55
    bm25_weight: float = 0.45
    rrf_k: int = 60


def weighted_rrf(
    dense_results: list[tuple[DocumentChunk, float]],
    bm25_results: list[tuple[DocumentChunk, float]],
    *,
    config: FusionConfig | None = None,
    limit: int,
) -> list[RetrievalHit]:
    config = config or FusionConfig()
    chunks: dict[str, DocumentChunk] = {}
    dense_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}
    dense_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    for rank, (chunk, score) in enumerate(dense_results, start=1):
        chunks[chunk.chunk_id] = chunk
        dense_scores[chunk.chunk_id] = float(score)
        dense_rank[chunk.chunk_id] = rank

    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        chunks[chunk.chunk_id] = chunk
        bm25_scores[chunk.chunk_id] = float(score)
        bm25_rank[chunk.chunk_id] = rank

    hits: list[RetrievalHit] = []
    for chunk_id, chunk in chunks.items():
        fusion_score = 0.0
        if chunk_id in dense_rank:
            fusion_score += config.dense_weight / (config.rrf_k + dense_rank[chunk_id])
        if chunk_id in bm25_rank:
            fusion_score += config.bm25_weight / (config.rrf_k + bm25_rank[chunk_id])
        hits.append(
            RetrievalHit.from_chunk(
                chunk,
                dense_score=dense_scores.get(chunk_id, 0.0),
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                fusion_score=fusion_score,
                rank_explanation={
                    "dense_rank": dense_rank.get(chunk_id),
                    "bm25_rank": bm25_rank.get(chunk_id),
                    "rrf_k": config.rrf_k,
                    "dense_weight": config.dense_weight,
                    "bm25_weight": config.bm25_weight,
                },
            )
        )

    hits.sort(key=lambda hit: (hit.fusion_score, hit.dense_score, hit.bm25_score), reverse=True)
    return hits[:limit]
