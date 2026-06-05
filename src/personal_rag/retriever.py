from __future__ import annotations

from dataclasses import dataclass

from personal_rag.fusion import FusionConfig, weighted_rrf
from personal_rag.metadata_filter import MetadataFilters, qdrant_filter_as_dict
from personal_rag.rerank import Reranker, rerank_hits
from personal_rag.schema import RetrievalHit
from personal_rag.store import PersonalVectorStore
from personal_rag.text import tokenize

SOURCE_BOOST_STOPWORDS = {
    "about",
    "and",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "section",
    "sections",
    "say",
    "says",
    "the",
    "to",
    "under",
    "us",
    "usc",
    "v",
    "what",
    "when",
    "which",
    "who",
    "with",
}


@dataclass
class HybridRetriever:
    store: PersonalVectorStore
    fusion_config: FusionConfig = FusionConfig()
    reranker: Reranker | None = None
    candidate_pool: int = 40
    rerank_top_n: int = 20

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        mode: str = "hybrid",
        filters: MetadataFilters | None = None,
        apply_reranker: bool = True,
        trace=None,
        trace_context: dict | None = None,
    ) -> list[RetrievalHit]:
        pool = max(top_k, self.candidate_pool)
        context = trace_context or {}
        filter_trace = {
            "metadata_filters": filters.as_dict() if filters else {},
            "qdrant_filter": qdrant_filter_as_dict(filters),
        }
        dense_results = []
        bm25_results = []
        if mode in {"dense", "hybrid"}:
            stage = (
                trace.start_stage(
                    "dense_search",
                    {"query": query, "top_k": pool, **filter_trace, **context},
                )
                if trace
                else None
            )
            dense_results = self.store.dense_search(query, top_k=pool, filters=filters)
            if trace and stage is not None:
                trace.end_stage(
                    stage,
                    {
                        "result_count": len(dense_results),
                        "results": chunk_score_summary(
                            dense_results,
                            limit=trace.max_items_per_stage,
                        ),
                    },
                )
        if mode in {"bm25", "hybrid"}:
            stage = (
                trace.start_stage(
                    "bm25_search",
                    {
                        "query": query,
                        "top_k": pool,
                        "sparse_backend": getattr(self.store, "sparse_backend", "in_memory"),
                        **filter_trace,
                        **context,
                    },
                )
                if trace
                else None
            )
            bm25_results = self.store.bm25_search(query, top_k=pool, filters=filters)
            if trace and stage is not None:
                trace.end_stage(
                    stage,
                    {
                        "result_count": len(bm25_results),
                        "results": chunk_score_summary(
                            bm25_results,
                            limit=trace.max_items_per_stage,
                        ),
                    },
                )

        fusion_stage = trace.start_stage(
            "single_query_fusion",
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "pool": pool,
                **filter_trace,
                **context,
            },
        ) if trace else None
        if mode == "dense":
            hits = [
                RetrievalHit.from_chunk(chunk, dense_score=score, fusion_score=score)
                for chunk, score in dense_results
            ][:top_k]
        elif mode == "bm25":
            hits = [
                RetrievalHit.from_chunk(chunk, bm25_score=score, fusion_score=score)
                for chunk, score in bm25_results
            ][:top_k]
        elif mode == "hybrid":
            hits = weighted_rrf(
                dense_results,
                bm25_results,
                config=self.fusion_config,
                limit=pool,
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        if trace and fusion_stage is not None:
            trace.end_stage(
                fusion_stage,
                {
                    "hit_count": len(hits),
                    "hits": trace.hits_summary(hits),
                },
            )

        if apply_reranker:
            rerank_stage = trace.start_stage(
                "cross_encoder_rerank",
                {
                    "query": query,
                    "input_count": len(hits),
                    "top_n": self.rerank_top_n,
                    **context,
                },
            ) if trace else None
            hits, reranker_summary = rerank_hits(
                self.reranker,
                query,
                hits,
                top_n=self.rerank_top_n,
                )
            if trace and rerank_stage is not None:
                if reranker_summary.get("error"):
                    trace.add_error(
                        "cross_encoder_rerank",
                        reranker_summary["error"],
                        stage_index=rerank_stage,
                    )
                trace.end_stage(
                    rerank_stage,
                    {
                        **reranker_summary,
                        "hits": trace.hits_summary(hits),
                    },
                )

        boost_stage = trace.start_stage(
            "source_ref_boost",
            {
                "query": query,
                "input_count": len(hits),
                **context,
            },
        ) if trace else None
        hits = apply_query_source_ref_boost(query, hits)
        if trace and boost_stage is not None:
            boosted_count = sum(1 for hit in hits if hit.rank_explanation.get("source_ref_boost"))
            trace.end_stage(
                boost_stage,
                {
                    "hit_count": len(hits),
                    "boosted_count": boosted_count,
                    "hits": trace.hits_summary(hits),
                },
            )
        return hits[:top_k]


def apply_query_source_ref_boost(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    query_terms = set(tokenize(query)) - SOURCE_BOOST_STOPWORDS
    if not query_terms:
        return hits

    boosted: list[RetrievalHit] = []
    for hit in hits:
        source_terms = set(tokenize(f"{hit.source_ref} {hit.title} {hit.section or ''}"))
        overlap = query_terms & source_terms
        source_ref_boost = 0.0
        if overlap:
            source_ref_boost = 0.02 * len(overlap)
            if hit.section and set(tokenize(hit.section)) & query_terms:
                source_ref_boost += 0.05
        if source_ref_boost:
            boosted.append(
                hit.model_copy(
                    update={
                        "fusion_score": hit.fusion_score + source_ref_boost,
                        "rank_explanation": {
                            **hit.rank_explanation,
                            "source_ref_overlap": sorted(overlap),
                            "source_ref_boost": source_ref_boost,
                        },
                    }
                )
            )
        else:
            boosted.append(hit)

    boosted.sort(key=lambda item: item.final_score, reverse=True)
    return boosted


def chunk_score_summary(results, *, limit: int) -> list[dict]:
    output = []
    for rank, (chunk, score) in enumerate(results[:limit], start=1):
        output.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "title": chunk.title,
                "source_ref": chunk.source_ref,
                "doc_type": chunk.doc_type,
                "score": float(score),
            }
        )
    return output
