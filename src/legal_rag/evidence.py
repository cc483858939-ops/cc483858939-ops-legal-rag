from __future__ import annotations

from legal_rag.metadata_filter import MetadataFilterDecision, MetadataFilters
from legal_rag.query_rewrite import (
    AcceptedQueryExtension,
    QueryExtensionResult,
    QueryRewriteResult,
    build_query_extensions,
)
from legal_rag.rerank import rerank_hits
from legal_rag.schema import RetrievalHit
from legal_rag.text import normalize_whitespace


def build_evidence_pack(
    query: str,
    mode: str,
    top_k: int,
    hits: list[RetrievalHit],
    *,
    query_rewrite: QueryRewriteResult | None = None,
    query_extensions: QueryExtensionResult | None = None,
    trace_id: str | None = None,
    trace_path: str | None = None,
    trace_retention_count: int | None = None,
    reranker_summary: dict | None = None,
    metadata_filters: MetadataFilterDecision | None = None,
) -> dict:
    rewrite = query_rewrite or QueryRewriteResult.original(query)
    extensions = query_extensions or build_query_extensions(
        rewrite,
        embedder=None,
        min_similarity=0.45,
    )
    return {
        "query": query,
        "retrieval_query": extensions.retrieval_query,
        "query_rewrite": rewrite.as_dict(),
        "query_extensions": extensions.as_dict(),
        "trace_id": trace_id,
        "trace_path": trace_path,
        "trace_retention_count": trace_retention_count,
        "reranker": reranker_summary or {},
        "metadata_filters": metadata_filters.as_dict() if metadata_filters else {},
        "mode": mode,
        "top_k": top_k,
        "hit_count": len(hits),
        "hits": [format_evidence_hit(hit, rank=index + 1) for index, hit in enumerate(hits)],
    }


def retrieve_with_query_extensions(
    retriever,
    accepted_queries: list[AcceptedQueryExtension],
    *,
    top_k: int,
    mode: str,
    trace=None,
    rerank_query: str | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> list[RetrievalHit]:
    hits, _ = retrieve_with_query_extensions_and_summary(
        retriever,
        accepted_queries,
        top_k=top_k,
        mode=mode,
        trace=trace,
        rerank_query=rerank_query,
        metadata_filters=metadata_filters,
    )
    return hits


def retrieve_with_query_extensions_and_summary(
    retriever,
    accepted_queries: list[AcceptedQueryExtension],
    *,
    top_k: int,
    mode: str,
    trace=None,
    rerank_query: str | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> tuple[list[RetrievalHit], dict]:
    if not accepted_queries:
        return [], {
            "enabled": False,
            "skipped": True,
            "reason": "no_accepted_queries",
            "input_count": 0,
            "reranked_count": 0,
        }

    pool = max(top_k, getattr(retriever, "candidate_pool", top_k))
    hits_by_query: list[tuple[AcceptedQueryExtension, list[RetrievalHit]]] = []
    for query in accepted_queries:
        stage = trace.start_stage(
            "per_query_retrieval",
            {
                "query": query.query,
                "source": query.source,
                "similarity": query.similarity,
                "weight": query.weight,
                "top_k": pool,
                "mode": mode,
                "metadata_filters": metadata_filters.as_dict() if metadata_filters else {},
            },
        ) if trace else None
        hits = retriever.retrieve(
            query.query,
            top_k=pool,
            mode=mode,
            filters=metadata_filters,
            apply_reranker=False,
            trace=trace,
            trace_context={
                "extension_query": query.query,
                "extension_source": query.source,
            },
        )
        if trace and stage is not None:
            trace.end_stage(
                stage,
                {
                    "hit_count": len(hits),
                    "hits": trace.hits_summary(hits),
                },
            )
        hits_by_query.append(
            (
                query,
                hits,
            )
        )
    fusion_stage = trace.start_stage(
        "multi_query_fusion",
        {
            "query_count": len(hits_by_query),
            "limit": top_k,
            "rrf_k": get_query_rrf_k(retriever),
            "metadata_filters": metadata_filters.as_dict() if metadata_filters else {},
        },
    ) if trace else None
    merged_hits = merge_query_extension_hits(
        hits_by_query,
        limit=max(top_k, getattr(retriever, "rerank_top_n", top_k)),
        rrf_k=get_query_rrf_k(retriever),
    )
    if trace and fusion_stage is not None:
        trace.end_stage(
            fusion_stage,
            {
                "hit_count": len(merged_hits),
                "hits": trace.hits_summary(merged_hits),
            },
        )

    chosen_rerank_query = rerank_query or accepted_queries[0].query
    rerank_stage = trace.start_stage(
        "cross_encoder_rerank",
        {
            "query": chosen_rerank_query,
            "input_count": len(merged_hits),
            "top_n": getattr(retriever, "rerank_top_n", top_k),
        },
    ) if trace else None
    reranked_hits, reranker_summary = rerank_hits(
        getattr(retriever, "reranker", None),
        chosen_rerank_query,
        merged_hits,
        top_n=getattr(retriever, "rerank_top_n", top_k),
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
                "hits": trace.hits_summary(reranked_hits),
            },
        )
    return reranked_hits[:top_k], reranker_summary


def merge_query_extension_hits(
    hits_by_query: list[tuple[AcceptedQueryExtension, list[RetrievalHit]]],
    *,
    limit: int,
    rrf_k: int,
) -> list[RetrievalHit]:
    merged: dict[str, RetrievalHit] = {}
    fusion_scores: dict[str, float] = {}
    dense_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}
    rerank_scores: dict[str, float | None] = {}
    contributions: dict[str, list[dict]] = {}

    for query in hits_by_query:
        extension, hits = query
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk_id
            merged.setdefault(chunk_id, hit)
            dense_scores[chunk_id] = max(dense_scores.get(chunk_id, 0.0), hit.dense_score)
            bm25_scores[chunk_id] = max(bm25_scores.get(chunk_id, 0.0), hit.bm25_score)
            if hit.rerank_score is not None:
                current_rerank = rerank_scores.get(chunk_id)
                rerank_scores[chunk_id] = (
                    hit.rerank_score
                    if current_rerank is None
                    else max(current_rerank, hit.rerank_score)
                )
            query_score = extension.weight / (rrf_k + rank)
            fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0.0) + query_score
            contributions.setdefault(chunk_id, []).append(
                {
                    "query": extension.query,
                    "source": extension.source,
                    "rank": rank,
                    "similarity": extension.similarity,
                    "weight": extension.weight,
                    "query_rrf_score": query_score,
                    "hit_final_score": hit.final_score,
                }
            )

    output: list[RetrievalHit] = []
    for chunk_id, hit in merged.items():
        output.append(
            hit.model_copy(
                update={
                    "dense_score": dense_scores.get(chunk_id, 0.0),
                    "bm25_score": bm25_scores.get(chunk_id, 0.0),
                    "fusion_score": fusion_scores.get(chunk_id, 0.0),
                    "rerank_score": None,
                    "rank_explanation": {
                        **hit.rank_explanation,
                        "query_fusion": {
                            "rrf_k": rrf_k,
                            "query_count": len(hits_by_query),
                            "max_rerank_score": rerank_scores.get(chunk_id),
                        },
                        "query_contributions": contributions.get(chunk_id, []),
                    },
                }
            )
        )

    output.sort(key=lambda hit: (hit.final_score, hit.dense_score, hit.bm25_score), reverse=True)
    return output[:limit]


def get_query_rrf_k(retriever) -> int:
    fusion_config = getattr(retriever, "fusion_config", None)
    return int(getattr(fusion_config, "rrf_k", 60))


def select_rerank_query(
    rewrite: QueryRewriteResult,
    accepted_queries: list[AcceptedQueryExtension],
) -> str:
    if rewrite.canonical_query:
        return rewrite.canonical_query
    for query in accepted_queries:
        if query.source != "original":
            return query.query
    return rewrite.original_query


def format_evidence_hit(hit: RetrievalHit, *, rank: int) -> dict:
    final_score = hit.final_score
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "source_id": hit.source_id,
        "title": hit.title,
        "citation": hit.citation,
        "doc_type": hit.doc_type,
        "jurisdiction": hit.jurisdiction,
        "court": hit.court,
        "date": hit.date.isoformat() if hit.date else None,
        "section": hit.section,
        "snippet": make_snippet(hit.text),
        "text": hit.text,
        "scores": {
            "dense": hit.dense_score,
            "bm25": hit.bm25_score,
            "fusion": hit.fusion_score,
            "rerank": hit.rerank_score,
            "final": final_score,
        },
        "final_score": final_score,
        "rank_explanation": hit.rank_explanation,
    }


def make_snippet(text: str, *, limit: int = 300) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."
