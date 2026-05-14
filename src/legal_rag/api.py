from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from legal_rag.answer import ExtractiveAnswerer
from legal_rag.config import ROOT_DIR, load_settings
from legal_rag.evidence import (
    build_evidence_pack,
    retrieve_with_query_extensions_and_summary,
    select_rerank_query,
)
from legal_rag.factory import build_retriever, build_rewriter, build_store
from legal_rag.ingest import ingest_manifest
from legal_rag.metadata_filter import normalize_metadata_filters
from legal_rag.observability import TraceCollector
from legal_rag.query_rewrite import build_query_extensions

app = FastAPI(title="legal-rag", version="0.1.0")
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    mode: str = "hybrid"
    filters: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=50)


class EvidenceRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    mode: str = "hybrid"
    filters: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def runtime():
    settings = load_settings()
    store = build_store(settings)
    retriever = build_retriever(settings, store)
    rewriter = build_rewriter(settings)
    answerer = ExtractiveAnswerer()
    return settings, store, retriever, answerer, rewriter


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    settings = load_settings()
    return {"status": "ok", "collection": settings.qdrant_collection}


@app.post("/ingest")
def ingest(manifest: str = str(ROOT_DIR / "configs" / "legal_sources.yml")) -> dict[str, int]:
    _, store, _, _, _ = runtime()
    chunks = ingest_manifest(Path(manifest), store)
    return {"chunks": len(chunks)}


@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    settings, _, retriever, _, _ = runtime()
    metadata_filters = normalize_metadata_filters(
        explicit_filters=request.filters,
        rewrite_filters={},
        rewrite_confidence=None,
        inferred_enabled=False,
        min_confidence=settings.query_rewrite_filter_min_confidence,
    )
    hits = retriever.retrieve(
        request.query,
        top_k=request.top_k,
        mode=request.mode,
        filters=metadata_filters.effective,
    )
    return {
        "metadata_filters": metadata_filters.as_dict(),
        "hits": [hit.model_dump(mode="json") for hit in hits],
    }


@app.post("/evidence")
def evidence(request: EvidenceRequest):
    settings, store, retriever, _, rewriter = runtime()
    trace = TraceCollector.from_settings(settings)
    request_stage = trace.start_stage(
        "request",
        {"query": request.query, "top_k": request.top_k, "mode": request.mode},
    )
    response = None
    try:
        trace.add_settings(
            {
                "collection": settings.qdrant_collection,
                "candidate_pool": settings.candidate_pool,
                "dense_weight": settings.dense_weight,
                "bm25_weight": settings.bm25_weight,
                "rrf_k": settings.rrf_k,
                "reranker_model": settings.reranker_model,
                "rerank_top_n": settings.rerank_top_n,
                "query_extension_min_similarity": settings.query_extension_min_similarity,
                "inferred_metadata_filters_enabled": settings.inferred_metadata_filters_enabled,
                "query_rewrite_filter_min_confidence": settings.query_rewrite_filter_min_confidence,
            }
        )
        rewrite_stage = trace.start_stage("query_rewrite", {"query": request.query})
        query_rewrite = rewriter.rewrite(request.query)
        trace.end_stage(rewrite_stage, query_rewrite.as_dict())

        filter_stage = trace.start_stage(
            "metadata_filter_normalization",
            {
                "explicit_filters": request.filters,
                "rewrite_filters": query_rewrite.filters,
                "rewrite_confidence": query_rewrite.confidence,
            },
        )
        metadata_filters = normalize_metadata_filters(
            explicit_filters=request.filters,
            rewrite_filters=query_rewrite.filters,
            rewrite_confidence=query_rewrite.confidence,
            inferred_enabled=settings.inferred_metadata_filters_enabled,
            min_confidence=settings.query_rewrite_filter_min_confidence,
        )
        trace.end_stage(filter_stage, metadata_filters.as_dict())

        extension_stage = trace.start_stage(
            "query_extension_filter",
            {
                "min_similarity": settings.query_extension_min_similarity,
                "candidate_count": (
                    (1 if query_rewrite.canonical_query else 0)
                    + len(query_rewrite.search_queries)
                    + len(query_rewrite.aliases)
                ),
            },
        )
        query_extensions = build_query_extensions(
            query_rewrite,
            embedder=getattr(store, "embedder", None),
            min_similarity=settings.query_extension_min_similarity,
        )
        trace.end_stage(extension_stage, query_extensions.as_dict())

        rerank_query = select_rerank_query(query_rewrite, query_extensions.accepted)
        hits, reranker_summary = retrieve_with_query_extensions_and_summary(
            retriever,
            query_extensions.accepted,
            top_k=request.top_k,
            mode=request.mode,
            trace=trace,
            rerank_query=rerank_query,
            metadata_filters=metadata_filters.effective,
        )
        pack_stage = trace.start_stage(
            "evidence_pack",
            {"hit_count": len(hits), "top_k": request.top_k},
        )
        response = build_evidence_pack(
            request.query,
            request.mode,
            request.top_k,
            hits,
            query_rewrite=query_rewrite,
            query_extensions=query_extensions,
            trace_id=trace.trace_id,
            trace_path=trace.public_path,
            trace_retention_count=trace.retention_count,
            reranker_summary=reranker_summary,
            metadata_filters=metadata_filters,
        )
        trace.end_stage(
            pack_stage,
            {
                "hit_count": len(hits),
                "hits": trace.hits_summary(hits),
            },
        )
        trace.end_stage(
            request_stage,
            {
                "hit_count": len(hits),
                "trace_id": trace.trace_id,
                "reranker": reranker_summary,
            },
        )
        return response
    except Exception as exc:
        trace.add_error("evidence", exc)
        trace.end_stage(request_stage, {"trace_id": trace.trace_id}, error=exc)
        raise
    finally:
        trace.write_rolling_json()


@app.post("/query")
def query(request: QueryRequest):
    _, _, retriever, answerer, _ = runtime()
    hits = retriever.retrieve(request.question, top_k=request.top_k)
    answer = answerer.answer(request.question, hits)
    return answer.model_dump(mode="json")
