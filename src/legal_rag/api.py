from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from legal_rag.answer import ExtractiveAnswerer
from legal_rag.config import load_settings
from legal_rag.evidence import (
    build_evidence_pack,
    retrieve_with_query_extensions_and_summary,
    select_rerank_query,
)
from legal_rag.factory import build_retriever, build_rewriter, build_store
from legal_rag.ingest import ingest_manifest
from legal_rag.llm_answer import GeneralKnowledgeAnswerer, LLMAnswerConfig, OpenAICompatibleAnswerer
from legal_rag.metadata_filter import normalize_metadata_filters
from legal_rag.observability import TraceCollector
from legal_rag.query_intent import (
    ConversationTurn,
    QueryIntentResult,
    answer_status_for_intent,
    build_intent_router,
    response_for_intent,
)
from legal_rag.query_rewrite import build_query_extensions
from legal_rag.relevance import EvidenceRelevanceGate, no_relevant_answer

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
    context: list[ConversationTurn] = Field(default_factory=list)


class LLMRequestConfig(BaseModel):
    provider: str = "openai_compatible"
    base_url: str
    api_key: str | None = None
    model: str
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=700, ge=64, le=4000)
    timeout_seconds: float = Field(default=45.0, ge=1.0, le=180.0)


class AnswerRequest(EvidenceRequest):
    llm: LLMRequestConfig


@lru_cache(maxsize=1)
def runtime():
    settings = load_settings()
    offline_store = settings.store_backend.strip().lower() in {"memory", "inmemory", "offline"}
    store = build_store(settings, offline=offline_store)
    if offline_store:
        ingest_manifest(Path(settings.corpus_manifest), store)
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
    return {
        "status": "ok",
        "collection": settings.qdrant_collection,
        "store_backend": settings.store_backend,
        "corpus_manifest": settings.corpus_manifest,
    }


@app.post("/ingest")
def ingest(manifest: str | None = None) -> dict[str, int]:
    settings, store, _, _, _ = runtime()
    manifest_path = Path(manifest or settings.corpus_manifest)
    chunks = ingest_manifest(manifest_path, store)
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
        {
            "endpoint": "/evidence",
            "query": request.query,
            "top_k": request.top_k,
            "mode": request.mode,
        },
    )
    try:
        response, hits, reranker_summary = _run_evidence_pipeline(
            request,
            settings=settings,
            store=store,
            retriever=retriever,
            rewriter=rewriter,
            trace=trace,
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


@app.post("/answer")
def answer(request: AnswerRequest):
    settings, store, retriever, _, rewriter = runtime()
    trace = TraceCollector.from_settings(settings)
    llm_config = _llm_config_from_request(request.llm)
    request_stage = trace.start_stage(
        "request",
        {
            "endpoint": "/answer",
            "query": request.query,
            "top_k": request.top_k,
            "mode": request.mode,
            "llm": llm_config.safe_dict(),
        },
    )
    try:
        response, hits, reranker_summary = _run_evidence_pipeline(
            request,
            settings=settings,
            store=store,
            retriever=retriever,
            rewriter=rewriter,
            trace=trace,
        )
        if response.get("retrieval_status") != "retrieved":
            should_generate_general = (
                response.get("retrieval_status") == "not_run"
                and response.get("requires_clarification") is not True
                and bool(request.query.strip())
            )
            should_fallback_to_model = (
                response.get("retrieval_status") == "no_relevant_evidence"
                and response.get("allow_model_fallback") is True
                and response.get("kb_required") is not True
            )
            if should_generate_general or should_fallback_to_model:
                answer_result = _generate_general_answer(
                    request,
                    llm_config=llm_config,
                    trace=trace,
                    fallback_from_kb=should_fallback_to_model,
                )
                response["answer"] = answer_result
                response["retrieval_status"] = (
                    "model_fallback" if should_fallback_to_model else "model_answer"
                )
                if should_generate_general:
                    response["answer_source"] = "model"
                trace.end_stage(
                    request_stage,
                    {
                        "hit_count": 0,
                        "trace_id": trace.trace_id,
                        "reranker": reranker_summary,
                        "answer_error": answer_result.get("error"),
                        "answer_status": answer_result.get("answer_status"),
                        "retrieval_status": response.get("retrieval_status"),
                        "intent": response.get("intent"),
                    },
                )
                return response
            answer_result = response.get("answer") or {}
            trace.end_stage(
                request_stage,
                {
                    "hit_count": 0,
                    "trace_id": trace.trace_id,
                    "reranker": reranker_summary,
                    "answer_status": answer_result.get("answer_status"),
                    "retrieval_status": response.get("retrieval_status"),
                    "intent": response.get("intent"),
                },
            )
            return response
        answer_stage = trace.start_stage(
            "answer_generation",
            {
                "hit_count": len(hits),
                "llm": llm_config.safe_dict(),
            },
        )
        answer_result = OpenAICompatibleAnswerer(llm_config).generate(request.query, hits)
        trace.end_stage(
            answer_stage,
            {
                "skipped": answer_result.get("skipped"),
                "reason": answer_result.get("reason"),
                "answer_status": answer_result.get("answer_status"),
                "duration_ms": answer_result.get("duration_ms"),
                "error": answer_result.get("error"),
                "citation_count": len(answer_result.get("citations") or []),
            },
        )
        response["answer"] = answer_result
        trace.end_stage(
            request_stage,
            {
                "hit_count": len(hits),
                "trace_id": trace.trace_id,
                "reranker": reranker_summary,
                "answer_error": answer_result.get("error"),
                "answer_status": answer_result.get("answer_status"),
            },
        )
        return response
    except Exception as exc:
        trace.add_error("answer", exc)
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


def _run_evidence_pipeline(
    request: EvidenceRequest,
    *,
    settings,
    store,
    retriever,
    rewriter,
    trace: TraceCollector,
):
    trace.add_settings(
        {
            "collection": settings.qdrant_collection,
            "store_backend": settings.store_backend,
            "corpus_manifest": settings.corpus_manifest,
            "candidate_pool": settings.candidate_pool,
            "dense_weight": settings.dense_weight,
            "bm25_weight": settings.bm25_weight,
            "rrf_k": settings.rrf_k,
            "reranker_model": settings.reranker_model,
            "rerank_top_n": settings.rerank_top_n,
            "query_extension_min_similarity": settings.query_extension_min_similarity,
            "inferred_metadata_filters_enabled": settings.inferred_metadata_filters_enabled,
            "query_rewrite_filter_min_confidence": settings.query_rewrite_filter_min_confidence,
            "intent_router_backend": getattr(settings, "intent_router_backend", "none"),
            "intent_router_model": getattr(settings, "intent_router_model", None),
            "intent_router_low_confidence_threshold": getattr(
                settings, "intent_router_low_confidence_threshold", None
            ),
        }
    )
    intent_stage = trace.start_stage("intent_route", {"query": request.query})
    context_turns = getattr(settings, "intent_router_context_turns", 5)
    router_context = request.context[-context_turns:] if context_turns else []
    intent = build_intent_router(settings).route(
        request.query,
        context=router_context,
        reranker_enabled=bool(settings.reranker_model),
    )
    trace.end_stage(intent_stage, intent.as_dict())
    if not intent.should_retrieve:
        return _non_retrieval_response(
            request,
            intent=intent,
            trace=trace,
        ), [], {"enabled": False, "skipped": True, "reason": "intent_not_retrieved"}

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
    relevance_stage = trace.start_stage(
        "relevance_gate",
        {
            "query": request.query,
            "input_count": len(hits),
        },
    )
    relevance = EvidenceRelevanceGate().evaluate(request.query, hits)
    trace.end_stage(relevance_stage, relevance.as_dict())
    if not relevance.relevant:
        pack_stage = trace.start_stage(
            "evidence_pack",
            {"hit_count": 0, "top_k": request.top_k, "retrieval_status": relevance.retrieval_status},
        )
        response = build_evidence_pack(
            request.query,
            request.mode,
            request.top_k,
            [],
            query_rewrite=query_rewrite,
            query_extensions=query_extensions,
            trace_id=trace.trace_id,
            trace_path=trace.public_path,
            trace_retention_count=trace.retention_count,
            reranker_summary=reranker_summary,
            metadata_filters=metadata_filters,
            route=intent.as_dict(),
            retrieval_status=relevance.retrieval_status,
            relevance=relevance.as_dict(),
        )
        response["answer"] = no_relevant_answer(request.query, relevance)
        trace.end_stage(pack_stage, {"hit_count": 0, "hits": []})
        return response, [], reranker_summary

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
        route=intent.as_dict(),
        retrieval_status=relevance.retrieval_status,
        relevance=relevance.as_dict(),
    )
    trace.end_stage(
        pack_stage,
        {
            "hit_count": len(hits),
            "hits": trace.hits_summary(hits),
        },
    )
    return response, hits, reranker_summary


def _non_retrieval_response(
    request: EvidenceRequest,
    *,
    intent: QueryIntentResult,
    trace: TraceCollector,
) -> dict[str, Any]:
    answer_status = answer_status_for_intent(intent)
    answer = {
        "enabled": True,
        "skipped": True,
        "reason": intent.priority_reason,
        "refused": answer_status == "refused",
        "answer_status": answer_status,
        "answer": response_for_intent(intent),
        "citations": [],
        "grounding": {
            "question_type": intent.intent,
            "answerable": False,
            "checked_hit_count": 0,
            "required_evidence": "非知识库检索输入",
            "answer_mode": "intent_route",
            "matched_cues": [],
            "missing_evidence": [],
        },
        "llm": None,
        "usage": None,
        "duration_ms": 0.0,
        "error": None,
    }
    return {
        "query": request.query,
        "intent": intent.intent,
        "intent_reason": intent.priority_reason,
        "need_retrieval": intent.need_retrieval,
        "domain": intent.domain,
        "query_type": intent.query_type,
        "retrieval_strategy": intent.retrieval_strategy,
        "answer_source": intent.answer_source,
        "kb_required": intent.kb_required,
        "allow_model_fallback": intent.allow_model_fallback,
        "requires_clarification": intent.requires_clarification,
        "clarification_question": intent.clarification_question,
        "intent_confidence": intent.confidence,
        "intent_source": intent.source,
        "intent_priority_reason": intent.priority_reason,
        "intent_error": intent.error,
        "retrieval_status": "not_run",
        "relevance": {},
        "retrieval_query": "",
        "query_rewrite": None,
        "query_extensions": None,
        "trace_id": trace.trace_id,
        "trace_path": trace.public_path,
        "trace_retention_count": trace.retention_count,
        "reranker": {"enabled": False, "skipped": True, "reason": "intent_not_retrieved"},
        "metadata_filters": {},
        "mode": request.mode,
        "top_k": request.top_k,
        "hit_count": 0,
        "hits": [],
        "answer": answer,
    }


def _generate_general_answer(
    request: AnswerRequest,
    *,
    llm_config: LLMAnswerConfig,
    trace: TraceCollector,
    fallback_from_kb: bool,
) -> dict[str, Any]:
    answer_stage = trace.start_stage(
        "general_answer_generation",
        {
            "fallback_from_kb": fallback_from_kb,
            "llm": llm_config.safe_dict(),
        },
    )
    answer_result = GeneralKnowledgeAnswerer(llm_config).generate(
        request.query,
        fallback_from_kb=fallback_from_kb,
        kb_required=False,
    )
    trace.end_stage(
        answer_stage,
        {
            "skipped": answer_result.get("skipped"),
            "reason": answer_result.get("reason"),
            "answer_status": answer_result.get("answer_status"),
            "duration_ms": answer_result.get("duration_ms"),
            "error": answer_result.get("error"),
            "citation_count": len(answer_result.get("citations") or []),
        },
    )
    return answer_result


def _llm_config_from_request(config: LLMRequestConfig) -> LLMAnswerConfig:
    return LLMAnswerConfig(
        provider=config.provider,
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
    )
