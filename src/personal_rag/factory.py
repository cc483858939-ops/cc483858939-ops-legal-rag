from __future__ import annotations

from pathlib import Path

from personal_rag.answer import ExtractiveAnswerer
from personal_rag.config import Settings, load_settings
from personal_rag.embeddings import FastEmbedEmbedder, HashingEmbedder, SentenceTransformerEmbedder
from personal_rag.fusion import FusionConfig
from personal_rag.ingest import ingest_manifest
from personal_rag.query_rewrite import QueryRewriter, build_query_rewriter
from personal_rag.rerank import build_optional_reranker
from personal_rag.retriever import HybridRetriever
from personal_rag.store import InMemoryPersonalStore, QdrantPersonalStore


def build_embedder(settings: Settings, *, offline: bool = False):
    if offline:
        return HashingEmbedder()
    backend = settings.embedding_backend.strip().lower()
    if backend == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unsupported EMBEDDING_BACKEND={settings.embedding_backend!r}")


def build_store(settings: Settings, *, offline: bool = False):
    embedder = build_embedder(settings, offline=offline)
    if offline:
        return InMemoryPersonalStore(embedder=embedder)
    return QdrantPersonalStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        embedder=embedder,
        sparse_backend=settings.sparse_backend,
        sparse_model=settings.sparse_model,
    )


def build_retriever(
    settings: Settings,
    store,
    *,
    offline_reranker: bool = False,
) -> HybridRetriever:
    reranker_model = "fake-lexical-reranker" if offline_reranker else settings.reranker_model
    return HybridRetriever(
        store=store,
        fusion_config=FusionConfig(
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
            rrf_k=settings.rrf_k,
        ),
        reranker=build_optional_reranker(
            reranker_model,
            device=settings.reranker_device,
            batch_size=settings.reranker_batch_size,
            max_length=settings.rerank_max_length,
            onnx_file=settings.reranker_onnx_file,
        ),
        candidate_pool=settings.candidate_pool,
        rerank_top_n=settings.rerank_top_n,
    )


def build_answerer() -> ExtractiveAnswerer:
    return ExtractiveAnswerer()


def build_rewriter(settings: Settings) -> QueryRewriter:
    return build_query_rewriter(settings)


def build_offline_runtime(manifest: str | Path):
    settings = load_settings()
    store = build_store(settings, offline=True)
    ingest_manifest(manifest, store)
    return build_retriever(settings, store), build_answerer()
