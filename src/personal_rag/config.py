from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "personal_kb_rag"
    store_backend: str = "qdrant"
    corpus_manifest: str = str(ROOT_DIR / "configs" / "personal_test_sources.yml")

    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    sparse_backend: str = "bm25_jieba"
    sparse_model: str = "Qdrant/bm25"
    dense_weight: float = 0.55
    bm25_weight: float = 0.45
    rrf_k: int = 60
    candidate_pool: int = 40

    reranker_model: str | None = None
    reranker_device: str = "cpu"
    reranker_batch_size: int = 8
    reranker_onnx_file: str = "onnx/model_quint8_avx2.onnx"
    rerank_top_n: int = Field(default=20, ge=1)
    rerank_max_length: int = Field(default=512, ge=64)

    chat_model_base_url: str | None = None
    chat_model_api_key: str | None = None
    chat_model_name: str | None = None

    query_rewrite_backend: str = "none"
    query_rewrite_base_url: str = "http://localhost:11434"
    query_rewrite_model: str = "qwen3.5:9b"
    query_rewrite_timeout_seconds: float = Field(default=60.0, ge=0.1)
    query_rewrite_max_queries: int = Field(default=4, ge=1, le=10)
    inferred_metadata_filters_enabled: bool = True
    query_rewrite_filter_min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    query_extension_min_similarity: float = Field(default=0.45, ge=0.0, le=1.0)
    intent_router_backend: str = "ollama"
    intent_router_base_url: str = "http://localhost:11434"
    intent_router_model: str = "qwen3.5:9b"
    intent_router_timeout_seconds: float = Field(default=60.0, ge=0.1)
    intent_router_low_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    intent_router_context_turns: int = Field(default=5, ge=0, le=12)

    default_top_k: int = Field(default=8, ge=1)

    trace_enabled: bool = True
    trace_path: str = str(ROOT_DIR / "runtime" / "traces" / "evidence_traces.json")
    trace_retention_count: int = Field(default=30, ge=1)
    trace_include_text: bool = False
    trace_text_chars: int = Field(default=300, ge=0)
    trace_max_items_per_stage: int = Field(default=20, ge=1)

    @field_validator("reranker_model", mode="before")
    @classmethod
    def normalize_disabled_reranker(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value


@dataclass(frozen=True)
class SourceItem:
    source_id: str
    doc_type: str
    title: str
    source_ref: str
    date: str | None
    section: str | None
    path: str | None
    url: str | None
    metadata: dict[str, Any]


def load_settings(env_file: str | Path | None = None) -> Settings:
    if env_file is not None:
        load_dotenv(env_file)
    elif (ROOT_DIR / ".env").exists():
        load_dotenv(ROOT_DIR / ".env")
    return Settings()


def env_reranker_enabled() -> bool:
    value = os.getenv("RERANKER_MODEL", "")
    return value.strip().lower() not in {"", "none", "null"}
