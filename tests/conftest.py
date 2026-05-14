from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.config import ROOT_DIR
from legal_rag.factory import build_answerer, build_retriever
from legal_rag.ingest import ingest_manifest
from legal_rag.store import InMemoryLegalStore


@pytest.fixture
def manifest_path() -> Path:
    return ROOT_DIR / "configs" / "legal_sources.yml"


@pytest.fixture
def eval_path() -> Path:
    return ROOT_DIR / "data" / "eval" / "legal_rag_eval.jsonl"


@pytest.fixture
def store(manifest_path: Path) -> InMemoryLegalStore:
    local_store = InMemoryLegalStore()
    ingest_manifest(manifest_path, local_store)
    return local_store


@pytest.fixture
def retriever(store: InMemoryLegalStore):
    from legal_rag.config import Settings

    settings = Settings(reranker_model=None, candidate_pool=20)
    return build_retriever(settings, store)


@pytest.fixture
def answerer():
    return build_answerer()
