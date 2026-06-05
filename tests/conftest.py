from __future__ import annotations

from pathlib import Path

import pytest

from personal_rag.config import ROOT_DIR
from personal_rag.factory import build_answerer, build_retriever
from personal_rag.ingest import ingest_manifest
from personal_rag.store import InMemoryPersonalStore


@pytest.fixture
def manifest_path() -> Path:
    return ROOT_DIR / "configs" / "personal_test_sources.yml"


@pytest.fixture
def eval_path() -> Path:
    return ROOT_DIR / "data" / "eval" / "personal_test_eval.yml"


@pytest.fixture
def store(manifest_path: Path) -> InMemoryPersonalStore:
    local_store = InMemoryPersonalStore()
    ingest_manifest(manifest_path, local_store)
    return local_store


@pytest.fixture
def retriever(store: InMemoryPersonalStore):
    from personal_rag.config import Settings

    settings = Settings(reranker_model=None, candidate_pool=20)
    return build_retriever(settings, store)


@pytest.fixture
def answerer():
    return build_answerer()
