from __future__ import annotations

import pytest

from legal_rag.config import Settings, env_reranker_enabled
from legal_rag.embeddings import HashingEmbedder
from legal_rag.evidence import retrieve_with_query_extensions_and_summary
from legal_rag.factory import build_embedder, build_retriever
from legal_rag.metadata_filter import MetadataFilters
from legal_rag.query_rewrite import AcceptedQueryExtension
from legal_rag.rerank import LexicalReranker, UnavailableReranker, rerank_hits
from legal_rag.schema import RetrievalHit
from legal_rag.store import InMemoryLegalStore


def test_default_embedding_uses_lightweight_chinese_fastembed() -> None:
    settings = Settings()

    assert settings.embedding_backend == "fastembed"
    assert settings.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert settings.reranker_model is None


def test_disabled_reranker_env_values_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "none")

    assert Settings().reranker_model is None
    assert not env_reranker_enabled()


def test_hybrid_retrieval_finds_statute(retriever) -> None:
    hits = retriever.retrieve("Federal Register proposed rule making notice", top_k=5)

    assert hits
    assert hits[0].citation == "5 U.S.C. § 553"
    assert hits[0].fusion_score > 0


def test_dense_only_and_bm25_only_modes_work(retriever) -> None:
    dense_hits = retriever.retrieve("fair use criticism teaching research", top_k=5, mode="dense")
    bm25_hits = retriever.retrieve("fair use criticism teaching research", top_k=5, mode="bm25")

    assert any(hit.citation == "17 U.S.C. § 107" for hit in dense_hits)
    assert any(hit.citation == "17 U.S.C. § 107" for hit in bm25_hits)


def test_metadata_filter_limits_retrieval_results(retriever) -> None:
    hits = retriever.retrieve(
        "Chevron Loper Bright agency interpretation statute",
        top_k=8,
        filters=MetadataFilters(values={"doc_type": ("statute",)}),
    )

    assert hits
    assert {hit.doc_type for hit in hits} == {"statute"}


def test_reranker_skips_when_model_is_empty(store: InMemoryLegalStore) -> None:
    retriever = build_retriever(Settings(reranker_model=None, candidate_pool=20), store)
    hits = retriever.retrieve("Twombly plausible pleading", top_k=3)

    assert hits
    assert all(hit.rerank_score is None for hit in hits)


def test_reranker_runs_when_configured(store: InMemoryLegalStore) -> None:
    retriever = build_retriever(
        Settings(reranker_model="fake-lexical-reranker", candidate_pool=20),
        store,
    )
    hits = retriever.retrieve("Twombly plausible pleading", top_k=3)

    assert hits
    assert all(hit.rerank_score is not None for hit in hits)
    assert isinstance(retriever.reranker, LexicalReranker)


def test_unavailable_reranker_fails_open() -> None:
    hits = [
        RetrievalHit(
            chunk_id="1",
            source_id="s",
            doc_type="case",
            title="A",
            citation="A",
            jurisdiction="US",
            text="alpha",
            fusion_score=0.5,
        )
    ]

    output, summary = rerank_hits(
        UnavailableReranker("cross-encoder/missing", "load failed"),
        "alpha",
        hits,
        top_n=20,
    )

    assert output == hits
    assert summary["skipped"] is True
    assert summary["reason"] == "reranker_unavailable"


def test_post_fusion_rerank_preserves_query_contributions() -> None:
    class FakeRetriever:
        candidate_pool = 2
        rerank_top_n = 20
        reranker = LexicalReranker()

        def retrieve(self, query: str, **kwargs):
            if query == "Twombly pleading":
                return [
                    RetrievalHit(
                        chunk_id="twombly",
                        source_id="twombly",
                        doc_type="case",
                        title="Twombly",
                        citation="Twombly",
                        jurisdiction="US",
                        text="Twombly plausible pleading standard",
                        fusion_score=0.1,
                    )
                ]
            return [
                RetrievalHit(
                    chunk_id="other",
                    source_id="other",
                    doc_type="case",
                    title="Other",
                    citation="Other",
                    jurisdiction="US",
                    text="unrelated administrative record",
                    fusion_score=0.9,
                )
            ]

    hits, summary = retrieve_with_query_extensions_and_summary(
        FakeRetriever(),
        [
            AcceptedQueryExtension("other", "original", 1.0, 1.0),
            AcceptedQueryExtension("Twombly pleading", "search_query", 0.9, 0.9),
        ],
        top_k=2,
        mode="hybrid",
        rerank_query="Twombly pleading",
    )

    assert hits[0].citation == "Twombly"
    assert summary["skipped"] is False
    assert "query_contributions" in hits[0].rank_explanation
    assert hits[0].rank_explanation["reranker"]["model"] == "fake-lexical-reranker"


def test_offline_embedder_uses_hashing_even_when_fastembed_is_default() -> None:
    embedder = build_embedder(Settings(embedding_backend="fastembed"), offline=True)

    assert isinstance(embedder, HashingEmbedder)


def test_unknown_embedding_backend_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_BACKEND"):
        build_embedder(Settings(embedding_backend="unknown"))


def test_cross_document_query_can_surface_multiple_authorities(retriever) -> None:
    hits = retriever.retrieve("section 1983 Monell policy custom", top_k=8)
    citations = {hit.citation for hit in hits}

    assert "42 U.S.C. § 1983" in citations
    assert "Monell v. Department of Social Services, 436 U.S. 658 (1978)" in citations


def test_no_answer_query_retrieval_does_not_force_answer(retriever, answerer) -> None:
    hits = retriever.retrieve("Delaware appraisal rights cash out merger", top_k=5)
    answer = answerer.answer("Delaware appraisal rights cash out merger", hits)

    assert answer.refused is True
