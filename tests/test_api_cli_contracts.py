from __future__ import annotations

import json

from fastapi.testclient import TestClient

from legal_rag.api import app, build_evidence_pack
from legal_rag.query_rewrite import (
    OllamaQueryRewriter,
    QueryRewriteResult,
    build_query_extensions,
    result_from_model_payload,
)
from legal_rag.schema import EvalCase, RetrievalHit


class MappingEmbedder:
    dimensions = 2

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


def test_schema_contracts_allow_planned_fields() -> None:
    case = EvalCase(
        id="x",
        question="q",
        expected_citations=["5 U.S.C. § 553"],
        expected_answer_patterns=["notice"],
    )

    assert case.filters == {}
    assert case.must_refuse is False


def test_retrieval_hit_final_score_prefers_rerank() -> None:
    hit = RetrievalHit(
        chunk_id="1",
        source_id="s",
        doc_type="statute",
        title="t",
        citation="c",
        jurisdiction="US",
        text="text",
        fusion_score=0.1,
        rerank_score=0.9,
    )

    assert hit.final_score == 0.9


def test_evidence_pack_formats_ranked_hits() -> None:
    hit = RetrievalHit(
        chunk_id="1",
        source_id="s",
        doc_type="statute",
        title="t",
        citation="5 U.S.C. § 553",
        jurisdiction="US",
        text=" ".join(["notice"] * 80),
        dense_score=0.2,
        bm25_score=0.4,
        fusion_score=0.6,
        rerank_score=None,
        rank_explanation={"dense_rank": 1},
    )

    pack = build_evidence_pack("notice rulemaking", "hybrid", 1, [hit])

    assert pack["query"] == "notice rulemaking"
    assert pack["retrieval_query"] == "notice rulemaking"
    assert pack["query_rewrite"]["applied"] is False
    assert pack["query_extensions"]["accepted"][0]["source"] == "original"
    assert pack["mode"] == "hybrid"
    assert pack["top_k"] == 1
    assert pack["hit_count"] == 1
    assert pack["hits"][0]["rank"] == 1
    assert pack["hits"][0]["snippet"].endswith("...")
    assert len(pack["hits"][0]["snippet"]) <= 300
    assert pack["hits"][0]["scores"]["final"] == 0.6
    assert pack["hits"][0]["final_score"] == 0.6
    assert pack["hits"][0]["rank_explanation"] == {"dense_rank": 1}


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_evidence_endpoint_uses_retriever(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 2

        def __init__(self) -> None:
            self.queries: list[str] = []
            self.filters = []

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            self.queries.append(query)
            self.filters.append(filters.as_dict() if filters else {})
            assert top_k == 2
            assert mode == "bm25"
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="s",
                    doc_type="statute",
                    title="t",
                    citation="5 U.S.C. § 553",
                    jurisdiction="US",
                    text="notice text",
                    fusion_score=0.7,
                )
            ]

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "evidence_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "test"
        candidate_pool = 2
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = MappingEmbedder(
            {
                "notice": [1.0, 0.0],
                "Federal Register notice": [1.0, 0.0],
                "Federal Register": [1.0, 0.0],
            }
        )

    class FakeRewriter:
        def rewrite(self, query: str):
            assert query == "notice"
            return QueryRewriteResult(
                original_query=query,
                retrieval_query="notice Federal Register",
                backend="test",
                model="fake",
                applied=True,
                canonical_query="Federal Register notice",
                search_queries=["Federal Register"],
                filters={"doc_type": "statute"},
                confidence=0.92,
            )

    import legal_rag.api as api_module

    fake_retriever = FakeRetriever()
    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), fake_retriever, None, FakeRewriter()),
    )
    client = TestClient(app)
    response = client.post(
        "/evidence",
        json={
            "query": "notice",
            "top_k": 2,
            "mode": "bm25",
            "filters": {"jurisdiction": "us"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "notice"
    assert data["retrieval_query"] == "notice | Federal Register notice | Federal Register"
    assert data["query_rewrite"]["backend"] == "test"
    assert data["query_rewrite"]["applied"] is True
    assert data["trace_id"]
    assert data["trace_retention_count"] == 10
    assert data["reranker"]["skipped"] is True
    assert data["query_extensions"]["accepted"][0]["source"] == "original"
    assert len(data["query_extensions"]["accepted"]) == 3
    assert data["query_extensions"]["rejected"] == []
    assert data["metadata_filters"]["effective_filters"] == {
        "doc_type": ["statute"],
        "jurisdiction": ["US"],
    }
    assert data["mode"] == "bm25"
    assert data["hit_count"] == 1
    assert fake_retriever.queries == ["notice", "Federal Register notice", "Federal Register"]
    assert all(
        item == {"doc_type": ["statute"], "jurisdiction": ["US"]}
        for item in fake_retriever.filters
    )
    assert data["hits"][0]["scores"]["final"] > 0
    contributions = data["hits"][0]["rank_explanation"]["query_contributions"]
    assert [item["query"] for item in contributions] == fake_retriever.queries
    traces = json.loads((tmp_path / "evidence_traces.json").read_text(encoding="utf-8"))
    assert len(traces) == 1
    stage_names = [stage["name"] for stage in traces[0]["stages"]]
    assert "query_rewrite" in stage_names
    assert "metadata_filter_normalization" in stage_names
    assert "query_extension_filter" in stage_names
    assert "multi_query_fusion" in stage_names
    assert "cross_encoder_rerank" in stage_names


def test_retrieve_endpoint_accepts_explicit_filters(monkeypatch) -> None:
    class FakeRetriever:
        def __init__(self) -> None:
            self.filters = None

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None):
            self.filters = filters
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="s",
                    doc_type="case",
                    title="t",
                    citation="c",
                    jurisdiction="US",
                    text="case text",
                    fusion_score=0.5,
                )
            ]

    class FakeSettings:
        query_rewrite_filter_min_confidence = 0.55

    import legal_rag.api as api_module

    fake_retriever = FakeRetriever()
    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), None, fake_retriever, None, None),
    )
    client = TestClient(app)
    response = client.post(
        "/retrieve",
        json={"query": "Chevron", "filters": {"doc_type": "case", "jurisdiction": "us"}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata_filters"]["effective_filters"] == {
        "doc_type": ["case"],
        "jurisdiction": ["US"],
    }
    assert fake_retriever.filters.as_dict() == {
        "doc_type": ["case"],
        "jurisdiction": ["US"],
    }


def test_query_extensions_filter_and_dedupe_candidates() -> None:
    rewrite = QueryRewriteResult(
        original_query="辛普森案",
        retrieval_query="expanded",
        applied=True,
        canonical_query="O.J. Simpson murder trial",
        search_queries=["People v. Simpson", "unrelated topic"],
        aliases=["People v. Simpson"],
    )
    embedder = MappingEmbedder(
        {
            "辛普森案": [1.0, 0.0],
            "O.J. Simpson murder trial": [0.0, 1.0],
            "People v. Simpson": [0.0, 1.0],
            "unrelated topic": [-1.0, 0.0],
        }
    )

    result = build_query_extensions(rewrite, embedder=embedder, min_similarity=0.45)

    assert [item.query for item in result.accepted] == [
        "辛普森案",
        "O.J. Simpson murder trial",
        "People v. Simpson",
    ]
    assert result.accepted[0].weight == 1.0
    assert result.accepted[1].similarity == 1.0
    assert [item.query for item in result.rejected] == ["unrelated topic"]
    assert result.rejected[0].reason == "below_similarity_threshold"


def test_query_extensions_fall_back_to_original_when_all_candidates_fail() -> None:
    rewrite = QueryRewriteResult(
        original_query="notice",
        retrieval_query="notice drift",
        applied=True,
        search_queries=["drift"],
    )
    embedder = MappingEmbedder({"notice": [1.0, 0.0], "drift": [0.0, 1.0]})

    result = build_query_extensions(rewrite, embedder=embedder, min_similarity=0.9)

    assert [item.query for item in result.accepted] == ["notice"]
    assert result.rejected[0].query == "drift"


def test_query_rewrite_payload_builds_expanded_retrieval_query() -> None:
    result = result_from_model_payload(
        "辛普森案",
        {
            "canonical_query": "O.J. Simpson murder trial",
            "search_queries": ["People v. Simpson", "O.J. Simpson criminal case"],
            "aliases": ["Orenthal James Simpson", "People v. Simpson"],
            "filters": {"doc_type": "case", "jurisdiction": "US"},
            "confidence": 0.82,
        },
        backend="ollama",
        model="gemma4:e2b",
        max_queries=4,
    )

    assert result.applied is True
    assert result.retrieval_query.startswith("辛普森案 O.J. Simpson murder trial")
    assert result.search_queries == ["People v. Simpson", "O.J. Simpson criminal case"]
    assert result.aliases == ["Orenthal James Simpson", "People v. Simpson"]
    assert result.filters["doc_type"] == "case"
    assert result.confidence == 0.82


def test_ollama_query_rewriter_disables_thinking(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        '{"canonical_query":"notice rulemaking",'
                        '"search_queries":["Federal Register notice"],'
                        '"aliases":["5 U.S.C. 553"],'
                        '"filters":{"doc_type":"statute","jurisdiction":"US"},'
                        '"confidence":0.9}'
                    )
                }
            }

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    import legal_rag.query_rewrite as query_rewrite_module

    monkeypatch.setattr(query_rewrite_module.httpx, "post", fake_post)

    result = OllamaQueryRewriter(
        base_url="http://ollama:11434",
        model="gemma4:e2b",
    ).rewrite("notice")

    assert captured["json"]["think"] is False
    assert result.applied is True
    assert result.canonical_query == "notice rulemaking"


def test_frontend_is_served() -> None:
    client = TestClient(app)

    index_response = client.get("/")
    css_response = client.get("/static/styles.css")

    assert index_response.status_code == 200
    assert "Legal RAG" in index_response.text
    assert css_response.status_code == 200
    assert "docket-shell" in css_response.text
