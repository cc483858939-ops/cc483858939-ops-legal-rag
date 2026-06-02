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
from legal_rag.query_intent import QueryIntentResult
from legal_rag.schema import EvalCase, RetrievalHit


class MappingEmbedder:
    dimensions = 2

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


def route_result(**overrides) -> QueryIntentResult:
    data = {
        "need_retrieval": True,
        "intent": "answer_question",
        "domain": "personal_kb",
        "query_type": "factual",
        "retrieval_strategy": "hybrid",
        "answer_source": "personal_kb",
        "kb_required": False,
        "allow_model_fallback": True,
        "requires_clarification": False,
        "clarification_question": None,
        "confidence": 0.91,
        "source": "llm",
        "priority_reason": "rag",
        "error": None,
    }
    data.update(overrides)
    return QueryIntentResult.model_validate(data)


def fake_settings(tmp_path, trace_name: str = "traces.json"):
    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / trace_name)
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    return FakeSettings()


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
    assert pack["candidate_hit_count"] == 1
    assert pack["relevant_hit_count"] == 1
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
        store_backend = "memory"
        corpus_manifest = "configs/test_sources.yml"
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
    assert data["retrieval_status"] == "retrieved"
    assert data["answer_source"] == "personal_kb"
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


def test_answer_endpoint_adds_llm_answer(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="support-returns",
                    doc_type="policy",
                    title="Returns and Refunds Policy",
                    citation="Support KB Returns and Refunds v2026.05",
                    jurisdiction="GLOBAL",
                    text="Refunds are issued within 7 business days after inspection approval.",
                    fusion_score=0.7,
                )
            ]

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "answer_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "support_rag"
        store_backend = "qdrant"
        corpus_manifest = "configs/support_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = MappingEmbedder({"退款多久到账": [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, hits: list[RetrievalHit]):
            return {
                "enabled": True,
                "skipped": False,
                "reason": None,
                "refused": False,
                "answer_status": "answered",
                "answer": "退款通常在验收通过后 7 个工作日内完成。[1]",
                "citations": [{"rank": 1, "title": hits[0].title, "citation": hits[0].citation}],
                "grounding": {
                    "question_type": "other",
                    "answerable": True,
                    "checked_hit_count": len(hits),
                    "required_evidence": "直接回答问题的检索证据",
                    "answer_mode": "direct",
                    "matched_cues": [],
                    "missing_evidence": [],
                },
                "llm": self.config.safe_dict(),
                "usage": {"total_tokens": 12},
                "duration_ms": 1.0,
                "error": None,
            }

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "OpenAICompatibleAnswerer", FakeAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": "退款多久到账",
            "top_k": 1,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "api_key": "secret",
                "model": "example-model",
                "thinking_enabled": True,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["hit_count"] == 1
    assert data["answer"]["answer"].startswith("退款通常")
    assert data["answer"]["refused"] is False
    assert data["answer"]["answer_status"] == "answered"
    assert data["answer"]["grounding"]["answerable"] is True
    assert data["answer"]["grounding"]["answer_mode"] == "direct"
    assert data["answer"]["grounding"]["missing_evidence"] == []
    assert data["answer"]["llm"]["model"] == "example-model"
    assert data["answer"]["llm"]["thinking_enabled"] is True
    assert data["answer"]["llm"]["timeout_seconds"] == 180.0
    assert data["answer"]["llm"]["api_key_configured"] is True
    assert "secret" not in json.dumps(data, ensure_ascii=False)
    traces = json.loads((tmp_path / "answer_traces.json").read_text(encoding="utf-8"))
    assert "answer_generation" in [stage["name"] for stage in traces[0]["stages"]]


def test_answer_endpoint_routes_statement_without_retrieval(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval should not run for statement input")

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "statement_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = None

    class FakeRewriter:
        def rewrite(self, query: str):
            raise AssertionError("query rewrite should not run for statement input")

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=False,
                intent="assistant_meta",
                domain="general_chat",
                query_type="statement",
                retrieval_strategy="none",
                answer_source="none",
                kb_required=False,
                allow_model_fallback=False,
                confidence=0.92,
                priority_reason="non_rag_intent",
            )

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, *, fallback_from_kb: bool = False, kb_required: bool = False):
            assert question == "你说蓝色是你最爱的颜色"
            assert fallback_from_kb is False
            assert kb_required is False
            return {
                "enabled": True,
                "skipped": False,
                "reason": "model_direct",
                "refused": False,
                "answer_status": "answered",
                "answer": "我没有个人偏好；如果你想聊蓝色，我可以继续。",
                "citations": [],
                "grounding": {
                    "question_type": "other",
                    "answerable": True,
                    "checked_hit_count": 0,
                    "required_evidence": "模型通用知识；不使用个人知识库引用",
                    "answer_mode": "general_knowledge",
                    "matched_cues": [],
                    "missing_evidence": [],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": "你说蓝色是你最爱的颜色",
            "top_k": 8,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "assistant_meta"
    assert data["domain"] == "general_chat"
    assert data["query_type"] == "statement"
    assert data["need_retrieval"] is False
    assert data["intent_confidence"] == 0.92
    assert data["intent_priority_reason"] == "non_rag_intent"
    assert data["answer_source"] == "model"
    assert data["retrieval_status"] == "model_answer"
    assert data["hit_count"] == 0
    assert data["hits"] == []
    assert data["query_rewrite"] is None
    assert data["answer"]["answer_status"] == "answered"
    assert data["answer"]["answer"].startswith("我没有个人偏好")
    assert data["answer"]["citations"] == []


def test_answer_endpoint_routes_memory_candidate_to_general_model(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval should not run for memory candidate")

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "memory_answer_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = None

    class FakeRewriter:
        def rewrite(self, query: str):
            raise AssertionError("query rewrite should not run for memory candidate")

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=False,
                intent="memory_candidate",
                domain="personal_kb",
                query_type="preference",
                retrieval_strategy="none",
                answer_source="none",
                kb_required=False,
                allow_model_fallback=False,
                confidence=0.96,
                priority_reason="non_rag_intent",
            )

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, *, fallback_from_kb: bool = False, kb_required: bool = False):
            assert question == "我最喜欢蓝色"
            assert fallback_from_kb is False
            assert kb_required is False
            return {
                "enabled": True,
                "skipped": False,
                "reason": "model_direct",
                "refused": False,
                "answer_status": "answered",
                "answer": "知道了，你最喜欢蓝色。",
                "citations": [],
                "grounding": {
                    "question_type": "other",
                    "answerable": True,
                    "checked_hit_count": 0,
                    "required_evidence": "模型通用知识；不使用个人知识库引用",
                    "answer_mode": "general_knowledge",
                    "matched_cues": [],
                    "missing_evidence": [],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": "我最喜欢蓝色",
            "top_k": 8,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "memory_candidate"
    assert data["need_retrieval"] is False
    assert data["answer_source"] == "model"
    assert data["retrieval_status"] == "model_answer"
    assert data["hit_count"] == 0
    assert data["hits"] == []
    assert data["answer"]["answer_status"] == "answered"
    assert data["answer"]["answer"] == "知道了，你最喜欢蓝色。"
    assert "重新 ingest" not in data["answer"]["answer"]


def test_evidence_endpoint_routes_memory_candidate_without_retrieval(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval should not run for memory candidate")

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "memory_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = None

    class FakeRewriter:
        def rewrite(self, query: str):
            raise AssertionError("query rewrite should not run for memory candidate")

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=False,
                intent="memory_candidate",
                domain="personal_kb",
                query_type="preference",
                retrieval_strategy="none",
                answer_source="none",
                kb_required=False,
                allow_model_fallback=False,
                confidence=0.94,
                priority_reason="non_rag_intent",
            )

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())

    client = TestClient(app)
    response = client.post("/evidence", json={"query": "记住我最喜欢蓝色"})

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "memory_candidate"
    assert data["need_retrieval"] is False
    assert data["intent_confidence"] == 0.94
    assert data["retrieval_status"] == "not_run"
    assert data["answer"]["answer_status"] == "memory_candidate"
    assert "不会自动写入知识库" in data["answer"]["answer"]


def test_evidence_endpoint_returns_high_confidence_clarification(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval should not run when clarification has priority")

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "clarification_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = None

    class FakeRewriter:
        def rewrite(self, query: str):
            raise AssertionError("query rewrite should not run when clarification has priority")

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=False,
                intent="unclear",
                domain="personal_kb",
                query_type="unknown",
                retrieval_strategy="none",
                answer_source="none",
                kb_required=False,
                allow_model_fallback=False,
                requires_clarification=True,
                clarification_question="你想问哪份笔记里的他？",
                confidence=0.88,
                priority_reason="clarification",
            )

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())

    client = TestClient(app)
    response = client.post("/evidence", json={"query": "那他为什么这样"})

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_status"] == "not_run"
    assert data["requires_clarification"] is True
    assert data["clarification_question"] == "你想问哪份笔记里的他？"
    assert data["answer"]["answer_status"] == "clarification"
    assert data["answer"]["answer"] == "你想问哪份笔记里的他？"


def test_low_confidence_router_result_continues_to_retrieval(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def __init__(self) -> None:
            self.called = False

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            self.called = True
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="note",
                    doc_type="note",
                    title="Color Note",
                    citation="Color Note v1",
                    jurisdiction="PERSONAL",
                    text="用户最喜欢的颜色是蓝色。",
                    bm25_score=0.5,
                    fusion_score=0.5,
                )
            ]

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "low_confidence_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = MappingEmbedder({"我最喜欢什么颜色": [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="casual_chat",
                domain="general_chat",
                query_type="preference",
                retrieval_strategy="hybrid",
                confidence=0.6,
                priority_reason="low_confidence",
            )

    import legal_rag.api as api_module

    fake_retriever = FakeRetriever()
    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), fake_retriever, None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())

    client = TestClient(app)
    response = client.post("/evidence", json={"query": "我最喜欢什么颜色"})

    assert response.status_code == 200
    data = response.json()
    assert fake_retriever.called is True
    assert data["retrieval_status"] == "retrieved"
    assert data["intent_priority_reason"] == "low_confidence"
    assert data["hit_count"] == 1


def test_answer_endpoint_direct_model_route_skips_retrieval(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval should not run for direct model answers")

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "direct_model_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = None

    class FakeRewriter:
        def rewrite(self, query: str):
            raise AssertionError("query rewrite should not run for direct model answers")

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=False,
                intent="answer_question",
                domain="general_chat",
                query_type="exact",
                retrieval_strategy="none",
                answer_source="model",
                kb_required=False,
                allow_model_fallback=False,
                confidence=0.95,
                priority_reason="model",
            )

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, *, fallback_from_kb: bool = False, kb_required: bool = False):
            assert question == "1+1等于几"
            assert fallback_from_kb is False
            assert kb_required is False
            return {
                "enabled": True,
                "skipped": False,
                "reason": "model_direct",
                "refused": False,
                "answer_status": "answered",
                "answer": "2",
                "citations": [],
                "grounding": {
                    "question_type": "exact",
                    "answerable": True,
                    "checked_hit_count": 0,
                    "required_evidence": "模型通用知识；不使用个人知识库引用",
                    "answer_mode": "general_knowledge",
                    "matched_cues": [],
                    "missing_evidence": [],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": "1+1等于几",
            "top_k": 8,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_status"] == "model_answer"
    assert data["answer_source"] == "model"
    assert data["need_retrieval"] is False
    assert data["hit_count"] == 0
    assert data["hits"] == []
    assert data["answer"]["answer_status"] == "answered"
    assert data["answer"]["answer"] == "2"
    assert data["answer"]["citations"] == []


def test_answer_endpoint_falls_back_to_model_when_personal_kb_has_no_relevant_evidence(
    monkeypatch, tmp_path
) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="unrelated",
                    doc_type="note",
                    title="Unrelated Note",
                    citation="Unrelated Note v1",
                    jurisdiction="PERSONAL",
                    text="退款规则和发货地址说明。",
                    dense_score=0.0,
                    bm25_score=0.0,
                    fusion_score=0.01,
                )
            ]

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "model_fallback_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = MappingEmbedder({"高斯是谁": [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="answer_question",
                domain="personal_kb",
                query_type="factual",
                retrieval_strategy="hybrid",
                answer_source="personal_kb",
                kb_required=False,
                allow_model_fallback=True,
                confidence=0.94,
                priority_reason="rag",
            )

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, *, fallback_from_kb: bool = False, kb_required: bool = False):
            assert question == "高斯是谁"
            assert fallback_from_kb is True
            assert kb_required is False
            return {
                "enabled": True,
                "skipped": False,
                "reason": "kb_no_relevant_evidence",
                "refused": False,
                "answer_status": "model_fallback",
                "answer": "个人知识库没有找到直接相关内容；按通用知识，高斯通常指卡尔·弗里德里希·高斯。",
                "citations": [],
                "grounding": {
                    "question_type": "factual",
                    "answerable": True,
                    "checked_hit_count": 0,
                    "required_evidence": "模型通用知识；不使用个人知识库引用",
                    "answer_mode": "general_knowledge_fallback",
                    "matched_cues": [],
                    "missing_evidence": ["personal_kb_evidence"],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": "高斯是谁",
            "top_k": 1,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer_source"] == "model"
    assert data["allow_model_fallback"] is True
    assert data["retrieval_status"] == "model_fallback"
    assert data["hit_count"] == 0
    assert data["candidate_hit_count"] == 1
    assert data["relevant_hit_count"] == 0
    assert data["hits"] == []
    assert data["answer"]["answer_status"] == "model_fallback"
    assert data["answer"]["citations"] == []
    assert data["answer"]["answer"].startswith("个人知识库没有找到直接相关内容")


def test_answer_endpoint_post_answer_fallbacks_when_retrieved_hits_do_not_answer(
    monkeypatch, tmp_path
) -> None:
    query = "爱因斯坦是谁"

    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="personal-test",
                    doc_type="note",
                    title="Personal Test Note",
                    citation="Personal Test Note v1",
                    jurisdiction="PERSONAL",
                    text="炫神最喜欢的歌是打火机。",
                    dense_score=0.0,
                    bm25_score=1.0,
                    fusion_score=0.5,
                )
            ]

    class FakeStore:
        embedder = MappingEmbedder({query: [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="answer_question",
                domain="personal_kb",
                query_type="factual",
                retrieval_strategy="hybrid",
                answer_source="personal_kb",
                kb_required=False,
                allow_model_fallback=True,
                confidence=0.94,
                priority_reason="kb_first",
            )

    class FakeAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, hits: list[RetrievalHit]):
            assert question == query
            return {
                "enabled": True,
                "skipped": False,
                "reason": None,
                "refused": False,
                "answer_status": "answered",
                "answer": "根据现有材料来看，没有关于爱因斯坦的信息。",
                "citations": [{"rank": 1, "title": hits[0].title, "citation": hits[0].citation}],
                "grounding": {
                    "question_type": "factual",
                    "answerable": True,
                    "checked_hit_count": len(hits),
                    "required_evidence": "直接回答问题的检索证据",
                    "answer_mode": "direct",
                    "matched_cues": [],
                    "missing_evidence": [],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, *, fallback_from_kb: bool = False, kb_required: bool = False):
            assert question == query
            assert fallback_from_kb is True
            assert kb_required is False
            return {
                "enabled": True,
                "skipped": False,
                "reason": "kb_no_relevant_evidence",
                "refused": False,
                "answer_status": "model_fallback",
                "answer": "个人知识库没有找到直接相关内容；根据一般常识，爱因斯坦是物理学家。",
                "citations": [],
                "grounding": {
                    "question_type": "factual",
                    "answerable": True,
                    "checked_hit_count": 0,
                    "required_evidence": "模型通用知识；不使用个人知识库引用",
                    "answer_mode": "general_knowledge_fallback",
                    "matched_cues": [],
                    "missing_evidence": ["personal_kb_evidence"],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (
            fake_settings(tmp_path, "post_answer_fallback_traces.json"),
            FakeStore(),
            FakeRetriever(),
            None,
            FakeRewriter(),
        ),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "OpenAICompatibleAnswerer", FakeAnswerer)
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": query,
            "top_k": 1,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_status"] == "model_fallback"
    assert data["answer_source"] == "model"
    assert data["allow_model_fallback"] is True
    assert data["hit_count"] == 1
    assert data["answer"]["answer_status"] == "model_fallback"
    assert data["answer"]["citations"] == []
    assert data["answer"]["answer"].startswith("个人知识库没有找到直接相关内容；根据一般常识，")


def test_answer_endpoint_does_not_post_fallback_when_retrieved_hit_has_matched_kb_cue(
    monkeypatch, tmp_path
) -> None:
    query = "打火机是什么"

    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="personal-test",
                    doc_type="note",
                    title="Personal Test Note",
                    citation="Personal Test Note v1",
                    jurisdiction="PERSONAL",
                    text="炫神最喜欢的歌是打火机。",
                    dense_score=0.0,
                    bm25_score=1.0,
                    fusion_score=0.5,
                )
            ]

    class FakeStore:
        embedder = MappingEmbedder({query: [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="answer_question",
                domain="personal_kb",
                query_type="definition",
                retrieval_strategy="hybrid",
                answer_source="personal_kb",
                kb_required=False,
                allow_model_fallback=True,
                confidence=0.94,
                priority_reason="kb_first",
            )

    class FakeAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, hits: list[RetrievalHit]):
            assert question == query
            return {
                "enabled": True,
                "skipped": False,
                "reason": None,
                "refused": False,
                "answer_status": "answered",
                "answer": "根据现有材料来看，没有关于“打火机”的定义。",
                "citations": [{"rank": 1, "title": hits[0].title, "citation": hits[0].citation}],
                "grounding": {
                    "question_type": "definition",
                    "answerable": True,
                    "checked_hit_count": len(hits),
                    "required_evidence": "实体在知识库语境中的描述、属性或关系证据",
                    "answer_mode": "contextual_definition",
                    "matched_cues": ["打火机"],
                    "missing_evidence": [],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, *args, **kwargs):
            raise AssertionError("matched KB evidence must not fall back to model")

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (
            fake_settings(tmp_path, "matched_kb_no_post_fallback_traces.json"),
            FakeStore(),
            FakeRetriever(),
            None,
            FakeRewriter(),
        ),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "OpenAICompatibleAnswerer", FakeAnswerer)
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": query,
            "top_k": 1,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_status"] == "retrieved"
    assert data["answer_source"] == "personal_kb"
    assert data["allow_model_fallback"] is True
    assert data["hit_count"] == 1
    assert data["relevance"]["matched_terms"] == ["打火机"]
    assert data["answer"]["answer_status"] == "answered"
    assert data["answer"]["grounding"]["matched_cues"] == ["打火机"]
    assert data["answer"]["answer"] == "根据现有材料来看，没有关于“打火机”的定义。"


def test_answer_endpoint_does_not_post_fallback_for_strict_kb_question(
    monkeypatch, tmp_path
) -> None:
    query = "我的笔记里爱因斯坦是谁"

    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="personal-test",
                    doc_type="note",
                    title="Personal Test Note",
                    citation="Personal Test Note v1",
                    jurisdiction="PERSONAL",
                    text="炫神最喜欢的歌是打火机。",
                    dense_score=0.0,
                    bm25_score=1.0,
                    fusion_score=0.5,
                )
            ]

    class FakeStore:
        embedder = MappingEmbedder({query: [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="answer_question",
                domain="personal_kb",
                query_type="factual",
                retrieval_strategy="hybrid",
                answer_source="personal_kb",
                kb_required=True,
                allow_model_fallback=False,
                confidence=0.94,
                priority_reason="kb_only",
            )

    class FakeAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, hits: list[RetrievalHit]):
            return {
                "enabled": True,
                "skipped": False,
                "reason": None,
                "refused": False,
                "answer_status": "answered",
                "answer": "根据现有材料来看，没有关于爱因斯坦的信息。",
                "citations": [{"rank": 1, "title": hits[0].title, "citation": hits[0].citation}],
                "grounding": {
                    "question_type": "factual",
                    "answerable": True,
                    "checked_hit_count": len(hits),
                    "required_evidence": "直接回答问题的检索证据",
                    "answer_mode": "direct",
                    "matched_cues": [],
                    "missing_evidence": [],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, *args, **kwargs):
            raise AssertionError("strict KB questions must not fall back to the model")

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (
            fake_settings(tmp_path, "strict_post_answer_traces.json"),
            FakeStore(),
            FakeRetriever(),
            None,
            FakeRewriter(),
        ),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "OpenAICompatibleAnswerer", FakeAnswerer)
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": query,
            "top_k": 1,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_status"] == "retrieved"
    assert data["answer_source"] == "personal_kb"
    assert data["kb_required"] is True
    assert data["allow_model_fallback"] is False
    assert data["answer"]["answer_status"] == "answered"
    assert data["answer"]["answer"] == "根据现有材料来看，没有关于爱因斯坦的信息。"


def test_answer_endpoint_does_not_post_fallback_for_llm_error(monkeypatch, tmp_path) -> None:
    query = "爱因斯坦是谁"

    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="personal-test",
                    doc_type="note",
                    title="Personal Test Note",
                    citation="Personal Test Note v1",
                    jurisdiction="PERSONAL",
                    text="炫神最喜欢的歌是打火机。",
                    dense_score=0.0,
                    bm25_score=1.0,
                    fusion_score=0.5,
                )
            ]

    class FakeStore:
        embedder = MappingEmbedder({query: [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="answer_question",
                domain="personal_kb",
                query_type="factual",
                retrieval_strategy="hybrid",
                answer_source="personal_kb",
                kb_required=False,
                allow_model_fallback=True,
                confidence=0.94,
                priority_reason="kb_first",
            )

    class FakeAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, hits: list[RetrievalHit]):
            return {
                "enabled": True,
                "skipped": True,
                "reason": "llm_error",
                "refused": False,
                "answer_status": "error",
                "answer": "insufficient evidence",
                "citations": [{"rank": 1, "title": hits[0].title, "citation": hits[0].citation}],
                "grounding": {},
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": "RuntimeError: boom",
            }

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, *args, **kwargs):
            raise AssertionError("LLM errors must not be converted to model fallback")

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (
            fake_settings(tmp_path, "llm_error_no_fallback_traces.json"),
            FakeStore(),
            FakeRetriever(),
            None,
            FakeRewriter(),
        ),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "OpenAICompatibleAnswerer", FakeAnswerer)
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": query,
            "top_k": 1,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_status"] == "retrieved"
    assert data["answer_source"] == "personal_kb"
    assert data["allow_model_fallback"] is True
    assert data["answer"]["answer_status"] == "error"
    assert data["answer"]["reason"] == "llm_error"


def test_evidence_endpoint_does_not_fallback_to_model_when_personal_kb_misses(
    monkeypatch, tmp_path
) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="unrelated",
                    doc_type="note",
                    title="Unrelated Note",
                    citation="Unrelated Note v1",
                    jurisdiction="PERSONAL",
                    text="退款规则和发货地址说明。",
                    dense_score=0.0,
                    bm25_score=0.0,
                    fusion_score=0.01,
                )
            ]

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "evidence_no_fallback_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = MappingEmbedder({"高斯是谁": [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="answer_question",
                domain="personal_kb",
                query_type="factual",
                retrieval_strategy="hybrid",
                answer_source="personal_kb",
                kb_required=False,
                allow_model_fallback=True,
                confidence=0.94,
                priority_reason="rag",
            )

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())

    client = TestClient(app)
    response = client.post("/evidence", json={"query": "高斯是谁", "top_k": 1, "mode": "hybrid"})

    assert response.status_code == 200
    data = response.json()
    assert data["allow_model_fallback"] is True
    assert data["retrieval_status"] == "no_relevant_evidence"
    assert data["hit_count"] == 0
    assert data["candidate_hit_count"] == 1
    assert data["relevant_hit_count"] == 0
    assert data["hits"] == []
    assert data["answer"]["answer_status"] == "refused"
    assert data["answer"]["reason"] == "no_relevant_evidence"


def test_answer_endpoint_refuses_strict_kb_low_relevance_without_fallback(monkeypatch, tmp_path) -> None:
    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="unrelated",
                    doc_type="note",
                    title="Unrelated Note",
                    citation="Unrelated Note v1",
                    jurisdiction="PERSONAL",
                    text="退款规则和发货地址说明。",
                    dense_score=0.0,
                    bm25_score=0.0,
                    fusion_score=0.01,
                )
            ]

    class FakeSettings:
        query_extension_min_similarity = 0.0
        trace_enabled = True
        trace_path = str(tmp_path / "low_relevance_traces.json")
        trace_retention_count = 10
        trace_include_text = False
        trace_text_chars = 300
        trace_max_items_per_stage = 20
        qdrant_collection = "personal_test"
        store_backend = "qdrant"
        corpus_manifest = "configs/personal_test_sources.yml"
        candidate_pool = 1
        dense_weight = 0.55
        bm25_weight = 0.45
        rrf_k = 60
        reranker_model = None
        rerank_top_n = 20
        inferred_metadata_filters_enabled = True
        query_rewrite_filter_min_confidence = 0.55

    class FakeStore:
        embedder = MappingEmbedder({"如何评价炫神": [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, hits: list[RetrievalHit]):
            raise AssertionError("LLM answerer should not run for low relevance hits")

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="answer_question",
                domain="personal_kb",
                query_type="evaluative",
                retrieval_strategy="hybrid",
                answer_source="personal_kb",
                kb_required=True,
                allow_model_fallback=False,
                confidence=0.9,
                priority_reason="rag",
            )

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (FakeSettings(), FakeStore(), FakeRetriever(), None, FakeRewriter()),
    )
    monkeypatch.setattr(api_module, "OpenAICompatibleAnswerer", FakeAnswerer)
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": "如何评价炫神",
            "top_k": 1,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "answer_question"
    assert data["query_type"] == "evaluative"
    assert data["retrieval_status"] == "no_relevant_evidence"
    assert data["hit_count"] == 0
    assert data["candidate_hit_count"] == 1
    assert data["relevant_hit_count"] == 0
    assert data["hits"] == []
    assert data["answer"]["reason"] == "no_relevant_evidence"
    assert data["answer"]["answer"] == "知识库没有找到与这个问题直接相关的材料，不能可靠回答。"


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
        model="qwen3.5:9b",
        max_queries=4,
    )

    assert result.applied is True
    assert result.retrieval_query.startswith("辛普森案 O.J. Simpson murder trial")
    assert result.search_queries == ["People v. Simpson", "O.J. Simpson criminal case"]
    assert result.aliases == ["Orenthal James Simpson", "People v. Simpson"]
    assert result.filters["doc_type"] == "case"
    assert result.confidence == 0.82


def test_query_rewrite_payload_drops_null_like_strings() -> None:
    result = result_from_model_payload(
        "颜色偏好",
        {
            "canonical_query": "null",
            "search_queries": ["favorite color", "none", ""],
            "aliases": ["null", "蓝色"],
            "filters": {},
            "confidence": 0.4,
        },
        backend="ollama",
        model="qwen3.5:9b",
        max_queries=4,
    )

    assert result.canonical_query is None
    assert result.search_queries == ["favorite color"]
    assert result.aliases == ["蓝色"]
    assert "null" not in result.retrieval_query.casefold()


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
        model="qwen3.5:9b",
    ).rewrite("notice")

    assert captured["json"]["think"] is False
    assert result.applied is True
    assert result.canonical_query == "notice rulemaking"


def test_answer_endpoint_model_fallbacks_for_bm25_only_general_definition(
    monkeypatch, tmp_path
) -> None:
    query = "灰度域和彩色域是什么"

    class FakeRetriever:
        candidate_pool = 1

        def retrieve(self, query: str, *, top_k: int, mode: str, filters=None, **kwargs):
            return [
                RetrievalHit(
                    chunk_id="1",
                    source_id="personal-test",
                    doc_type="note",
                    title="Personal Test Note",
                    citation="Personal Test Note v1",
                    jurisdiction="PERSONAL",
                    text="个人知识库助手会记录记忆分类、RAG 指标、ShowMaker 和炫神的个人测试描述。",
                    dense_score=0.0,
                    bm25_score=2.5,
                    fusion_score=0.05,
                )
            ]

    class FakeStore:
        embedder = MappingEmbedder({query: [1.0, 0.0]})

    class FakeRewriter:
        def rewrite(self, query: str):
            return QueryRewriteResult(original_query=query, retrieval_query=query)

    class FakeRouter:
        def route(self, query: str, **kwargs):
            return route_result(
                need_retrieval=True,
                intent="answer_question",
                domain="personal_kb",
                query_type="definition",
                retrieval_strategy="hybrid",
                answer_source="personal_kb",
                kb_required=False,
                allow_model_fallback=True,
                confidence=0.92,
                priority_reason="kb_first",
            )

    class FakeAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, hits: list[RetrievalHit]):
            raise AssertionError("grounded answerer should not run for irrelevant candidates")

    class FakeGeneralAnswerer:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, question: str, *, fallback_from_kb: bool = False, kb_required: bool = False):
            assert question == query
            assert fallback_from_kb is True
            assert kb_required is False
            return {
                "enabled": True,
                "skipped": False,
                "reason": "kb_no_relevant_evidence",
                "refused": False,
                "answer_status": "model_fallback",
                "answer": "个人知识库没有找到直接相关内容；根据一般常识，灰度域通常指只包含亮度信息的表示，彩色域通常指包含颜色通道的表示。",
                "citations": [],
                "grounding": {
                    "question_type": "definition",
                    "answerable": True,
                    "checked_hit_count": 0,
                    "required_evidence": "模型通用知识；不使用个人知识库引用",
                    "answer_mode": "general_knowledge_fallback",
                    "matched_cues": [],
                    "missing_evidence": ["personal_kb_evidence"],
                },
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": 1.0,
                "error": None,
            }

    import legal_rag.api as api_module

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (
            fake_settings(tmp_path, "bm25_only_general_definition_traces.json"),
            FakeStore(),
            FakeRetriever(),
            None,
            FakeRewriter(),
        ),
    )
    monkeypatch.setattr(api_module, "build_intent_router", lambda settings: FakeRouter())
    monkeypatch.setattr(api_module, "OpenAICompatibleAnswerer", FakeAnswerer)
    monkeypatch.setattr(api_module, "GeneralKnowledgeAnswerer", FakeGeneralAnswerer)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "query": query,
            "top_k": 1,
            "mode": "hybrid",
            "llm": {
                "provider": "openai_compatible",
                "base_url": "https://api.example.com/v1",
                "model": "example-model",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_status"] == "model_fallback"
    assert data["answer_source"] == "model"
    assert data["hit_count"] == 0
    assert data["candidate_hit_count"] == 1
    assert data["relevant_hit_count"] == 0
    assert data["hits"] == []
    assert data["answer"]["answer_status"] == "model_fallback"
    assert data["answer"]["citations"] == []
    assert "灰度" in data["answer"]["answer"]
    assert "彩色" in data["answer"]["answer"]


def test_frontend_is_served() -> None:
    client = TestClient(app)

    index_response = client.get("/")
    css_response = client.get("/static/styles.css")
    js_response = client.get("/static/app.js")

    assert index_response.status_code == 200
    assert "Legal RAG" in index_response.text
    assert "thinkingToggle" in index_response.text
    assert "Thinking" in index_response.text
    assert css_response.status_code == 200
    assert "docket-shell" in css_response.text
    assert "--composer-space" in css_response.text
    assert ".thinking-toggle" in css_response.text
    assert js_response.status_code == 200
    assert "thinking_enabled" in js_response.text
    assert "THINKING_LLM_TIMEOUT_SECONDS = 180" in js_response.text
    assert "timeout_seconds: answerTimeoutSeconds()" in js_response.text
    assert "setThinkingEnabled" in js_response.text
    assert "模型生成超时，已尝试关闭 Thinking 重试" in js_response.text
    assert "模型只返回了思考内容或空正文" in js_response.text
    assert "证据不足" in js_response.text
    assert "基于现有材料" in js_response.text
    assert "模型未能回答" in js_response.text
    assert "普通对话" in js_response.text
    assert "需要澄清" in js_response.text
    assert "待保存信息" in js_response.text
    assert "知识库没有找到直接相关材料" in js_response.text
    assert "个人库命中" in js_response.text
    assert "个人库未命中，模型通用知识回答" in js_response.text
    assert "严格知识库问题，证据不足" in js_response.text
    assert "Answer source" in js_response.text
    assert "KB required" in js_response.text
    assert "Model fallback" in js_response.text
    assert "个人库返回" in js_response.text
    assert "相关证据" in js_response.text
    assert "Candidates" in js_response.text
    assert "Relevant" in js_response.text
    assert "路由决策" in js_response.text
    assert "intent_confidence" in js_response.text
    assert 'pack.retrieval_status === "not_run"' in js_response.text
    assert "collectConversationContext" in js_response.text
    assert "updateComposerMetrics" in js_response.text
    assert 'scrollPageToLatest("auto")' in js_response.text
    assert 'function scrollPageToLatest(behavior = "smooth")' in js_response.text
    assert "scrollChatToBottom(behavior)" in js_response.text
