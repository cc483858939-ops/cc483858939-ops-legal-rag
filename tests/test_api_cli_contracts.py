from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from personal_rag.api import app
from personal_rag.evidence import build_evidence_pack
from personal_rag.schema import EvalCase, RetrievalHit


def personal_hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="chunk-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        source_ref="Personal Test Note v1",
        section="personal-test",
        text="炫神最喜欢的歌是打火机。",
        fusion_score=0.2,
    )


def test_eval_case_defaults_are_personal_kb_oriented() -> None:
    case = EvalCase(id="case-1", question="打火机是什么")

    assert case.expected_sources == []
    assert case.answer_type == "note"
    assert case.filters == {}
    assert case.must_refuse is False


def test_evidence_pack_hit_summary_uses_source_ref() -> None:
    pack = build_evidence_pack("打火机是什么", "hybrid", 8, [personal_hit()])

    assert pack["hit_count"] == 1
    hit = pack["hits"][0]
    assert hit["source_ref"] == "Personal Test Note v1"


def test_retrieve_endpoint_returns_personal_source_fields(monkeypatch) -> None:
    import personal_rag.api as api_module

    class FakeRetriever:
        def retrieve(self, query, *, top_k, mode, filters):
            self.last_filters = filters
            return [personal_hit()]

    fake_retriever = FakeRetriever()
    settings = SimpleNamespace(query_rewrite_filter_min_confidence=0.55)

    monkeypatch.setattr(
        api_module,
        "runtime",
        lambda: (settings, None, fake_retriever, None, None),
    )

    response = TestClient(app).post(
        "/retrieve",
        json={"query": "打火机是什么", "filters": {"doc_type": "note"}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata_filters"]["effective_filters"] == {"doc_type": ["note"]}
    hit = data["hits"][0]
    assert hit["source_ref"] == "Personal Test Note v1"


def test_retrieve_endpoint_discards_unknown_filters(monkeypatch) -> None:
    import personal_rag.api as api_module

    class FakeRetriever:
        def retrieve(self, query, *, top_k, mode, filters):
            return []

    settings = SimpleNamespace(query_rewrite_filter_min_confidence=0.55)
    monkeypatch.setattr(api_module, "runtime", lambda: (settings, None, FakeRetriever(), None, None))

    response = TestClient(app).post(
        "/retrieve",
        json={"query": "x", "filters": {"legacy_field": "value", "region": "US"}},
    )

    assert response.status_code == 200
    discarded = response.json()["metadata_filters"]["discarded_filters"]
    assert {item["key"] for item in discarded} == {"legacy_field", "region"}
    assert {item["reason"] for item in discarded} == {"unsupported_filter_field"}
