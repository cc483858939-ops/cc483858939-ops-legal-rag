from __future__ import annotations

from legal_rag.llm_answer import (
    AnswerabilityGate,
    LLMAnswerConfig,
    OpenAICompatibleAnswerer,
    build_support_spans,
    validate_grounded_claims,
)
from legal_rag.schema import RetrievalHit


def _hit(text: str, *, chunk_id: str = "1") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text=text,
        fusion_score=0.1,
    )


def test_support_spans_split_adjacent_alias_facts() -> None:
    spans = build_support_spans([_hit("甲方是选手。乙方又称小乙。")])

    assert [span.text for span in spans] == ["甲方是选手。", "乙方又称小乙。"]
    assert spans[0].span_id == "S1"
    assert spans[1].span_id == "S2"


def test_claim_validation_rejects_alias_bleed_from_adjacent_span() -> None:
    hit = _hit("甲方是选手。乙方又称小乙。")
    spans = build_support_spans([hit])
    grounding = AnswerabilityGate().evaluate("甲方是谁", [hit])

    checks = validate_grounded_claims("根据你的个人知识库，甲方又称小乙。[S1]", spans, grounding=grounding)

    assert checks
    assert checks[0].supported is False
    assert "又称" not in checks[0].missing_terms
    assert "小乙" in checks[0].missing_terms


def test_openai_answerer_retries_then_falls_back_on_unsupported_claim(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        return FakeResponse("根据你的个人知识库，甲方又称小乙。[S1]")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("甲方是谁", [_hit("甲方是选手。乙方又称小乙。")])

    assert len(calls) == 2
    assert result["reason"] == "unsupported_claim_fallback"
    assert result["answer"].startswith("根据你的个人知识库，甲方是选手。")
    assert result["grounding"]["unsupported_claims"] == []
    assert result["grounding"]["used_support_span_ids"] == ["S1"]
    assert [citation["support_span_id"] for citation in result["citations"]] == ["S1"]


def test_factual_question_refuses_when_required_scope_is_missing(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise AssertionError("LLM should not be called when required query scope is missing")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("王者荣耀的第一中单是谁", [_hit("ShowMaker 是世界第一中单。")])

    assert result["skipped"] is True
    assert result["refused"] is True
    assert result["answer_status"] == "refused"
    assert result["reason"] == "missing_required_query_scope"
    assert "王者荣耀" in result["grounding"]["missing_evidence"]


def test_citations_only_include_used_support_spans(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "根据你的个人知识库，炫神又被称为炫狗。[S2]"}}]}

    def fake_post(url, *, json, headers, timeout):
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("炫神别名是什么", [_hit("ShowMaker 是世界第一中单。炫神，又被称为炫狗。")])

    assert result["refused"] is False
    assert result["grounding"]["used_support_span_ids"] == ["S2"]
    assert [citation["support_span_id"] for citation in result["citations"]] == ["S2"]
