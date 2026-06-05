from __future__ import annotations

from personal_rag.query_intent import (
    DOMAIN_VALUES,
    QueryIntentRouter,
    RouteDecision,
    normalize_route_decision,
    parse_route_decision,
    route_decision_schema,
)


def test_domain_values_are_personal_only() -> None:
    assert DOMAIN_VALUES == ("personal_kb", "general_chat", "unknown")


def test_route_decision_schema_only_includes_personal_domains() -> None:
    domain_enum = route_decision_schema()["properties"]["domain"]["enum"]

    assert domain_enum == ["personal_kb", "general_chat", "unknown"]


def test_parse_route_decision_accepts_personal_kb_domain() -> None:
    decision = parse_route_decision(
        """
        {
          "need_retrieval": true,
          "intent": "answer_question",
          "domain": "personal_kb",
          "query_type": "factual",
          "retrieval_strategy": "hybrid",
          "answer_source": "personal_kb",
          "kb_required": false,
          "allow_model_fallback": true,
          "requires_clarification": false,
          "clarification_question": null,
          "confidence": 0.9
        }
        """
    )

    assert decision.domain == "personal_kb"


def test_strict_personal_kb_question_routes_to_kb_only() -> None:
    decision = RouteDecision(
        need_retrieval=True,
        intent="answer_question",
        domain="personal_kb",
        query_type="factual",
        retrieval_strategy="hybrid",
        answer_source="personal_kb",
        kb_required=False,
        allow_model_fallback=True,
        requires_clarification=False,
        clarification_question=None,
        confidence=0.9,
    )

    result = normalize_route_decision(decision, reranker_enabled=False, query="我的知识库里打火机是什么")

    assert result.need_retrieval is True
    assert result.answer_source == "personal_kb"
    assert result.kb_required is True
    assert result.allow_model_fallback is False
    assert result.priority_reason == "kb_only"


def test_general_model_decision_can_skip_retrieval() -> None:
    decision = RouteDecision(
        need_retrieval=False,
        intent="answer_question",
        domain="general_chat",
        query_type="procedural",
        retrieval_strategy="none",
        answer_source="model",
        kb_required=False,
        allow_model_fallback=False,
        requires_clarification=False,
        clarification_question=None,
        confidence=0.95,
    )

    result = normalize_route_decision(
        decision,
        reranker_enabled=False,
        query="根据通用知识回答我，心肺复苏的基本流程是什么",
    )

    assert result.need_retrieval is False
    assert result.answer_source == "model"
    assert result.priority_reason == "direct_model"


def test_disabled_router_falls_back_to_personal_kb() -> None:
    result = QueryIntentRouter().route("ShowMaker是谁")

    assert result.need_retrieval is True
    assert result.answer_source == "personal_kb"
    assert result.priority_reason == "router_disabled"
