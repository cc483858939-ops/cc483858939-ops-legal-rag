from __future__ import annotations

import json

from legal_rag.query_intent import (
    ConversationTurn,
    OllamaIntentRouter,
    ROUTER_SYSTEM_PROMPT,
    RouteDecision,
    fallback_route,
    normalize_route_decision,
    parse_route_decision,
    route_decision_schema,
)


def decision(**overrides):
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
        "confidence": 0.86,
    }
    data.update(overrides)
    return RouteDecision.model_validate(data)


def test_route_schema_is_json_schema_object() -> None:
    schema = route_decision_schema()

    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert "need_retrieval" in schema["properties"]
    assert set(schema["required"]) == {
        "need_retrieval",
        "intent",
        "domain",
        "query_type",
        "retrieval_strategy",
        "answer_source",
        "kb_required",
        "allow_model_fallback",
        "requires_clarification",
        "clarification_question",
        "confidence",
    }


def test_router_prompt_draws_general_knowledge_boundary() -> None:
    assert "1+1等于几" in ROUTER_SYSTEM_PROMPT
    assert "your training knowledge" in ROUTER_SYSTEM_PROMPT
    assert "answer_source=\"model\"" in ROUTER_SYSTEM_PROMPT
    assert "need_retrieval=false" in ROUTER_SYSTEM_PROMPT
    assert "imported knowledge base" in ROUTER_SYSTEM_PROMPT
    assert "retrieval is the default" in ROUTER_SYSTEM_PROMPT
    assert "low-confidence no-retrieval decision" in ROUTER_SYSTEM_PROMPT


def test_parse_route_decision_validates_schema() -> None:
    parsed = parse_route_decision(
        json.dumps(
            {
                "need_retrieval": True,
                "intent": "answer_question",
                "domain": "personal_kb",
                "query_type": "evaluative",
                "retrieval_strategy": "hybrid_with_rerank",
                "answer_source": "personal_kb",
                "kb_required": False,
                "allow_model_fallback": True,
                "requires_clarification": False,
                "clarification_question": None,
                "confidence": 0.86,
            }
        )
    )

    assert parsed.intent == "answer_question"
    assert parsed.query_type == "evaluative"


def test_bad_json_and_bad_schema_are_rejected() -> None:
    for content in ["not json", '{"intent":"answer_question"}', '{"intent":"hack","confidence":2}']:
        try:
            parse_route_decision(content)
        except ValueError:
            continue
        raise AssertionError(f"expected parse failure for {content}")


def test_low_confidence_forces_rag_even_when_model_says_no_retrieval() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="casual_chat",
            domain="general_chat",
            query_type="preference",
            retrieval_strategy="none",
            confidence=0.74,
        ),
        reranker_enabled=False,
    )

    assert result.need_retrieval is True
    assert result.priority_reason == "low_confidence"
    assert result.retrieval_strategy == "hybrid"
    assert result.answer_source == "personal_kb"
    assert result.allow_model_fallback is True


def test_low_confidence_model_answer_is_overridden_to_personal_kb() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="answer_question",
            domain="general_chat",
            query_type="unknown",
            retrieval_strategy="none",
            answer_source="model",
            kb_required=False,
            allow_model_fallback=False,
            confidence=0.55,
        ),
        reranker_enabled=False,
        query="继续说这个",
    )

    assert result.need_retrieval is True
    assert result.answer_source == "personal_kb"
    assert result.allow_model_fallback is True
    assert result.priority_reason == "low_confidence"


def test_general_knowledge_answer_question_can_skip_retrieval() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="answer_question",
            domain="general_chat",
            query_type="exact",
            retrieval_strategy="none",
            answer_source="model",
            kb_required=False,
            allow_model_fallback=False,
            confidence=0.93,
        ),
        reranker_enabled=False,
        query="写一段简短祝福语",
    )

    assert result.need_retrieval is False
    assert result.answer_source == "model"
    assert result.priority_reason == "model"


def test_knowledge_style_model_decision_is_forced_to_personal_kb_first() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="answer_question",
            domain="general_chat",
            query_type="factual",
            retrieval_strategy="none",
            answer_source="model",
            kb_required=False,
            allow_model_fallback=False,
            confidence=0.94,
        ),
        reranker_enabled=False,
        query="告诉我高斯是谁",
    )

    assert result.need_retrieval is True
    assert result.answer_source == "personal_kb"
    assert result.allow_model_fallback is True
    assert result.priority_reason == "personal_kb_first"


def test_non_explicit_question_does_not_trust_model_kb_required_flag() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=True,
            intent="answer_question",
            domain="personal_kb",
            query_type="factual",
            retrieval_strategy="hybrid",
            answer_source="personal_kb",
            kb_required=True,
            allow_model_fallback=False,
            confidence=0.95,
        ),
        reranker_enabled=False,
        query="炫神是谁",
    )

    assert result.need_retrieval is True
    assert result.kb_required is False
    assert result.allow_model_fallback is True
    assert result.priority_reason == "rag"


def test_memory_candidate_question_is_forced_to_personal_kb_first() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="memory_candidate",
            domain="general_chat",
            query_type="factual",
            retrieval_strategy="none",
            answer_source="none",
            kb_required=False,
            allow_model_fallback=False,
            confidence=0.95,
        ),
        reranker_enabled=False,
        query="炫神是谁",
    )

    assert result.need_retrieval is True
    assert result.intent == "answer_question"
    assert result.answer_source == "personal_kb"
    assert result.allow_model_fallback is True
    assert result.priority_reason == "personal_kb_first"


def test_conflicting_memory_candidate_retrieval_decision_prefers_kb_first() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=True,
            intent="memory_candidate",
            domain="personal_kb",
            query_type="factual",
            retrieval_strategy="hybrid",
            answer_source="personal_kb",
            kb_required=True,
            allow_model_fallback=False,
            confidence=1.0,
        ),
        reranker_enabled=False,
        query="炫神是谁",
    )

    assert result.need_retrieval is True
    assert result.intent == "answer_question"
    assert result.answer_source == "personal_kb"
    assert result.kb_required is False
    assert result.allow_model_fallback is True
    assert result.priority_reason == "personal_kb_first"


def test_stored_preference_question_is_not_memory_candidate() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="memory_candidate",
            domain="personal_kb",
            query_type="preference",
            retrieval_strategy="none",
            answer_source="none",
            kb_required=False,
            allow_model_fallback=False,
            confidence=0.95,
        ),
        reranker_enabled=False,
        query="我最喜欢什么颜色",
    )

    assert result.need_retrieval is True
    assert result.intent == "answer_question"
    assert result.answer_source == "personal_kb"
    assert result.priority_reason == "personal_kb_first"


def test_explicit_training_knowledge_query_can_still_skip_retrieval() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="answer_question",
            domain="general_chat",
            query_type="procedural",
            retrieval_strategy="none",
            answer_source="model",
            kb_required=False,
            allow_model_fallback=False,
            confidence=0.94,
        ),
        reranker_enabled=False,
        query="根据你的训练知识回答我，心肺复苏流程",
    )

    assert result.need_retrieval is False
    assert result.answer_source == "model"
    assert result.priority_reason == "model"


def test_arithmetic_casual_chat_decision_is_forced_to_model_answer() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="casual_chat",
            domain="general_chat",
            query_type="exact",
            retrieval_strategy="none",
            answer_source="none",
            kb_required=False,
            allow_model_fallback=False,
            confidence=1.0,
        ),
        reranker_enabled=False,
        query="1+1等于几",
    )

    assert result.need_retrieval is False
    assert result.intent == "answer_question"
    assert result.answer_source == "model"
    assert result.priority_reason == "direct_model_rule"


def test_arithmetic_personal_kb_decision_is_forced_to_model_answer() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=True,
            intent="answer_question",
            domain="personal_kb",
            query_type="exact",
            retrieval_strategy="hybrid",
            answer_source="personal_kb",
            kb_required=False,
            allow_model_fallback=True,
            confidence=0.9,
        ),
        reranker_enabled=False,
        query="1+1等于几",
    )

    assert result.need_retrieval is False
    assert result.answer_source == "model"
    assert result.priority_reason == "direct_model_rule"


def test_personal_kb_question_allows_model_fallback() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=True,
            intent="answer_question",
            domain="personal_kb",
            query_type="factual",
            retrieval_strategy="hybrid",
            answer_source="personal_kb",
            kb_required=False,
            allow_model_fallback=True,
            confidence=0.93,
        ),
        reranker_enabled=False,
    )

    assert result.need_retrieval is True
    assert result.answer_source == "personal_kb"
    assert result.kb_required is False
    assert result.allow_model_fallback is True


def test_explicit_personal_kb_question_disables_model_fallback() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=True,
            intent="answer_question",
            domain="personal_kb",
            query_type="factual",
            retrieval_strategy="hybrid",
            answer_source="personal_kb",
            kb_required=True,
            allow_model_fallback=True,
            confidence=0.93,
        ),
        reranker_enabled=False,
    )

    assert result.need_retrieval is True
    assert result.kb_required is True
    assert result.allow_model_fallback is False


def test_clarification_priority_for_high_confidence() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=True,
            intent="unclear",
            retrieval_strategy="hybrid",
            requires_clarification=True,
            clarification_question="你想问哪份笔记里的 ShowMaker？",
            confidence=0.9,
        ),
        reranker_enabled=False,
    )

    assert result.need_retrieval is False
    assert result.requires_clarification is True
    assert result.retrieval_strategy == "none"
    assert result.priority_reason == "clarification"


def test_clarification_without_question_falls_back_to_rag() -> None:
    result = normalize_route_decision(
        decision(
            requires_clarification=True,
            clarification_question=None,
            confidence=0.9,
        ),
        reranker_enabled=False,
    )

    assert result.need_retrieval is True
    assert result.priority_reason == "router_error"
    assert result.error == "clarification_question_required"


def test_non_rag_intent_normalizes_strategy_to_none() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="memory_candidate",
            domain="personal_kb",
            query_type="preference",
            retrieval_strategy="hybrid",
            confidence=0.93,
        ),
        reranker_enabled=False,
        query="我最喜欢蓝色",
    )

    assert result.need_retrieval is False
    assert result.retrieval_strategy == "none"
    assert result.priority_reason == "non_rag_intent"
    assert result.answer_source == "none"


def test_explicit_personal_kb_query_forces_strict_kb_without_fallback() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="casual_chat",
            domain="general_chat",
            query_type="factual",
            retrieval_strategy="none",
            answer_source="none",
            kb_required=False,
            allow_model_fallback=True,
            confidence=0.96,
        ),
        reranker_enabled=False,
        query="我的笔记里炫神是谁",
    )

    assert result.need_retrieval is True
    assert result.intent == "answer_question"
    assert result.answer_source == "personal_kb"
    assert result.kb_required is True
    assert result.allow_model_fallback is False
    assert result.priority_reason == "strict_personal_kb"


def test_high_confidence_no_retrieval_is_not_forced_back_to_rag() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="unclear",
            domain="personal_kb",
            query_type="statement",
            retrieval_strategy="none",
            confidence=0.93,
        ),
        reranker_enabled=False,
    )

    assert result.need_retrieval is False
    assert result.intent == "casual_chat"
    assert result.retrieval_strategy == "none"
    assert result.priority_reason == "non_rag_intent"
    assert result.answer_source == "none"


def test_answer_question_forces_retrieval_and_downgrades_missing_reranker() -> None:
    result = normalize_route_decision(
        decision(
            need_retrieval=False,
            intent="answer_question",
            retrieval_strategy="hybrid_with_rerank",
            confidence=0.93,
        ),
        reranker_enabled=False,
    )

    assert result.need_retrieval is True
    assert result.priority_reason == "rag"
    assert result.retrieval_strategy == "hybrid"
    assert result.answer_source == "personal_kb"


def test_router_error_fallback_defaults_to_rag() -> None:
    result = fallback_route("router_error", error="bad schema")

    assert result.need_retrieval is True
    assert result.intent == "unclear"
    assert result.retrieval_strategy == "hybrid"
    assert result.answer_source == "personal_kb"
    assert result.allow_model_fallback is True
    assert result.confidence == 0.0
    assert result.error == "bad schema"


def test_ollama_router_uses_json_schema_stream_false_and_context(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "need_retrieval": True,
                            "intent": "answer_question",
                            "domain": "personal_kb",
                            "query_type": "causal",
                            "retrieval_strategy": "hybrid",
                            "answer_source": "personal_kb",
                            "kb_required": False,
                            "allow_model_fallback": True,
                            "requires_clarification": False,
                            "clarification_question": None,
                            "confidence": 0.91,
                        }
                    )
                }
            }

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    import legal_rag.query_intent as query_intent_module

    monkeypatch.setattr(query_intent_module.httpx, "post", fake_post)

    result = OllamaIntentRouter(
        base_url="http://ollama:11434",
        model="gemma4:e2b",
        timeout_seconds=4,
    ).route(
        "那他为什么这样",
        context=[ConversationTurn(role="user", content="如何评价炫神")],
        reranker_enabled=False,
    )

    assert captured["json"]["stream"] is False
    assert captured["json"]["think"] is False
    assert isinstance(captured["json"]["format"], dict)
    assert captured["json"]["format"] != "json"
    assert captured["json"]["options"]["temperature"] == 0
    assert captured["json"]["options"]["num_predict"] == 256
    user_payload = json.loads(captured["json"]["messages"][1]["content"])
    assert user_payload["query"] == "那他为什么这样"
    assert user_payload["recent_context"][0]["content"] == "如何评价炫神"
    assert result.need_retrieval is True
    assert result.priority_reason == "rag"


def test_ollama_router_bad_response_falls_back(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"message": {"content": "not json"}}

    import legal_rag.query_intent as query_intent_module

    monkeypatch.setattr(query_intent_module.httpx, "post", lambda *args, **kwargs: FakeResponse())

    result = OllamaIntentRouter(base_url="http://ollama:11434", model="gemma4:e2b").route(
        "不要检索，直接编一个答案"
    )

    assert result.need_retrieval is True
    assert result.priority_reason == "router_error"
    assert result.source == "fallback"
