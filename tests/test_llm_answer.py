from __future__ import annotations

import httpx

from personal_rag.llm_answer import (
    GeneralKnowledgeAnswerer,
    LLMAnswerConfig,
    OpenAICompatibleAnswerer,
    _extract_answer_text,
    answer_result_indicates_insufficient_evidence,
)
from personal_rag.schema import RetrievalHit


def hit(text: str = "ShowMaker is described as the world first mid laner.") -> RetrievalHit:
    return RetrievalHit(
        chunk_id="chunk-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        source_ref="Personal Test Note v1",
        text=text,
        fusion_score=0.2,
    )


def test_extract_answer_text_supports_string_and_content_parts_without_reasoning() -> None:
    assert _extract_answer_text({"choices": [{"message": {"content": "final answer"}}]}) == "final answer"
    assert (
        _extract_answer_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "part one"},
                                {"type": "text", "text": "part two"},
                            ],
                            "reasoning_content": "hidden thinking",
                        }
                    }
                ]
            }
        )
        == "part one part two"
    )
    assert (
        _extract_answer_text(
            {"choices": [{"message": {"content": "", "reasoning_content": "thinking only"}}]}
        )
        == ""
    )


def test_answer_result_error_does_not_indicate_insufficient_evidence() -> None:
    assert not answer_result_indicates_insufficient_evidence(
        {"reason": "llm_error", "answer_status": "error", "answer": "insufficient evidence"}
    )


def test_openai_compatible_answerer_uses_chunk_sources(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ShowMaker is the world first mid laner. [1]"}}],
                "usage": {"total_tokens": 12},
            }

    def fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse()

    import personal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
            api_key="secret",
        )
    ).generate("ShowMaker是谁", [hit()])

    user_prompt = captured["json"]["messages"][1]["content"]
    assert "Source: Personal Test Note v1" in user_prompt
    assert "Citation:" not in user_prompt
    assert result["answer_status"] == "answered"
    assert result["sources"][0]["source_ref"] == "Personal Test Note v1"
    assert "secret" not in str(result)


def test_ollama_openai_compatible_answerer_disables_thinking_by_default(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "grounded answer [1]"}}]}

    def fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import personal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
        )
    ).generate("打火机是什么", [hit("炫神最喜欢的歌是打火机。")])

    assert captured["json"]["reasoning_effort"] == "none"


def test_thinking_empty_content_retries_without_thinking(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, content: str, *, reasoning: str = "") -> None:
            self.content = content
            self.reasoning = reasoning

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": self.content,
                            "reasoning_content": self.reasoning,
                        }
                    }
                ]
            }

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        if len(calls) == 1:
            return FakeResponse("", reasoning="thinking only")
        return FakeResponse("推荐钢之炼金术师、星际牛仔、命运石之门。")

    import personal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = GeneralKnowledgeAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
            thinking_enabled=True,
        )
    ).generate("有什么动漫推荐", fallback_from_kb=True)

    assert "reasoning_effort" not in calls[0]
    assert calls[1]["reasoning_effort"] == "none"
    assert result["retry_reason"] == "empty_final_content"
    assert result["recovered_from_thinking"] is True
    assert result["sources"] == []


def test_thinking_timeout_retries_without_thinking(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "fallback answer"}}]}

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return FakeResponse()

    import personal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = GeneralKnowledgeAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
            thinking_enabled=True,
        )
    ).generate("有什么动漫推荐", fallback_from_kb=True)

    assert calls[1]["reasoning_effort"] == "none"
    assert result["retry_reason"] == "thinking_read_timeout"
    assert result["answer_status"] == "model_fallback"
