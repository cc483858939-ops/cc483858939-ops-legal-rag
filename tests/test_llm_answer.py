from __future__ import annotations

from legal_rag.llm_answer import (
    AnswerabilityGate,
    GeneralKnowledgeAnswerer,
    LLMAnswerConfig,
    OpenAICompatibleAnswerer,
    _extract_answer_text,
    answer_result_indicates_insufficient_evidence,
    build_extractive_answer,
    build_support_spans,
    classify_question_type,
    extract_query_terms,
    refine_grounding_for_answer_units,
    select_support_spans,
    validate_grounded_claims,
)
from legal_rag.schema import RetrievalHit


def test_answer_result_indicates_insufficient_evidence_from_reason_status_and_text() -> None:
    assert answer_result_indicates_insufficient_evidence(
        {"reason": "llm_refused", "answer_status": "answered", "answer": "unused"}
    )
    assert answer_result_indicates_insufficient_evidence(
        {"reason": "insufficient_causal_evidence", "answer_status": "answered", "answer": "unused"}
    )
    assert answer_result_indicates_insufficient_evidence(
        {"reason": "insufficient_exact_evidence", "answer_status": "answered", "answer": "unused"}
    )
    assert answer_result_indicates_insufficient_evidence(
        {"reason": None, "answer_status": "llm_refused", "answer": "unused"}
    )
    assert answer_result_indicates_insufficient_evidence(
        {"reason": None, "answer_status": "answered", "answer": "根据现有材料来看，没有关于爱因斯坦的信息。"}
    )
    assert answer_result_indicates_insufficient_evidence(
        {"reason": None, "answer_status": "answered", "answer": "There is insufficient evidence to answer."}
    )
    assert answer_result_indicates_insufficient_evidence(
        {"reason": None, "answer_status": "answered", "answer": "现有材料主要涉及 RAG 指标和人物描述，并未提及灰度域或彩色域。"}
    )


def test_answer_result_error_does_not_indicate_insufficient_evidence() -> None:
    assert not answer_result_indicates_insufficient_evidence(
        {"reason": "llm_error", "answer_status": "error", "answer": "insufficient evidence"}
    )


def test_extract_answer_text_supports_string_and_content_parts_without_reasoning() -> None:
    assert (
        _extract_answer_text({"choices": [{"message": {"content": "最终回答"}}]})
        == "最终回答"
    )
    assert (
        _extract_answer_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "第一段"},
                                {"type": "text", "text": "第二段"},
                            ],
                            "reasoning_content": "隐藏思考",
                        }
                    }
                ]
            }
        )
        == "第一段 第二段"
    )
    assert (
        _extract_answer_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "只有思考，没有最终回答",
                        }
                    }
                ]
            }
        )
        == ""
    )


def test_openai_compatible_answerer_posts_grounded_prompt(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "退款通常在验收通过后 7 个工作日内完成。[1]"}}],
                "usage": {"total_tokens": 42},
            }

    def fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="support-returns",
        doc_type="policy",
        title="Returns and Refunds Policy",
        citation="Support KB Returns and Refunds v2026.05",
        jurisdiction="GLOBAL",
        text="Refunds are issued within 7 business days after inspection approval.",
        fusion_score=0.1,
    )
    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1/",
            model="example-model",
            api_key="secret",
        )
    ).generate("退款多久到账", [hit])

    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["model"] == "example-model"
    assert "Evidence" in captured["json"]["messages"][1]["content"]
    assert "why/reason/causal questions" in captured["json"]["messages"][0]["content"]
    assert result["answer"].startswith("退款通常")
    assert result["answer_status"] == "answered"
    assert result["refused"] is False
    assert result["grounding"]["answerable"] is True
    assert result["usage"]["total_tokens"] == 42
    assert result["llm"]["api_key_configured"] is True
    assert "secret" not in str(result)


def test_ollama_openai_compatible_answerer_disables_thinking(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "根据材料回答。 [1]"}}]}

    def fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="炫神最喜欢的歌是打火机。",
        fusion_score=0.1,
    )
    OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
        )
    ).generate("打火机是什么", [hit])

    assert captured["url"] == "http://ollama:11434/v1/chat/completions"
    assert captured["json"]["reasoning_effort"] == "none"


def test_ollama_openai_compatible_answerer_allows_thinking_when_enabled(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "根据材料回答。 [1]"}}]}

    def fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="炫神是一个电竞主播。",
        fusion_score=0.1,
    )
    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
            thinking_enabled=True,
        )
    ).generate("炫神是谁", [hit])

    assert captured["url"] == "http://ollama:11434/v1/chat/completions"
    assert "reasoning_effort" not in captured["json"]
    assert result["llm"]["thinking_enabled"] is True


def test_ollama_answerer_retries_empty_thinking_response_without_thinking(monkeypatch) -> None:
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
                ],
                "usage": {"total_tokens": 18},
            }

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        if len(calls) == 1:
            return FakeResponse("", reasoning="先思考动漫推荐")
        return FakeResponse("推荐《钢之炼金术师 FA》。 [1]")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="anime-note",
        doc_type="note",
        title="Anime Note",
        citation="Anime Note v1",
        jurisdiction="PERSONAL",
        text="动漫推荐可以参考钢之炼金术师。",
        fusion_score=0.1,
    )
    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
            thinking_enabled=True,
        )
    ).generate("有什么动漫推荐", [hit])

    assert len(calls) == 2
    assert "reasoning_effort" not in calls[0]
    assert calls[1]["reasoning_effort"] == "none"
    assert result["answer"] == "推荐《钢之炼金术师 FA》。 [1]"
    assert result["retry_count"] == 1
    assert result["retry_reason"] == "empty_final_content"
    assert result["recovered_from_thinking"] is True


def test_ollama_answerer_retries_read_timeout_without_thinking(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "根据材料回答。 [1]"}}]}

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        if len(calls) == 1:
            import httpx

            raise httpx.ReadTimeout("timed out")
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="炫神是一个电竞主播。",
        fusion_score=0.1,
    )
    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
            thinking_enabled=True,
        )
    ).generate("炫神是谁", [hit])

    assert len(calls) == 2
    assert "reasoning_effort" not in calls[0]
    assert calls[1]["reasoning_effort"] == "none"
    assert result["retry_count"] == 1
    assert result["retry_reason"] == "thinking_read_timeout"
    assert result["recovered_from_thinking"] is True
    assert result["answer_status"] == "answered"


def test_extract_query_terms_removes_chinese_factual_question_cue() -> None:
    assert extract_query_terms("炫神是谁") == ["炫神"]
    assert extract_query_terms("高斯是谁") == ["高斯"]


def test_extract_query_terms_splits_personal_kb_relation_queries() -> None:
    assert extract_query_terms("炫神的父亲有哪些") == ["炫神", "父亲"]
    assert extract_query_terms("大司马和炫神是什么关系") == ["大司马", "炫神"]
    assert extract_query_terms("我的笔记里炫神是谁") == ["炫神"]
    assert extract_query_terms("那他是谁") == []


def test_default_answerability_allows_grounded_definition_factual_and_other_questions() -> None:
    hit = RetrievalHit(
        chunk_id="personal-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text=(
            "炫神最喜欢的歌是打火机，因为其中的一句歌词是 吉隆坡的天气，他是翻云又覆雨。"
            "炫神，又被称为炫狗。电棍是他的兄弟。"
            "长期事实记忆适合保存稳定偏好。短期会话记忆用于维持当前对话上下文。"
        ),
        fusion_score=0.1,
    )
    cases = [
        ("打火机歌词是什么", "definition"),
        ("吉隆坡天气那句歌词后面是什么", "definition"),
        ("炫神最爱的歌叫啥", "definition"),
        ("炫神别名是什么", "definition"),
        ("电棍跟炫神什么关系", "factual"),
        ("炫神最孝顺谁", "factual"),
        ("用户稳定偏好应该记到哪里", "factual"),
        ("当前对话上下文靠哪种记忆维持", "factual"),
        ("个人知识库助手比智能客服多做什么", "other"),
    ]

    gate = AnswerabilityGate()
    for question, question_type in cases:
        result = gate.evaluate(question, [hit])

        assert result.question_type == question_type
        assert result.answerable is True
        expected_mode = (
            "contextual_definition"
            if question_type == "definition" and result.matched_cues
            else "direct"
        )
        assert result.answer_mode == expected_mode
        assert result.answer_status == "answered"
        assert result.missing_evidence == []
        assert result.reason is None


def test_definition_answerability_refuses_when_no_entity_context_matches() -> None:
    hit = RetrievalHit(
        chunk_id="personal-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="个人知识库助手会记录记忆分类、RAG 指标、ShowMaker 和炫神的个人测试描述。",
        fusion_score=0.1,
    )

    result = AnswerabilityGate().evaluate("灰度域和彩色域是什么", [hit])

    assert result.question_type == "definition"
    assert result.answerable is False
    assert result.answer_mode == "refuse"
    assert result.answer_status == "refused"
    assert result.reason == "no_evidence"
    assert result.missing_evidence == ["entity_context"]


def test_contextual_definition_refusal_retries_with_personal_kb_meaning(monkeypatch) -> None:
    calls: list[dict] = []
    replies = [
        "根据现有材料来看，没有关于“打火机”的定义。",
        "根据你的个人知识库，打火机被描述为炫神最喜欢的歌。[1]",
    ]

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": self.content}}],
                "usage": {"total_tokens": 42},
            }

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        return FakeResponse(replies[len(calls) - 1])

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="personal-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="炫神最喜欢的歌是打火机，因为其中的一句歌词是 吉隆坡的天气，他是翻云又覆雨。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("打火机是什么", [hit])

    assert len(calls) == 2
    assert "contextual_definition questions" in calls[0]["messages"][0]["content"]
    assert "requiring an encyclopedia definition" in calls[1]["messages"][0]["content"]
    assert result["retry_count"] == 1
    assert result["answer_status"] == "answered"
    assert result["grounding"]["answer_mode"] == "contextual_definition"
    assert result["grounding"]["matched_cues"] == ["打火机"]
    assert "炫神" in result["answer"]
    assert "最喜欢的歌" in result["answer"]
    assert "打火机" in result["answer"]
    assert result["citations"]


def test_contextual_definition_uses_extractive_answer_when_llm_keeps_refusing(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "根据现有材料来看，没有关于“打火机”的定义。"}}
                ],
                "usage": {"total_tokens": 42},
            }

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="personal-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="炫神最喜欢的歌是打火机，因为其中的一句歌词是 吉隆坡的天气，他是翻云又覆雨。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("打火机是什么", [hit])

    assert len(calls) == 2
    assert result["reason"] == "contextual_definition_extractive"
    assert result["answer_status"] == "answered"
    assert result["grounding"]["answer_mode"] == "contextual_definition"
    assert "炫神" in result["answer"]
    assert "最喜欢的歌" in result["answer"]
    assert "打火机" in result["answer"]
    assert result["citations"]


def test_exact_contextual_definition_prefers_tight_relation_over_neighboring_facts(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "根据你的个人知识库，打火机是炫神最喜欢的歌，因为妹妹名字带雨、妈妈名字带云。[1]"
                        }
                    }
                ],
                "usage": {"total_tokens": 42},
            }

    def fake_post(*args, **kwargs):
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="personal-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="炫神最喜欢的歌是打火机，因为其中的一句歌词是 吉隆坡的天气，他是翻云又覆雨。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("打火机是什么", [hit])

    assert result["reason"] == "contextual_definition_extractive"
    assert result["answer"] == "根据你的个人知识库，打火机是炫神最喜欢的歌 [S1]"
    assert "妹妹" not in result["answer"]
    assert "妈妈" not in result["answer"]


def test_support_spans_split_adjacent_alias_facts_without_splitting_lists() -> None:
    adjacent = build_support_spans([
        RetrievalHit(
            chunk_id="1",
            source_id="synthetic",
            doc_type="note",
            title="Synthetic Note",
            citation="Synthetic Note v1",
            jurisdiction="PERSONAL",
            text="甲方是选手，乙方又称小乙。他的父亲包括：张三,李四,王五。",
            fusion_score=0.1,
        )
    ])

    assert [span.text for span in adjacent] == [
        "甲方是选手",
        "乙方又称小乙。",
        "他的父亲包括：张三,李四,王五。",
    ]


def test_support_span_selection_expands_same_subject_detail_without_adjacent_object_fact() -> None:
    hit = RetrievalHit(
        chunk_id="1",
        source_id="synthetic",
        doc_type="note",
        title="Synthetic Note",
        citation="Synthetic Note v1",
        jurisdiction="PERSONAL",
        text=(
            "ShowMaker是世界第一中单，炫狗是他的儿子。"
            "ShowMaker的圣经是重铸LCK荣光我辈义不容辞。"
            "他的父亲包括：Faker,ShowMaker,Theshy。"
            "他对ShowMaker最孝顺。"
        ),
        fusion_score=0.1,
    )
    grounding = AnswerabilityGate().evaluate("英雄联盟世界第一中单是谁，他有什么成就", [hit])

    spans = select_support_spans("英雄联盟世界第一中单是谁，他有什么成就", [hit], grounding=grounding)
    texts = [span.text for span in spans]

    assert "ShowMaker是世界第一中单" in texts
    assert "ShowMaker的圣经是重铸LCK荣光我辈义不容辞。" in texts
    assert not any("炫狗是他的儿子" in text for text in texts)
    assert not any("他对ShowMaker最孝顺" in text for text in texts)


def test_relation_question_prefers_relation_span_over_alias_span() -> None:
    hit = RetrievalHit(
        chunk_id="1",
        source_id="synthetic",
        doc_type="note",
        title="Synthetic Note",
        citation="Synthetic Note v1",
        jurisdiction="PERSONAL",
        text="炫神，又被称为炫狗。电棍是他的兄弟。",
        fusion_score=0.1,
    )
    question = "电棍跟炫神什么关系"
    grounding = AnswerabilityGate().evaluate(question, [hit])

    spans = select_support_spans(question, [hit], grounding=grounding)

    assert spans[0].text == "电棍是他的兄弟。"


def test_compound_question_marks_missing_answer_unit_as_partial() -> None:
    hit = RetrievalHit(
        chunk_id="1",
        source_id="synthetic",
        doc_type="note",
        title="Synthetic Note",
        citation="Synthetic Note v1",
        jurisdiction="PERSONAL",
        text="甲方是世界第一中单。",
        fusion_score=0.1,
    )
    question = "世界第一中单是谁，他有什么成就"
    grounding = AnswerabilityGate().evaluate(question, [hit])
    spans = select_support_spans(question, [hit], grounding=grounding)

    refined = refine_grounding_for_answer_units(question, grounding, spans)

    assert refined.answer_status == "partial"
    assert refined.answer_mode == "partial_compound"


def test_extractive_fallback_uses_same_subject_detail_span_for_partial_compound() -> None:
    hit = RetrievalHit(
        chunk_id="1",
        source_id="synthetic",
        doc_type="note",
        title="Synthetic Note",
        citation="Synthetic Note v1",
        jurisdiction="PERSONAL",
        text=(
            "ShowMaker是世界第一中单，炫狗是他的儿子。"
            "ShowMaker的圣经是重铸LCK荣光我辈义不容辞。"
        ),
        fusion_score=0.1,
    )
    question = "英雄联盟世界第一中单是谁，他有什么成就"
    grounding = AnswerabilityGate().evaluate(question, [hit])
    spans = select_support_spans(question, [hit], grounding=grounding)
    refined = refine_grounding_for_answer_units(question, grounding, spans)

    answer, used_ids = build_extractive_answer(question, spans, refined)

    assert used_ids == ["S1", "S3"]
    assert "ShowMaker是世界第一中单" in answer
    assert "重铸LCK荣光我辈义不容辞" in answer
    assert "材料不足以完整回答全部问题" in answer
    assert "炫狗是他的儿子" not in answer


def test_partial_compound_answer_uses_extractive_path_without_llm(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise AssertionError("partial compound answers should use extractive support spans")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="synthetic",
        doc_type="note",
        title="Synthetic Note",
        citation="Synthetic Note v1",
        jurisdiction="PERSONAL",
        text=(
            "ShowMaker是世界第一中单，炫狗是他的儿子。"
            "ShowMaker的圣经是重铸LCK荣光我辈义不容辞。"
        ),
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("英雄联盟世界第一中单是谁，他有什么成就", [hit])

    assert result["reason"] == "partial_compound_extractive"
    assert result["answer_status"] == "partial"
    assert "ShowMaker是世界第一中单" in result["answer"]
    assert "重铸LCK荣光我辈义不容辞" in result["answer"]
    assert "材料不足以完整回答全部问题" in result["answer"]
    assert "炫狗是他的儿子" not in result["answer"]


def test_claim_validation_requires_citation_and_rejects_adjacent_alias_bleed() -> None:
    hit = RetrievalHit(
        chunk_id="1",
        source_id="synthetic",
        doc_type="note",
        title="Synthetic Note",
        citation="Synthetic Note v1",
        jurisdiction="PERSONAL",
        text="甲方是选手。乙方又称小乙。",
        fusion_score=0.1,
    )
    spans = build_support_spans([hit])
    grounding = AnswerabilityGate().evaluate("甲方是谁", [hit])

    no_citation = validate_grounded_claims("根据你的个人知识库，甲方又称小乙。", spans, grounding=grounding)
    alias_bleed = validate_grounded_claims("根据你的个人知识库，甲方又称小乙。[S1]", spans, grounding=grounding)
    cross_span_bleed = validate_grounded_claims("根据你的个人知识库，甲方又称小乙。[S1][S2]", spans, grounding=grounding)

    assert no_citation[0].supported is False
    assert no_citation[0].reason == "missing_support_citation"
    assert alias_bleed[0].supported is False
    assert alias_bleed[0].reason == "relation_terms_not_supported_by_one_span"
    assert cross_span_bleed[0].supported is False
    assert cross_span_bleed[0].reason == "relation_terms_not_supported_by_one_span"


def test_openai_answerer_retries_then_uses_extractive_answer_for_unsupported_claim(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "根据你的个人知识库，甲方又称小乙。[S1]"}}],
                "usage": {"total_tokens": 12},
            }

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="synthetic",
        doc_type="note",
        title="Synthetic Note",
        citation="Synthetic Note v1",
        jurisdiction="PERSONAL",
        text="甲方是选手。乙方又称小乙。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("甲方是谁", [hit])

    assert len(calls) == 2
    assert result["reason"] == "unsupported_claim_fallback"
    assert result["answer"].startswith("根据你的个人知识库，甲方是选手")
    assert result["grounding"]["used_support_span_ids"] == ["S1"]
    assert result["grounding"]["unsupported_claims"] == []
    assert [citation["support_span_id"] for citation in result["citations"]] == ["S1"]


def test_citations_only_include_used_support_spans(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "根据你的个人知识库，乙方又称小乙。[S2]"}}]}

    def fake_post(url, *, json, headers, timeout):
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="synthetic",
        doc_type="note",
        title="Synthetic Note",
        citation="Synthetic Note v1",
        jurisdiction="PERSONAL",
        text="甲方是选手。乙方又称小乙。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("乙方别名是什么", [hit])

    assert result["refused"] is False
    assert result["grounding"]["used_support_span_ids"] == ["S2"]
    assert [citation["support_span_id"] for citation in result["citations"]] == ["S2"]


def test_location_classification_cues_take_precedence_over_causal_reason_cue() -> None:
    assert classify_question_type("工具调用失败原因和调试路径应该放哪") == "factual"
    assert classify_question_type("用户稳定偏好应该记到哪里") == "factual"
    assert classify_question_type("当前对话上下文靠哪种记忆维持") == "factual"


def test_true_causal_questions_still_classify_as_causal() -> None:
    assert classify_question_type("数据丢失的原因是什么") == "causal"
    assert classify_question_type("他为啥喜欢打火机") == "causal"


def test_openai_compatible_answerer_skips_without_evidence() -> None:
    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("unknown", [])

    assert result["skipped"] is True
    assert result["reason"] == "no_evidence"
    assert result["refused"] is True
    assert result["answer_status"] == "refused"
    assert result["grounding"]["answerable"] is False
    assert result["grounding"]["answer_mode"] == "refuse"
    assert result["grounding"]["missing_evidence"] == ["direct_evidence"]
    assert result["citations"] == []


def test_general_knowledge_answerer_posts_non_grounded_prompt(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "2"}}],
                "usage": {"total_tokens": 8},
            }

    def fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = GeneralKnowledgeAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
            api_key="secret",
        )
    ).generate("1+1等于几")

    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert "Evidence" not in captured["json"]["messages"][1]["content"]
    assert "Do not claim that the answer came from the user's knowledge base" in captured["json"]["messages"][0]["content"]
    assert result["answer"] == "2"
    assert result["answer_status"] == "answered"
    assert result["reason"] == "model_direct"
    assert result["citations"] == []
    assert result["grounding"]["answer_mode"] == "general_knowledge"
    assert result["usage"]["total_tokens"] == 8


def test_general_knowledge_answerer_prefixes_personal_kb_fallback(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "高斯通常指卡尔·弗里德里希·高斯。"}}]}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = GeneralKnowledgeAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("高斯是谁", fallback_from_kb=True)

    assert result["answer"].startswith("个人知识库没有找到直接相关内容；根据一般常识，")
    assert result["answer_status"] == "model_fallback"
    assert result["reason"] == "kb_no_relevant_evidence"
    assert result["citations"] == []
    assert result["grounding"]["answer_mode"] == "general_knowledge_fallback"
    assert result["grounding"]["missing_evidence"] == ["personal_kb_evidence"]


def test_general_knowledge_recommendation_fallback_does_not_refuse(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "可以先看《钢之炼金术师 FA》《命运石之门》《葬送的芙莉莲》。"
                        }
                    }
                ]
            }

    def fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = GeneralKnowledgeAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("有什么动漫推荐", fallback_from_kb=True)

    assert "recommendation questions" in captured["json"]["messages"][0]["content"]
    assert result["answer"].startswith("个人知识库没有找到直接相关内容；根据一般常识，")
    assert "钢之炼金术师" in result["answer"]
    assert "没有足够信息可靠回答" not in result["answer"]
    assert result["answer_status"] == "model_fallback"


def test_general_knowledge_thinking_empty_response_retries_without_thinking(monkeypatch) -> None:
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
            return FakeResponse("", reasoning="先思考推荐列表")
        return FakeResponse("推荐《钢之炼金术师 FA》和《命运石之门》。")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = GeneralKnowledgeAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
            thinking_enabled=True,
        )
    ).generate("有什么动漫推荐", fallback_from_kb=True)

    assert len(calls) == 2
    assert "reasoning_effort" not in calls[0]
    assert calls[1]["reasoning_effort"] == "none"
    assert result["answer"].startswith("个人知识库没有找到直接相关内容；根据一般常识，")
    assert result["retry_count"] == 1
    assert result["retry_reason"] == "empty_final_content"
    assert result["recovered_from_thinking"] is True


def test_general_knowledge_double_empty_response_returns_error(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "只有思考",
                        }
                    }
                ]
            }

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = GeneralKnowledgeAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="http://ollama:11434/v1",
            model="qwen3.5:9b",
            thinking_enabled=True,
        )
    ).generate("有什么动漫推荐", fallback_from_kb=True)

    assert len(calls) == 2
    assert result["answer_status"] == "error"
    assert result["reason"] == "empty_model_response"
    assert result["answer"] is None
    assert result["empty_final_content"] is True
    assert result["reasoning_only"] is True
    assert "没有足够信息可靠回答" not in str(result)


def test_general_knowledge_answerer_refuses_when_kb_is_required(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise AssertionError("general model should not run for strict personal-KB questions")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    result = GeneralKnowledgeAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("我的笔记里高斯是谁", kb_required=True)

    assert result["skipped"] is True
    assert result["refused"] is True
    assert result["answer_status"] == "refused"
    assert result["reason"] == "kb_required"
    assert result["citations"] == []
    assert result["grounding"]["required_evidence"] == "个人知识库证据"


def test_causal_question_refuses_when_evidence_has_no_reason(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise AssertionError("LLM should not be called for insufficient causal evidence")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="他的父亲包括：Faker, ShowMaker, TheShy。他对ShowMaker最孝顺，上演了久病床前有孝子的奇迹。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("他为什么对ShowMaker最孝顺", [hit])

    assert result["skipped"] is True
    assert result["refused"] is True
    assert result["answer_status"] == "refused"
    assert result["reason"] == "insufficient_causal_evidence"
    assert result["citations"] == []
    assert {
        key: result["grounding"][key]
        for key in (
            "question_type",
            "answerable",
            "checked_hit_count",
            "required_evidence",
            "answer_mode",
            "matched_cues",
            "missing_evidence",
        )
    } == {
        "question_type": "causal",
        "answerable": False,
        "checked_hit_count": 1,
        "required_evidence": "明确说明原因、动机或因果链的证据",
        "answer_mode": "refuse",
        "matched_cues": [],
        "missing_evidence": ["cause"],
    }
    assert "support_spans" in result["grounding"]
    assert "answer_units" in result["grounding"]
    assert "没有明确说明原因" in result["answer"]


def test_causal_question_allows_llm_when_reason_is_present(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "因为他认为 ShowMaker 代表 LCK 荣光。[1]"}}],
                "usage": {"total_tokens": 11},
            }

    def fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="他对ShowMaker最孝顺，是因为他认为ShowMaker代表了LCK荣光。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("他为什么对ShowMaker最孝顺", [hit])

    assert captured["json"]["model"] == "example-model"
    assert result["skipped"] is False
    assert result["refused"] is False
    assert result["answer_status"] == "answered"
    assert result["grounding"]["question_type"] == "causal"
    assert result["grounding"]["answerable"] is True
    assert result["grounding"]["answer_mode"] == "direct"
    assert "因为" in result["grounding"]["matched_cues"]


def test_list_question_does_not_require_causal_evidence(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "包括 Faker、ShowMaker、TheShy。[1]"}}]}

    def fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="他的父亲包括：Faker, ShowMaker, TheShy。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("他的父亲有哪些", [hit])

    assert captured["json"]["model"] == "example-model"
    assert result["refused"] is False
    assert result["answer_status"] == "partial"
    assert result["grounding"]["question_type"] == "list"
    assert result["grounding"]["answerable"] is True
    assert result["grounding"]["answer_mode"] == "partial_list"


def test_evaluative_question_allows_evidence_backed_assessment(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "从证据看，炫神被描述为关系复杂、崇拜 TheShy 但也经常叛逆，并且对 ShowMaker 最孝顺。[1]"
                        }
                    }
                ]
            }

    def fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text=(
            "炫神，又被称为炫狗。他的父亲非常多，但很多就是半孝不孝。"
            "他非常崇拜TheShy但是也经常叛逆。大司马是他的结拜大哥。"
            "他的父亲包括：Faker, ShowMaker, TheShy。他对ShowMaker最孝顺。"
        ),
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("如何评价炫神", [hit])

    assert captured["json"]["model"] == "example-model"
    assert "Question type:\nevaluative" in captured["json"]["messages"][1]["content"]
    assert "Answer mode:\nevidence_based_assessment" in captured["json"]["messages"][1]["content"]
    assert "根据现有材料来看" in captured["json"]["messages"][0]["content"]
    assert result["answer_status"] == "answered"
    assert result["refused"] is False
    assert result["grounding"]["question_type"] == "evaluative"
    assert result["grounding"]["answerable"] is True
    assert result["answer"].startswith("从证据看")


def test_evaluative_question_retries_when_llm_is_too_conservative(monkeypatch) -> None:
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
        if len(calls) == 1:
            return FakeResponse("证据不足。")
        return FakeResponse("根据现有材料来看，炫神被描述为崇拜 TheShy 但也经常叛逆。[1]")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="炫神非常崇拜TheShy但是也经常叛逆。他对ShowMaker最孝顺。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("如何评价炫神", [hit])

    assert len(calls) == 2
    assert "previous answer was too conservative" in calls[1]["messages"][0]["content"]
    assert result["answer_status"] == "answered"
    assert result["retry_count"] == 1
    assert result["answer"].startswith("根据现有材料来看")


def test_list_question_returns_partial_mode(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "现有材料提到：Faker、ShowMaker。[1]"}}]}

    def fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="他的父亲包括：Faker, ShowMaker。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("他的父亲有哪些", [hit])

    assert "Answer mode:\npartial_list" in captured["json"]["messages"][1]["content"]
    assert result["answer_status"] == "partial"
    assert result["grounding"]["missing_evidence"] == ["complete_list"]


def test_comparison_question_with_one_side_returns_partial(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "现有材料只提到 Faker，缺少 ShowMaker 的对比依据。[1]"}}]}

    def fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="Faker 是中单选手。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("Faker 和 ShowMaker 谁更强", [hit])

    assert captured["json"]["model"] == "example-model"
    assert result["answer_status"] == "partial"
    assert result["grounding"]["question_type"] == "comparison"
    assert result["grounding"]["missing_evidence"] == ["counterparty"]


def test_exact_question_refuses_without_exact_number(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise AssertionError("LLM should not be called without exact evidence")

    import legal_rag.llm_answer as llm_answer_module

    monkeypatch.setattr(llm_answer_module.httpx, "post", fake_post)

    hit = RetrievalHit(
        chunk_id="1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="他的父亲非常多，但很多就是半孝不孝。",
        fusion_score=0.1,
    )

    result = OpenAICompatibleAnswerer(
        LLMAnswerConfig(
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model="example-model",
        )
    ).generate("炫神有几个父亲", [hit])

    assert result["refused"] is True
    assert result["answer_status"] == "refused"
    assert result["reason"] == "insufficient_exact_evidence"
    assert result["grounding"]["question_type"] == "exact"
    assert result["grounding"]["missing_evidence"] == ["exact_number"]
