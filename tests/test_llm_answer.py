from __future__ import annotations

from legal_rag.llm_answer import (
    AnswerabilityGate,
    GeneralKnowledgeAnswerer,
    LLMAnswerConfig,
    OpenAICompatibleAnswerer,
    classify_question_type,
    extract_query_terms,
)
from legal_rag.schema import RetrievalHit


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
        assert result.answer_mode == "direct"
        assert result.answer_status == "answered"
        assert result.missing_evidence == []
        assert result.reason is None


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
    assert result["grounding"] == {
        "question_type": "causal",
        "answerable": False,
        "checked_hit_count": 1,
        "required_evidence": "明确说明原因、动机或因果链的证据",
        "answer_mode": "refuse",
        "matched_cues": [],
        "missing_evidence": ["cause"],
    }
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
