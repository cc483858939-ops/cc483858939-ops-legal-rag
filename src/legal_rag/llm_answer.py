from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from legal_rag.schema import RetrievalHit
from legal_rag.text import normalize_whitespace, tokenize

INSUFFICIENT_CAUSAL_EVIDENCE = (
    "知识库检索到了相关内容，但没有明确说明原因，不能可靠回答这个“为什么”问题。"
)

INSUFFICIENT_EXACT_EVIDENCE = (
    "知识库检索到了相关内容，但没有明确的数字、日期或精确事实，不能可靠回答这个精确问题。"
)

LLM_REFUSED_ANSWER = "模型未能基于已检索材料组织回答。"

NO_EVIDENCE_ANSWER = "当前知识库没有检索到足够证据，不能生成可靠回答。"

CAUSAL_QUESTION_CUES = (
    "为什么",
    "为啥",
    "为何",
    "原因",
    "怎么会",
    "凭什么",
    "why",
    "reason",
    "because",
    "due to",
)

CAUSAL_EVIDENCE_CUES = (
    "因为",
    "由于",
    "原因",
    "导致",
    "所以",
    "因此",
    "为了",
    "出于",
    "based on",
    "because",
    "due to",
    "reason",
    "therefore",
)

LIST_QUESTION_CUES = ("有哪些", "哪些", "列出", "包括", "list", "which", "what are")
DEFINITION_QUESTION_CUES = ("是什么", "什么意思", "定义", "叫啥", "what is", "define", "meaning")
FACTUAL_QUESTION_CUES = ("是谁", "谁", "哪里", "是否", "什么关系", "啥关系", "who", "where")
RELATION_QUESTION_CUES = ("什么关系", "啥关系", "跟", "和", "与", "relationship")
LOCATION_CLASSIFICATION_QUESTION_CUES = (
    "放哪",
    "记到哪",
    "记到哪里",
    "应该放",
    "靠哪种",
    "哪种记忆",
    "属于哪",
)
EVALUATIVE_QUESTION_CUES = (
    "如何评价",
    "怎么评价",
    "评价",
    "怎么看",
    "如何看待",
    "看待",
    "evaluate",
    "assess",
    "what do you think",
)
COMPARISON_QUESTION_CUES = (
    "比较",
    "对比",
    "相比",
    "区别",
    "差异",
    "哪个更",
    "谁更",
    "versus",
    " vs ",
    "compare",
    "difference",
    "better",
)
PROCEDURAL_QUESTION_CUES = (
    "怎么做",
    "如何做",
    "怎么办",
    "怎么处理",
    "如何处理",
    "步骤",
    "流程",
    "how to",
    "steps",
    "procedure",
)
EXACT_QUESTION_CUES = (
    "多少",
    "几个",
    "几次",
    "多久",
    "什么时候",
    "何时",
    "哪一年",
    "日期",
    "时间",
    "count",
    "number",
    "how many",
    "when",
    "date",
)
SUMMARY_QUESTION_CUES = ("总结", "概括", "归纳", "summary", "summarize")
PROCEDURAL_EVIDENCE_CUES = (
    "步骤",
    "流程",
    "首先",
    "然后",
    "最后",
    "第一",
    "第二",
    "step",
    "procedure",
    "first",
    "then",
    "finally",
)
LLM_REFUSAL_CUES = (
    "证据不足",
    "无法回答",
    "不能回答",
    "无法确定",
    "不知道",
    "没有足够",
    "缺乏足够",
    "没有关于",
    "没有找到关于",
    "未包含",
    "未提及",
    "未提到",
    "并未提及",
    "无法基于当前材料",
    "无法基于现有材料",
    "insufficient evidence",
    "not enough evidence",
    "not enough information",
    "i don't know",
    "cannot answer",
    "can't answer",
)
LLM_REFUSAL_PATTERNS = (
    r"没有[^。！？\n]{0,30}信息",
    r"现有材料[^。！？\n]{0,50}没有",
    r"根据现有材料[^。！？\n]{0,50}没有",
    r"材料[^。！？\n]{0,50}未包含",
    r"材料[^。！？\n]{0,50}未提及",
    r"现有材料主要涉及[^。！？\n]{0,80}未提及",
    r"无法基于[^。！？\n]{0,30}材料[^。！？\n]{0,20}回答",
    r"证据[^。！？\n]{0,30}不足",
)
INSUFFICIENT_EVIDENCE_REASONS = {
    "llm_refused",
    "no_evidence",
    "insufficient_causal_evidence",
    "insufficient_exact_evidence",
}
INSUFFICIENT_EVIDENCE_STATUSES = {"llm_refused", "refused"}
RETRYABLE_LLM_REFUSAL_TYPES = {"evaluative", "list", "summary", "procedural"}
STOP_QUERY_TERMS = {
    "为什么",
    "为何",
    "原因",
    "怎么会",
    "凭什么",
    "有哪些",
    "哪些",
    "是什么",
    "什么意思",
    "如何评价",
    "怎么评价",
    "评价",
    "怎么看",
    "如何看待",
    "看待",
    "比较",
    "对比",
    "相比",
    "区别",
    "差异",
    "哪个更",
    "谁更",
    "怎么做",
    "如何做",
    "怎么办",
    "步骤",
    "流程",
    "多少",
    "几个",
    "几次",
    "多久",
    "什么时候",
    "何时",
    "哪一年",
    "日期",
    "时间",
    "总结",
    "概括",
    "归纳",
    "what",
    "why",
    "reason",
    "because",
    "due",
    "to",
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "and",
    "how",
    "many",
    "when",
    "date",
    "count",
    "number",
    "compare",
    "difference",
    "better",
    "summary",
    "summarize",
}

QUESTION_CUES_FOR_TERM_REMOVAL = (
    CAUSAL_QUESTION_CUES
    + LIST_QUESTION_CUES
    + DEFINITION_QUESTION_CUES
    + EVALUATIVE_QUESTION_CUES
    + COMPARISON_QUESTION_CUES
    + PROCEDURAL_QUESTION_CUES
    + EXACT_QUESTION_CUES
    + SUMMARY_QUESTION_CUES
    + FACTUAL_QUESTION_CUES
)
QUERY_SCOPE_CUES_FOR_TERM_REMOVAL = (
    "我的知识库里",
    "个人知识库里",
    "我的笔记里",
    "我的资料里",
    "我的文档里",
    "我的语料里",
    "知识库里",
    "笔记里",
    "资料里",
    "文档里",
    "语料里",
    "根据我的知识库",
    "根据个人知识库",
    "根据我的笔记",
    "根据我的资料",
    "根据我的文档",
    "根据知识库",
)
RELATION_TERMS_FOR_REMOVAL = {
    "关系",
    "什么关系",
}
PRONOUN_QUERY_TERMS = {
    "那他",
    "那她",
    "那它",
    "这个人",
    "那个人",
    "这人",
    "那人",
    "他",
    "她",
    "它",
    "这个",
    "那个",
}
WEAK_EVIDENCE_QUERY_TERMS = {
    "什么",
    "是什么",
    "定义",
    "意思",
    "哪个",
    "哪些",
    "哪里",
    "怎么",
    "如何",
    "知识库",
    "个人",
    "材料",
    "资料",
    "问题",
    "相关",
    "内容",
    "领域",
    "what",
    "define",
    "meaning",
    "question",
    "knowledge",
    "base",
    "the",
    "and",
    "are",
    "is",
    "of",
    "to",
}
QUERY_TERM_EXPANSIONS = {
    "爹": ("父亲",),
    "爸": ("父亲",),
    "爸爸": ("父亲",),
    "老爹": ("父亲",),
    "干爹": ("父亲",),
}


@dataclass(frozen=True)
class LLMAnswerConfig:
    provider: str
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 700
    timeout_seconds: float = 45.0
    thinking_enabled: bool = False

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "thinking_enabled": self.thinking_enabled,
            "api_key_configured": bool(self.api_key),
        }


@dataclass(frozen=True)
class ChatCompletionResult:
    data: dict[str, Any]
    answer: str
    retry_count: int = 0
    retry_reason: str | None = None
    recovered_from_thinking: bool = False
    empty_final_content: bool = False
    reasoning_only: bool = False


class ChatCompletionCallError(RuntimeError):
    def __init__(
        self,
        exc: Exception,
        *,
        retry_count: int = 0,
        retry_reason: str | None = None,
    ) -> None:
        super().__init__(str(exc))
        self.original = exc
        self.retry_count = retry_count
        self.retry_reason = retry_reason


@dataclass(frozen=True)
class AnswerabilityResult:
    question_type: str
    answerable: bool
    checked_hit_count: int
    required_evidence: str
    answer_mode: str
    answer_status: str
    matched_cues: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "answerable": self.answerable,
            "checked_hit_count": self.checked_hit_count,
            "required_evidence": self.required_evidence,
            "answer_mode": self.answer_mode,
            "matched_cues": self.matched_cues,
            "missing_evidence": self.missing_evidence,
        }


class AnswerabilityGate:
    def evaluate(self, question: str, hits: list[RetrievalHit]) -> AnswerabilityResult:
        question_type = classify_question_type(question)
        checked_hit_count = min(len(hits), 8)
        if not hits:
            return AnswerabilityResult(
                question_type=question_type,
                answerable=False,
                checked_hit_count=0,
                required_evidence="直接回答问题的检索证据",
                answer_mode="refuse",
                answer_status="refused",
                missing_evidence=["direct_evidence"],
                reason="no_evidence",
            )

        if question_type == "causal":
            matched_cues = matched_causal_evidence_cues(question, hits[:8])
            answerable = bool(matched_cues)
            return AnswerabilityResult(
                question_type=question_type,
                answerable=answerable,
                checked_hit_count=checked_hit_count,
                required_evidence="明确说明原因、动机或因果链的证据",
                answer_mode="direct" if answerable else "refuse",
                answer_status="answered" if answerable else "refused",
                matched_cues=matched_cues,
                missing_evidence=[] if answerable else ["cause"],
                reason=None if answerable else "insufficient_causal_evidence",
            )

        if question_type == "exact":
            matched_cues = matched_exact_evidence_cues(question, hits[:8])
            answerable = bool(matched_cues)
            return AnswerabilityResult(
                question_type=question_type,
                answerable=answerable,
                checked_hit_count=checked_hit_count,
                required_evidence="明确的数字、日期、时间或精确事实证据",
                answer_mode="direct" if answerable else "refuse",
                answer_status="answered" if answerable else "refused",
                matched_cues=matched_cues,
                missing_evidence=[] if answerable else ["exact_number"],
                reason=None if answerable else "insufficient_exact_evidence",
            )

        if question_type == "evaluative":
            return AnswerabilityResult(
                question_type=question_type,
                answerable=True,
                checked_hit_count=checked_hit_count,
                required_evidence="可支撑评价的描述性事实证据",
                answer_mode="evidence_based_assessment",
                answer_status="answered",
            )

        if question_type == "list":
            return AnswerabilityResult(
                question_type=question_type,
                answerable=True,
                checked_hit_count=checked_hit_count,
                required_evidence="可列举条目的证据",
                answer_mode="partial_list",
                answer_status="partial",
                missing_evidence=["complete_list"],
            )

        if question_type == "comparison":
            found_terms = matched_query_terms_in_hits(extract_query_terms(question), hits[:8])
            has_both_sides = len(found_terms) >= 2
            return AnswerabilityResult(
                question_type=question_type,
                answerable=True,
                checked_hit_count=checked_hit_count,
                required_evidence="被比较对象双方的证据",
                answer_mode="direct",
                answer_status="answered" if has_both_sides else "partial",
                matched_cues=found_terms,
                missing_evidence=[] if has_both_sides else ["counterparty"],
            )

        if question_type == "procedural":
            matched_cues = matched_procedural_evidence_cues(hits[:8])
            has_steps = bool(matched_cues)
            return AnswerabilityResult(
                question_type=question_type,
                answerable=True,
                checked_hit_count=checked_hit_count,
                required_evidence="步骤、流程或处理办法证据",
                answer_mode="direct" if has_steps else "partial_procedure",
                answer_status="answered" if has_steps else "partial",
                matched_cues=matched_cues,
                missing_evidence=[] if has_steps else ["complete_steps"],
            )

        matched_terms = matched_evidence_query_terms(question, hits[:8])
        if question_type == "definition" and matched_terms:
            return AnswerabilityResult(
                question_type=question_type,
                answerable=True,
                checked_hit_count=checked_hit_count,
                required_evidence="实体在知识库语境中的描述、属性或关系证据",
                answer_mode="contextual_definition",
                answer_status="answered",
                matched_cues=matched_terms,
                missing_evidence=[],
                reason=None,
            )
        if question_type == "definition":
            return AnswerabilityResult(
                question_type=question_type,
                answerable=False,
                checked_hit_count=checked_hit_count,
                required_evidence="实体在知识库语境中的描述、属性或关系证据",
                answer_mode="refuse",
                answer_status="refused",
                matched_cues=[],
                missing_evidence=["entity_context"],
                reason="no_evidence",
            )
        return AnswerabilityResult(
            question_type=question_type,
            answerable=True,
            checked_hit_count=checked_hit_count,
            required_evidence="直接回答问题的检索证据",
            answer_mode="direct",
            answer_status="answered",
            matched_cues=matched_terms,
            missing_evidence=[],
            reason=None,
        )


class OpenAICompatibleAnswerer:
    def __init__(
        self,
        config: LLMAnswerConfig,
        *,
        answerability_gate: AnswerabilityGate | None = None,
    ) -> None:
        self.config = config
        self.answerability_gate = answerability_gate or AnswerabilityGate()

    def generate(self, question: str, hits: list[RetrievalHit]) -> dict[str, Any]:
        started = time.perf_counter()
        grounding = self.answerability_gate.evaluate(question, hits)
        if not hits:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "no_evidence",
                "refused": True,
                "answer_status": "refused",
                "answer": NO_EVIDENCE_ANSWER,
                "citations": [],
                "grounding": _grounding_payload(grounding, hits),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": None,
            }

        if not grounding.answerable:
            return {
                "enabled": True,
                "skipped": True,
                "reason": grounding.reason,
                "refused": True,
                "answer_status": "refused",
                "answer": refusal_answer_for(grounding),
                "citations": [],
                "grounding": _grounding_payload(grounding, hits),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": None,
            }

        if not self.config.base_url.strip() or not self.config.model.strip():
            return {
                "enabled": True,
                "skipped": True,
                "reason": "missing_llm_config",
                "refused": False,
                "answer_status": "error",
                "answer": None,
                "citations": [],
                "grounding": _grounding_payload(grounding, hits),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": "base_url and model are required",
            }

        try:
            chat = _chat_completion_with_recovery(
                self.config,
                messages=_build_messages(question, hits, grounding=grounding),
            )
            data = chat.data
            answer = chat.answer
            retry_count = chat.retry_count
            retry_reason = chat.retry_reason
            recovered_from_thinking = chat.recovered_from_thinking
            if not answer:
                return _empty_model_response_result(
                    started=started,
                    grounding=_grounding_payload(grounding, hits),
                    llm=self.config.safe_dict(),
                    citations=[],
                    chat=chat,
                )
            if _should_retry_llm_refusal(grounding, answer):
                retry_chat = _chat_completion_with_recovery(
                    self.config,
                    messages=_build_messages(question, hits, grounding=grounding, retry=True),
                )
                retry_count += 1 + retry_chat.retry_count
                retry_reason = _join_retry_reasons(
                    retry_reason,
                    retry_chat.retry_reason,
                    "llm_refusal",
                )
                recovered_from_thinking = (
                    recovered_from_thinking or retry_chat.recovered_from_thinking
                )
                data = retry_chat.data
                answer = retry_chat.answer
                if not answer:
                    return _empty_model_response_result(
                        started=started,
                        grounding=_grounding_payload(grounding, hits),
                        llm=self.config.safe_dict(),
                        citations=[],
                        chat=retry_chat,
                        retry_count=retry_count,
                        retry_reason=retry_reason,
                        recovered_from_thinking=recovered_from_thinking,
                    )

            if _should_return_llm_refusal(grounding, answer):
                return {
                    "enabled": True,
                    "skipped": False,
                    "reason": "llm_refused",
                    "refused": True,
                    "answer_status": "llm_refused",
                    "answer": LLM_REFUSED_ANSWER,
                    "citations": [],
                    "grounding": _grounding_payload(grounding, hits),
                    "llm": self.config.safe_dict(),
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                    "duration_ms": _elapsed_ms(started),
                    "retry_count": retry_count,
                    "retry_reason": retry_reason,
                    "recovered_from_thinking": recovered_from_thinking,
                    "error": None,
                }

            return {
                "enabled": True,
                "skipped": False,
                "reason": None,
                "refused": False,
                "answer_status": grounding.answer_status,
                "answer": answer,
                "citations": _citations(hits),
                "grounding": _grounding_payload(grounding, hits),
                "llm": self.config.safe_dict(),
                "usage": data.get("usage") if isinstance(data, dict) else None,
                "duration_ms": _elapsed_ms(started),
                "retry_count": retry_count,
                "retry_reason": retry_reason,
                "recovered_from_thinking": recovered_from_thinking,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - answer layer must fail open.
            retry_count = getattr(exc, "retry_count", 0)
            retry_reason = getattr(exc, "retry_reason", None)
            return {
                "enabled": True,
                "skipped": True,
                "reason": "llm_error",
                "refused": False,
                "answer_status": "error",
                "answer": None,
                "citations": [],
                "grounding": _grounding_payload(grounding, hits),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "retry_count": retry_count,
                "retry_reason": retry_reason,
                "recovered_from_thinking": False,
                "error": _format_exception(exc),
            }


class GeneralKnowledgeAnswerer:
    def __init__(self, config: LLMAnswerConfig) -> None:
        self.config = config

    def generate(
        self,
        question: str,
        *,
        fallback_from_kb: bool = False,
        kb_required: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        grounding = {
            "question_type": classify_question_type(question),
            "answerable": not kb_required,
            "checked_hit_count": 0,
            "required_evidence": (
                "个人知识库证据" if kb_required else "模型通用知识；不使用个人知识库引用"
            ),
            "answer_mode": "general_knowledge_fallback" if fallback_from_kb else "general_knowledge",
            "matched_cues": [],
            "missing_evidence": ["personal_kb_evidence"] if fallback_from_kb else [],
        }
        if kb_required:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "kb_required",
                "refused": True,
                "answer_status": "refused",
                "answer": NO_EVIDENCE_ANSWER,
                "citations": [],
                "grounding": grounding,
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": None,
            }

        if not self.config.base_url.strip() or not self.config.model.strip():
            return {
                "enabled": True,
                "skipped": True,
                "reason": "missing_llm_config",
                "refused": False,
                "answer_status": "error",
                "answer": None,
                "citations": [],
                "grounding": grounding,
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": "base_url and model are required",
            }

        try:
            chat = _chat_completion_with_recovery(
                self.config,
                messages=_build_general_messages(question, fallback_from_kb=fallback_from_kb),
            )
            data = chat.data
            answer = chat.answer
            if not answer:
                return _empty_model_response_result(
                    started=started,
                    grounding=grounding,
                    llm=self.config.safe_dict(),
                    citations=[],
                    chat=chat,
                )
            if fallback_from_kb:
                answer = _prefix_kb_fallback_answer(question, answer)
            return {
                "enabled": True,
                "skipped": False,
                "reason": "kb_no_relevant_evidence" if fallback_from_kb else "model_direct",
                "refused": False,
                "answer_status": "model_fallback" if fallback_from_kb else "answered",
                "answer": answer,
                "citations": [],
                "grounding": grounding,
                "llm": self.config.safe_dict(),
                "usage": data.get("usage") if isinstance(data, dict) else None,
                "duration_ms": _elapsed_ms(started),
                "retry_count": chat.retry_count,
                "retry_reason": chat.retry_reason,
                "recovered_from_thinking": chat.recovered_from_thinking,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - general answer layer must fail open as an error.
            retry_count = getattr(exc, "retry_count", 0)
            retry_reason = getattr(exc, "retry_reason", None)
            return {
                "enabled": True,
                "skipped": True,
                "reason": "llm_error",
                "refused": False,
                "answer_status": "error",
                "answer": None,
                "citations": [],
                "grounding": grounding,
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "retry_count": retry_count,
                "retry_reason": retry_reason,
                "recovered_from_thinking": False,
                "error": _format_exception(exc),
            }


def _build_messages(
    question: str,
    hits: list[RetrievalHit],
    *,
    grounding: AnswerabilityResult | None = None,
    retry: bool = False,
) -> list[dict[str, str]]:
    evidence = "\n\n".join(_format_evidence(index, hit) for index, hit in enumerate(hits[:8], 1))
    grounding = grounding or AnswerabilityResult(
        question_type=classify_question_type(question),
        answerable=bool(hits),
        checked_hit_count=min(len(hits), 8),
        required_evidence="直接回答问题的检索证据",
        answer_mode="direct",
        answer_status="answered" if hits else "refused",
        missing_evidence=[] if hits else ["direct_evidence"],
    )
    if retry and grounding.answer_mode == "contextual_definition":
        retry_note = (
            "The previous answer treated this as requiring an encyclopedia definition. "
            "Use the evidence to explain what the entity means in the user's personal "
            "knowledge-base context instead. "
        )
    elif retry:
        retry_note = (
            "The previous answer was too conservative. The evidence is usable for this question "
            "type, so produce the allowed grounded or partial answer instead of replying only "
            "that evidence is insufficient. "
        )
    else:
        retry_note = ""
    contextual_definition_instruction = (
        "For contextual_definition questions, the user is asking what the entity means in "
        "their personal knowledge base. If the evidence mentions the entity as a song, alias, "
        "preference, relation, title, or other contextual fact, answer that contextual meaning. "
        "Keep the contextual definition to the entity's role, type, or direct relation in the "
        "knowledge base, such as 'X is Y's favorite song'. Do not include reasons, family clues, "
        "lyrics beyond the identifying phrase, or neighboring facts unless the user asks why or "
        "asks about those facts. "
        "Do not require the evidence to contain an encyclopedia-style definition. Do not add "
        "outside common knowledge. If the user asks in Chinese, frame it as '根据你的个人知识库，'. "
        if grounding.answer_mode == "contextual_definition"
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a grounded RAG answer layer. Answer only from the provided evidence. "
                "Treat the evidence chunks as the user's personal knowledge base. When the user asks in "
                "Chinese, prefer starting the answer with '根据你的个人知识库，'. "
                "First decide whether the evidence chunks directly answer, partially answer, or cannot "
                "answer the question. "
                "The Question type, Answer mode, and Missing evidence fields are an upstream "
                "answerability decision. If Answer mode is not 'refuse' and Missing evidence is "
                "'none', treat the evidence as sufficient and do not answer with insufficient "
                "evidence. Extract the directly supported answer instead. If Answer mode is "
                "partial, provide the supported partial answer instead of refusing. "
                "Use the same language as the user when possible. Cite claims with bracketed "
                "chunk ids like [1] or [2]. Every factual claim must be supported by the cited "
                "chunk. Do not cite a chunk unless that chunk supports the claim. "
                "Preserve exact names, titles, metric names, and key phrases from the evidence chunks "
                "when they answer the question. "
                "Use the complete chunk context to resolve pronouns, aliases, relations, and nearby "
                "subject references. Do not discard relevant surrounding context inside the same chunk. "
                "Keep the answer tightly focused on the entity and relation asked by the user. "
                "Do not bind a neighboring fact to the wrong subject. If one source contains several "
                "facts, include only the "
                "fact that answers the requested slot; do not mention siblings, parents, songs, "
                "reasons, aliases, or other adjacent facts unless the question asks for them. "
                "When an entity appears inside a list, answer only the list relation for that entity; "
                "do not attach later sentences about the list owner or another list member to it. "
                "Only state that X is also known as, called, or aliased as Y when the same chunk "
                "directly makes X the subject of that alias statement. If the alias sentence is about "
                "a different subject, do not transfer that alias to the queried entity. "
                "中文主体绑定规则："
                "“A，又被称为B”只说明A的别名是B，不能说明相邻句里的C也叫A或B。"
                "“他的父亲包括：A,B,C”只说明A/B/C是“他”的父亲列表成员；"
                "后续“他对X最孝顺”里的“他”仍指前文主体，不指列表里的A/B/C。"
                "如果用户问列表成员A是谁，只回答A与前文主体的列表关系，例如“A是某人的父亲之一”，"
                "不要把后续关于前文主体的行为、偏好或别名写成A的事实。"
                "如果用户问“某个称号/身份是谁”，只取直接包含该称号/身份的句子的主语作为答案。"
                "For name questions, answer the name if present; if only a partial name clue is "
                "present, state only that clue and say the full name is not in the evidence. "
                f"{contextual_definition_instruction}"
                "For why/reason/causal questions, answer only when the evidence explicitly "
                "states a cause, reason, motive, or causal chain. If the evidence only repeats "
                "a related fact or conclusion without explaining why, say the evidence is "
                "insufficient. If Answer mode is direct for a causal question, an explicit cause "
                "cue was already found; use the sentence containing that cause cue and cite it. "
                "For evaluate/assess/opinion-style questions, synthesize an evidence-backed "
                "assessment from descriptive facts in the evidence. Start with '根据现有材料来看' "
                "when the user asks in Chinese. Do not require the evidence to literally contain "
                "an evaluation label, but do not add outside opinions. "
                "For list questions, list only items mentioned in evidence and say they are what "
                "the current materials mention. "
                "For comparison questions, compare only sides that have evidence; if one side is "
                "missing, give the supported side and state the gap. "
                "For procedural questions, give known steps only; if the complete procedure is "
                "missing, state that the materials only support a partial procedure. "
                "For exact number/date/time questions, answer only when exact evidence is present. "
                "Do not invent policy, law, numbers, timelines, URLs, or commitments. "
                "Keep answers short: use one compact paragraph by default, or at most three "
                "bullets when a list or procedure is necessary. "
                f"{retry_note}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Question type:\n{grounding.question_type}\n\n"
                f"Answer mode:\n{grounding.answer_mode}\n\n"
                f"Required evidence:\n{grounding.required_evidence}\n\n"
                f"Missing evidence:\n{', '.join(grounding.missing_evidence) if grounding.missing_evidence else 'none'}\n\n"
                f"Evidence chunks:\n{evidence}\n\n"
                "Answer by units when the question has multiple parts. If answerable, write a concise answer. If the answer mode is partial, write the "
                "partial answer and explicitly frame it as based on current materials. Include "
                "chunk citations next to relevant claims. If not answerable from evidence chunks, say what "
                "specific evidence is missing."
            ),
        },
    ]


def _build_general_messages(question: str, *, fallback_from_kb: bool = False) -> list[dict[str, str]]:
    fallback_instruction = (
        "用户的问题已经先检索个人知识库，但没有找到直接相关内容。"
        "如果用户使用中文，回答必须以“个人知识库没有找到直接相关内容；根据一般常识，”开头；"
        "如果用户使用其他语言，也要先说明 personal knowledge base did not contain direct evidence. "
        "这并不代表不能回答；只要问题可以由一般常识、常见推荐、通用解释或常规建议回答，"
        "就直接给出有用答案。只能使用通用模型知识回答，不要编造个人知识库事实，不要生成引用。"
        if fallback_from_kb
        else "This question does not require personal knowledge-base retrieval. Answer directly from general model knowledge only. "
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the general-knowledge answer layer for a personal knowledge-base assistant. "
                f"{fallback_instruction}"
                "Use the same language as the user when possible. Keep the answer concise and useful. "
                "For recommendation questions, do not refuse only because the user did not provide "
                "preferences. Give 5 to 8 broadly useful recommendations, with short reasons, and "
                "say the user can narrow by genre, era, mood, or style. "
                "Do not claim that the answer came from the user's knowledge base. Do not include "
                "bracketed RAG citations. If the user explicitly asks what their notes, documents, "
                "or knowledge base say, state that personal-KB evidence is required instead of "
                "answering from general knowledge."
            ),
        },
        {"role": "user", "content": question},
    ]


def _grounding_payload(grounding: AnswerabilityResult, hits: list[RetrievalHit]) -> dict[str, Any]:
    payload = grounding.as_dict()
    context_hit_count = min(len(hits), 8)
    payload.update(
        {
            "context_hit_count": context_hit_count,
            "relevant_hit_count": context_hit_count if grounding.answerable else 0,
        }
    )
    return payload


def _format_evidence(index: int, hit: RetrievalHit) -> str:
    text = normalize_whitespace(hit.text)[:1800]
    return (
        f"[{index}] {hit.title}\n"
        f"Citation: {hit.citation}\n"
        f"Doc type: {hit.doc_type}\n"
        f"Section: {hit.section or '-'}\n"
        f"Text: {text}"
    )


def _citations(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    return [
        _citation_for_hit(index, hit)
        for index, hit in enumerate(hits[:8], 1)
    ]


def _citation_for_hit(rank: int, hit: RetrievalHit) -> dict[str, Any]:
    return {
        "rank": rank,
        "title": hit.title,
        "citation": hit.citation,
        "chunk_id": hit.chunk_id,
        "source_id": hit.source_id,
    }


def _chat_completions_url(base_url: str) -> str:
    clean = base_url.strip().rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def _chat_payload(config: LLMAnswerConfig, *, messages: list[dict[str, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if _is_ollama_openai_endpoint(config.base_url) and not config.thinking_enabled:
        payload["reasoning_effort"] = "none"
    return payload


def _chat_completion_with_recovery(
    config: LLMAnswerConfig,
    *,
    messages: list[dict[str, str]],
) -> ChatCompletionResult:
    retry_count = 0
    retry_reason: str | None = None
    recovered_from_thinking = False
    try:
        data = _post_chat_completion(config, messages=messages)
    except httpx.ReadTimeout as exc:
        if not _should_retry_without_thinking(config):
            raise
        retry_count = 1
        retry_reason = "thinking_read_timeout"
        try:
            data = _post_chat_completion(_without_thinking(config), messages=messages)
        except Exception as retry_exc:  # noqa: BLE001 - preserve retry metadata.
            raise ChatCompletionCallError(
                retry_exc,
                retry_count=retry_count,
                retry_reason=retry_reason,
            ) from retry_exc
        answer = _extract_answer_text(data)
        recovered_from_thinking = bool(answer)
        return ChatCompletionResult(
            data=data,
            answer=answer,
            retry_count=retry_count,
            retry_reason=retry_reason,
            recovered_from_thinking=recovered_from_thinking,
            empty_final_content=not bool(answer),
            reasoning_only=_has_reasoning_without_final_content(data),
        )

    answer = _extract_answer_text(data)
    if not answer and _should_retry_without_thinking(config):
        retry_count = 1
        retry_reason = "empty_final_content"
        try:
            data = _post_chat_completion(_without_thinking(config), messages=messages)
        except Exception as retry_exc:  # noqa: BLE001 - preserve retry metadata.
            raise ChatCompletionCallError(
                retry_exc,
                retry_count=retry_count,
                retry_reason=retry_reason,
            ) from retry_exc
        answer = _extract_answer_text(data)
        recovered_from_thinking = bool(answer)

    return ChatCompletionResult(
        data=data,
        answer=answer,
        retry_count=retry_count,
        retry_reason=retry_reason,
        recovered_from_thinking=recovered_from_thinking,
        empty_final_content=not bool(answer),
        reasoning_only=_has_reasoning_without_final_content(data),
    )


def _post_chat_completion(
    config: LLMAnswerConfig,
    *,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    response = httpx.post(
        _chat_completions_url(config.base_url),
        json=_chat_payload(config, messages=messages),
        headers=_chat_headers(config),
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _chat_headers(config: LLMAnswerConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_key and config.api_key.strip():
        headers["Authorization"] = f"Bearer {config.api_key.strip()}"
    return headers


def _should_retry_without_thinking(config: LLMAnswerConfig) -> bool:
    return config.thinking_enabled


def _without_thinking(config: LLMAnswerConfig) -> LLMAnswerConfig:
    return replace(config, thinking_enabled=False)


def _is_ollama_openai_endpoint(base_url: str) -> bool:
    normalized = base_url.strip().casefold()
    return "ollama" in normalized or ":11434" in normalized


def _extract_answer_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return normalize_whitespace("\n".join(_content_part_text(part) for part in content))
        text = choices[0].get("text") if isinstance(choices[0], dict) else None
        if isinstance(text, str):
            return text.strip()
    return ""


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    for key in ("text", "content", "value"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _has_reasoning_without_final_content(data: Any) -> bool:
    if not isinstance(data, dict) or _extract_answer_text(data):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return False
    return any(
        _has_nonempty_text(message.get(key))
        for key in ("reasoning_content", "thinking", "reasoning")
    )


def _has_nonempty_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_nonempty_text(item) for item in value)
    if isinstance(value, dict):
        return any(_has_nonempty_text(item) for item in value.values())
    return False


def _empty_model_response_result(
    *,
    started: float,
    grounding: dict[str, Any],
    llm: dict[str, Any],
    citations: list[dict[str, Any]],
    chat: ChatCompletionResult,
    retry_count: int | None = None,
    retry_reason: str | None = None,
    recovered_from_thinking: bool | None = None,
) -> dict[str, Any]:
    final_retry_count = chat.retry_count if retry_count is None else retry_count
    final_retry_reason = chat.retry_reason if retry_reason is None else retry_reason
    final_recovered = (
        chat.recovered_from_thinking
        if recovered_from_thinking is None
        else recovered_from_thinking
    )
    return {
        "enabled": True,
        "skipped": True,
        "reason": "empty_model_response",
        "refused": False,
        "answer_status": "error",
        "answer": None,
        "citations": citations,
        "grounding": grounding,
        "llm": llm,
        "usage": chat.data.get("usage") if isinstance(chat.data, dict) else None,
        "duration_ms": _elapsed_ms(started),
        "retry_count": final_retry_count,
        "retry_reason": final_retry_reason,
        "recovered_from_thinking": final_recovered,
        "empty_final_content": True,
        "reasoning_only": chat.reasoning_only,
        "error": "empty_final_content: model returned reasoning/thinking content or an empty final answer",
    }


def _join_retry_reasons(*reasons: str | None) -> str | None:
    cleaned = [reason for reason in reasons if reason]
    return ",".join(dict.fromkeys(cleaned)) if cleaned else None


def _format_exception(exc: Exception) -> str:
    original = getattr(exc, "original", exc)
    return f"{type(original).__name__}: {str(original)[:500]}"


def _prefix_kb_fallback_answer(question: str, answer: str) -> str:
    clean = normalize_whitespace(answer)
    if _looks_chinese(question):
        prefix = "个人知识库没有找到直接相关内容；根据一般常识，"
        if clean.startswith("个人知识库没有找到") or clean.startswith("你的个人知识库没有找到"):
            return clean
        return f"{prefix}{clean}" if clean else "个人知识库没有找到直接相关内容；根据一般常识，我也没有足够信息可靠回答。"
    prefix = "I did not find directly relevant content in your personal knowledge base. From general knowledge, "
    if clean.casefold().startswith("i did not find directly relevant content"):
        return clean
    return f"{prefix}{clean}" if clean else "I did not find directly relevant content in your personal knowledge base, and I do not have enough general information to answer reliably."


def _looks_chinese(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def refusal_answer_for(grounding: AnswerabilityResult) -> str:
    if grounding.reason == "insufficient_causal_evidence":
        return INSUFFICIENT_CAUSAL_EVIDENCE
    if grounding.reason == "insufficient_exact_evidence":
        return INSUFFICIENT_EXACT_EVIDENCE
    return NO_EVIDENCE_ANSWER


def _should_retry_llm_refusal(grounding: AnswerabilityResult, answer: str) -> bool:
    if not _looks_like_llm_refusal(answer):
        return False
    if grounding.question_type in RETRYABLE_LLM_REFUSAL_TYPES:
        return True
    return grounding.answer_mode == "contextual_definition" and bool(grounding.matched_cues)


def _should_return_llm_refusal(grounding: AnswerabilityResult, answer: str) -> bool:
    return _should_retry_llm_refusal(grounding, answer)


def answer_result_indicates_insufficient_evidence(answer_result: dict[str, Any]) -> bool:
    if answer_result.get("answer_status") == "error":
        return False
    reason = str(answer_result.get("reason") or "").casefold()
    answer_status = str(answer_result.get("answer_status") or "").casefold()
    if reason in INSUFFICIENT_EVIDENCE_REASONS:
        return True
    if answer_status in INSUFFICIENT_EVIDENCE_STATUSES:
        return True
    answer = answer_result.get("answer")
    return isinstance(answer, str) and _looks_like_llm_refusal(answer)


def _looks_like_llm_refusal(answer: str) -> bool:
    normalized = normalize_whitespace(answer).casefold()
    if not normalized:
        return True
    if any(cue.casefold() in normalized for cue in LLM_REFUSAL_CUES):
        return True
    return any(re.search(pattern, normalized) for pattern in LLM_REFUSAL_PATTERNS)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def classify_question_type(question: str) -> str:
    normalized = question.casefold()
    if _contains_any(normalized, LOCATION_CLASSIFICATION_QUESTION_CUES):
        return "factual"
    if _contains_any(normalized, CAUSAL_QUESTION_CUES):
        return "causal"
    if _contains_any(normalized, EVALUATIVE_QUESTION_CUES):
        return "evaluative"
    if _contains_any(normalized, COMPARISON_QUESTION_CUES):
        return "comparison"
    if _contains_any(normalized, LIST_QUESTION_CUES):
        return "list"
    if _contains_any(normalized, EXACT_QUESTION_CUES):
        return "exact"
    if _contains_any(normalized, PROCEDURAL_QUESTION_CUES):
        return "procedural"
    if _contains_any(normalized, DEFINITION_QUESTION_CUES):
        return "definition"
    if _contains_any(normalized, SUMMARY_QUESTION_CUES):
        return "summary"
    if _contains_any(normalized, FACTUAL_QUESTION_CUES):
        return "factual"
    return "other"


def matched_causal_evidence_cues(question: str, hits: list[RetrievalHit]) -> list[str]:
    query_terms = evidence_query_terms(question)
    matched: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        sentences = split_evidence_sentences(hit.text)
        term_sentence_indexes = {
            index
            for index, sentence in enumerate(sentences)
            if any(term.casefold() in sentence.casefold() for term in query_terms)
        }
        for index, sentence in enumerate(sentences):
            normalized_sentence = sentence.casefold()
            has_query_context = (
                not query_terms
                or index in term_sentence_indexes
                or index - 1 in term_sentence_indexes
                or index + 1 in term_sentence_indexes
            )
            if not has_query_context:
                continue
            for cue in CAUSAL_EVIDENCE_CUES:
                if cue.casefold() in normalized_sentence and cue not in seen:
                    matched.append(cue)
                    seen.add(cue)
    return matched


def matched_exact_evidence_cues(question: str, hits: list[RetrievalHit]) -> list[str]:
    matched: list[str] = []
    for hit in hits:
        for sentence in split_evidence_sentences(hit.text):
            if re.search(r"\d+(?:\.\d+)?", sentence):
                matched.append("number")
            if re.search(r"\d{4}\s*年|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/]\d{1,2}[-/]\d{1,2}", sentence):
                matched.append("date")
            if re.search(r"[一二三四五六七八九十百千万亿两]+", sentence):
                matched.append("chinese_number")
    return dedupe_preserve_order(matched)


def matched_procedural_evidence_cues(hits: list[RetrievalHit]) -> list[str]:
    matched: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        normalized_text = hit.text.casefold()
        for cue in PROCEDURAL_EVIDENCE_CUES:
            if cue.casefold() in normalized_text and cue not in seen:
                matched.append(cue)
                seen.add(cue)
    return matched


def matched_query_terms_in_hits(query_terms: list[str], hits: list[RetrievalHit]) -> list[str]:
    matched: list[str] = []
    text = "\n".join(hit.text for hit in hits).casefold()
    for term in query_terms:
        if term.casefold() in text:
            matched.append(term)
    return dedupe_preserve_order(matched)


def matched_evidence_query_terms(question: str, hits: list[RetrievalHit]) -> list[str]:
    matched = matched_query_terms_in_hits(_valid_evidence_query_terms(extract_query_terms(question)), hits)
    if matched:
        return matched
    return matched_query_terms_in_hits(_valid_evidence_query_terms(tokenize(question)), hits)


def evidence_query_terms(question: str) -> list[str]:
    return [*_valid_evidence_query_terms(extract_query_terms(question)), *_valid_evidence_query_terms(tokenize(question))]


def _valid_evidence_query_terms(raw_terms: list[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = term.strip().casefold()
        candidates = [term.strip(), *QUERY_TERM_EXPANSIONS.get(normalized, ())]
        for candidate in candidates:
            candidate_normalized = candidate.strip().casefold()
            if not _is_valid_evidence_query_term(candidate_normalized) or candidate_normalized in seen:
                continue
            terms.append(candidate.strip())
            seen.add(candidate_normalized)
    return terms


def extract_query_terms(question: str) -> list[str]:
    normalized = question.casefold()
    for cue in QUERY_SCOPE_CUES_FOR_TERM_REMOVAL:
        normalized = normalized.replace(cue.casefold(), " ")
    for cue in QUESTION_CUES_FOR_TERM_REMOVAL:
        normalized = normalized.replace(cue.casefold(), " ")

    terms: list[str] = []
    for match in re.finditer(r"[a-z0-9][a-z0-9_.-]{1,}", normalized):
        term = match.group(0).strip("_.-")
        if _is_valid_query_term(term):
            terms.append(term)
    for match in re.finditer(r"[\u4e00-\u9fff]{2,}", normalized):
        chunk = match.group(0)
        parts = [part for part in re.split(r"[和与及、，,/的]+", chunk) if part]
        if len(parts) > 1:
            for part in parts:
                if _is_valid_query_term(part):
                    terms.append(part)
            continue
        if _is_valid_query_term(chunk):
            terms.append(chunk)
    return dedupe_preserve_order(terms)


def _is_valid_query_term(term: str) -> bool:
    term = term.strip()
    return bool(
        term
        and len(term) >= 2
        and term not in STOP_QUERY_TERMS
        and term not in RELATION_TERMS_FOR_REMOVAL
        and term not in PRONOUN_QUERY_TERMS
    )


def _is_valid_evidence_query_term(term: str) -> bool:
    return bool(
        term
        and len(term) >= 2
        and term not in WEAK_EVIDENCE_QUERY_TERMS
        and term not in STOP_QUERY_TERMS
        and term not in RELATION_TERMS_FOR_REMOVAL
        and term not in PRONOUN_QUERY_TERMS
        and not term.isdigit()
    )


def split_evidence_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    return [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*|\n+", normalized) if part.strip()]


def _contains_any(value: str, cues: tuple[str, ...]) -> bool:
    return any(cue.casefold() in value for cue in cues)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
