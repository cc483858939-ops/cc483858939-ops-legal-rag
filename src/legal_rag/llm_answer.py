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
RELATION_ANSWER_CUES = (
    "兄弟",
    "大哥",
    "结拜",
    "父亲",
    "母亲",
    "儿子",
    "女儿",
    "孝顺",
    "崇拜",
)
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
CLAIM_RELATION_CUES = (
    "又称",
    "也称",
    "也叫",
    "别名",
    "被称为",
    "被描述为",
    "父亲",
    "母亲",
    "兄弟",
    "关系",
    "成就",
    "口号",
    "最喜欢",
    "第一中单",
    "是",
    " is ",
    " are ",
)
MISSING_EVIDENCE_CUES = (
    "缺少",
    "没有",
    "未提到",
    "未提及",
    "没有找到",
    "材料不足",
    "不足以",
    "不能可靠回答",
    "证据不足",
    "insufficient",
    "not enough",
    "did not find",
)
CLAIM_TOKEN_STOPWORDS = {
    "根据",
    "你的",
    "个人知识库",
    "知识库",
    "现有材料",
    "材料",
    "来看",
    "可以",
    "说明",
    "提到",
    "只提到",
    "描述",
    "被描述为",
    "被称为",
    "又称",
    "也称",
    "也叫",
    "别名",
    "父亲",
    "母亲",
    "兄弟",
    "关系",
    "成就",
    "口号",
    "证据",
    "依据",
    "当前",
    "支持",
    "回答",
    "没有",
    "未提到",
    "未提及",
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
class SupportSpan:
    span_id: str
    hit_rank: int
    chunk_id: str
    source_id: str
    title: str
    citation: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "hit_rank": self.hit_rank,
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "title": self.title,
            "citation": self.citation,
            "text": self.text,
        }


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    support_span_ids: list[str]
    supported: bool
    missing_terms: list[str] = field(default_factory=list)
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "support_span_ids": self.support_span_ids,
            "supported": self.supported,
            "missing_terms": self.missing_terms,
            "reason": self.reason,
        }


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
        support_spans = select_support_spans(question, hits, grounding=grounding)
        grounding = refine_grounding_for_answer_units(question, grounding, support_spans)
        if not hits:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "no_evidence",
                "refused": True,
                "answer_status": "refused",
                "answer": NO_EVIDENCE_ANSWER,
                "citations": [],
                "grounding": _grounding_payload(grounding, support_spans, question=question),
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
                "grounding": _grounding_payload(grounding, support_spans, question=question),
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
                "grounding": _grounding_payload(grounding, support_spans, question=question),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": "base_url and model are required",
            }

        if grounding.answer_mode == "partial_compound":
            answer, used_span_ids = build_extractive_answer(question, support_spans, grounding)
            claim_checks = validate_grounded_claims(answer, support_spans, grounding=grounding)
            unsupported_claims = [check for check in claim_checks if not check.supported]
            return {
                "enabled": True,
                "skipped": False,
                "reason": "partial_compound_extractive",
                "refused": False,
                "answer_status": grounding.answer_status,
                "answer": answer,
                "citations": _citations_for_spans(support_spans, used_span_ids),
                "grounding": _grounding_payload(
                    grounding,
                    support_spans,
                    question=question,
                    claim_checks=claim_checks,
                    unsupported_claims=unsupported_claims,
                    used_span_ids=used_span_ids,
                ),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "retry_count": 0,
                "retry_reason": None,
                "recovered_from_thinking": False,
                "error": None,
            }

        try:
            chat = _chat_completion_with_recovery(
                self.config,
                messages=_build_messages(question, support_spans, grounding=grounding),
            )
            data = chat.data
            answer = chat.answer
            retry_count = chat.retry_count
            retry_reason = chat.retry_reason
            recovered_from_thinking = chat.recovered_from_thinking
            if not answer:
                return _empty_model_response_result(
                    started=started,
                    grounding=_grounding_payload(grounding, support_spans, question=question),
                    llm=self.config.safe_dict(),
                    citations=_citations_for_spans(support_spans, []),
                    chat=chat,
                )
            if _should_retry_llm_refusal(grounding, answer):
                retry_chat = _chat_completion_with_recovery(
                    self.config,
                    messages=_build_messages(question, support_spans, grounding=grounding, retry=True),
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
                        grounding=_grounding_payload(grounding, support_spans, question=question),
                        llm=self.config.safe_dict(),
                        citations=_citations_for_spans(support_spans, []),
                        chat=retry_chat,
                        retry_count=retry_count,
                        retry_reason=retry_reason,
                        recovered_from_thinking=recovered_from_thinking,
                    )

            if _should_return_llm_refusal(grounding, answer):
                contextual_answer, used_span_ids = _contextual_definition_extractive_answer(
                    question,
                    support_spans,
                    grounding=grounding,
                )
                if contextual_answer:
                    return {
                        "enabled": True,
                        "skipped": False,
                        "reason": "contextual_definition_extractive",
                        "refused": False,
                        "answer_status": grounding.answer_status,
                        "answer": contextual_answer,
                        "citations": _citations_for_spans(support_spans, used_span_ids),
                        "grounding": _grounding_payload(
                            grounding,
                            support_spans,
                            question=question,
                            used_span_ids=used_span_ids,
                        ),
                        "llm": self.config.safe_dict(),
                        "usage": data.get("usage") if isinstance(data, dict) else None,
                        "duration_ms": _elapsed_ms(started),
                        "retry_count": retry_count,
                        "retry_reason": retry_reason,
                        "recovered_from_thinking": recovered_from_thinking,
                        "error": None,
                    }
                if grounding.answer_mode != "refuse" and support_spans:
                    answer, used_span_ids = build_extractive_answer(question, support_spans, grounding)
                    claim_checks = validate_grounded_claims(answer, support_spans, grounding=grounding)
                    unsupported_claims = [check for check in claim_checks if not check.supported]
                    return {
                        "enabled": True,
                        "skipped": False,
                        "reason": "llm_refusal_extractive",
                        "refused": False,
                        "answer_status": grounding.answer_status,
                        "answer": answer,
                        "citations": _citations_for_spans(support_spans, used_span_ids),
                        "grounding": _grounding_payload(
                            grounding,
                            support_spans,
                            question=question,
                            claim_checks=claim_checks,
                            unsupported_claims=unsupported_claims,
                            used_span_ids=used_span_ids,
                        ),
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
                    "reason": "llm_refused",
                    "refused": True,
                    "answer_status": "llm_refused",
                    "answer": LLM_REFUSED_ANSWER,
                    "citations": [],
                    "grounding": _grounding_payload(grounding, support_spans, question=question),
                    "llm": self.config.safe_dict(),
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                    "duration_ms": _elapsed_ms(started),
                    "retry_count": retry_count,
                    "retry_reason": retry_reason,
                    "recovered_from_thinking": recovered_from_thinking,
                    "error": None,
                }

            if _should_prefer_contextual_definition_extractive_answer(question, grounding):
                contextual_answer, used_span_ids = _contextual_definition_extractive_answer(
                    question,
                    support_spans,
                    grounding=grounding,
                    relation_only=True,
                )
                if contextual_answer:
                    return {
                        "enabled": True,
                        "skipped": False,
                        "reason": "contextual_definition_extractive",
                        "refused": False,
                        "answer_status": grounding.answer_status,
                        "answer": contextual_answer,
                        "citations": _citations_for_spans(support_spans, used_span_ids),
                        "grounding": _grounding_payload(
                            grounding,
                            support_spans,
                            question=question,
                            used_span_ids=used_span_ids,
                        ),
                        "llm": self.config.safe_dict(),
                        "usage": data.get("usage") if isinstance(data, dict) else None,
                        "duration_ms": _elapsed_ms(started),
                        "retry_count": retry_count,
                        "retry_reason": retry_reason,
                        "recovered_from_thinking": recovered_from_thinking,
                        "error": None,
                    }

            claim_checks = validate_grounded_claims(answer, support_spans, grounding=grounding)
            unsupported_claims = [check for check in claim_checks if not check.supported]
            if unsupported_claims:
                retry_chat = _chat_completion_with_recovery(
                    self.config,
                    messages=_build_messages(
                        question,
                        support_spans,
                        grounding=grounding,
                        retry=True,
                        unsupported_claims=unsupported_claims,
                    ),
                )
                retry_count += 1 + retry_chat.retry_count
                retry_reason = _join_retry_reasons(
                    retry_reason,
                    retry_chat.retry_reason,
                    "unsupported_claim",
                )
                recovered_from_thinking = (
                    recovered_from_thinking or retry_chat.recovered_from_thinking
                )
                data = retry_chat.data
                answer = retry_chat.answer
                if not answer:
                    return _empty_model_response_result(
                        started=started,
                        grounding=_grounding_payload(grounding, support_spans, question=question),
                        llm=self.config.safe_dict(),
                        citations=_citations_for_spans(support_spans, []),
                        chat=retry_chat,
                        retry_count=retry_count,
                        retry_reason=retry_reason,
                        recovered_from_thinking=recovered_from_thinking,
                    )
                claim_checks = validate_grounded_claims(answer, support_spans, grounding=grounding)
                unsupported_claims = [check for check in claim_checks if not check.supported]

            if unsupported_claims:
                answer, used_span_ids = build_extractive_answer(question, support_spans, grounding)
                claim_checks = validate_grounded_claims(answer, support_spans, grounding=grounding)
                unsupported_claims = [check for check in claim_checks if not check.supported]
                return {
                    "enabled": True,
                    "skipped": False,
                    "reason": "unsupported_claim_fallback",
                    "refused": False,
                    "answer_status": "partial" if grounding.answer_status == "partial" else "answered",
                    "answer": answer,
                    "citations": _citations_for_spans(support_spans, used_span_ids),
                    "grounding": _grounding_payload(
                        grounding,
                        support_spans,
                        question=question,
                        claim_checks=claim_checks,
                        unsupported_claims=unsupported_claims,
                        used_span_ids=used_span_ids,
                    ),
                    "llm": self.config.safe_dict(),
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                    "duration_ms": _elapsed_ms(started),
                    "retry_count": retry_count,
                    "retry_reason": retry_reason,
                    "recovered_from_thinking": recovered_from_thinking,
                    "error": None,
                }

            used_span_ids = used_support_span_ids(answer, support_spans)
            return {
                "enabled": True,
                "skipped": False,
                "reason": None,
                "refused": False,
                "answer_status": grounding.answer_status,
                "answer": answer,
                "citations": _citations_for_spans(support_spans, used_span_ids),
                "grounding": _grounding_payload(
                    grounding,
                    support_spans,
                    question=question,
                    claim_checks=claim_checks,
                    unsupported_claims=unsupported_claims,
                    used_span_ids=used_span_ids,
                ),
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
                "citations": _citations_for_spans(support_spans, []),
                "grounding": _grounding_payload(grounding, support_spans, question=question),
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
    support_spans: list[SupportSpan],
    *,
    grounding: AnswerabilityResult | None = None,
    retry: bool = False,
    unsupported_claims: list[ClaimCheck] | None = None,
) -> list[dict[str, str]]:
    evidence = "\n\n".join(_format_support_span(span) for span in support_spans[:8])
    grounding = grounding or AnswerabilityResult(
        question_type=classify_question_type(question),
        answerable=bool(support_spans),
        checked_hit_count=len(support_spans),
        required_evidence="直接回答问题的检索证据",
        answer_mode="direct",
        answer_status="answered" if support_spans else "refused",
        missing_evidence=[] if support_spans else ["direct_evidence"],
    )
    unsupported_note = ""
    if unsupported_claims:
        bad_claims = "; ".join(check.claim for check in unsupported_claims[:3])
        unsupported_note = (
            "The previous answer contained unsupported claims: "
            f"{bad_claims}. Remove any claim not directly supported by the cited support span. "
            "Do not cite adjacent spans to combine unrelated facts. "
        )
    if unsupported_note:
        retry_note = unsupported_note
    elif retry and grounding.answer_mode == "contextual_definition":
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
                "Treat the support spans as the user's personal knowledge base. When the user asks in "
                "Chinese, prefer starting the answer with '根据你的个人知识库，'. "
                "First decide whether the support spans directly answer, partially answer, or cannot "
                "answer the question. "
                "The Question type, Answer mode, and Missing evidence fields are an upstream "
                "answerability decision. If Answer mode is not 'refuse' and Missing evidence is "
                "'none', treat the evidence as sufficient and do not answer with insufficient "
                "evidence. Extract the directly supported answer instead. If Answer mode is "
                "partial, provide the supported partial answer instead of refusing. "
                "Use the same language as the user when possible. Cite claims with bracketed "
                "support span ids like [S1] or [S2]. Every factual claim must be supported by the "
                "cited support span. Do not cite a span unless that exact span supports the claim. "
                "Preserve exact names, titles, metric names, and key phrases from the support spans "
                "when they answer the question. "
                "Keep the answer tightly focused on the entity and relation asked by the user. "
                "Do not merge adjacent facts across support spans. If one source contains several "
                "neighboring facts, include only the "
                "fact that answers the requested slot; do not mention siblings, parents, songs, "
                "reasons, aliases, or other adjacent facts unless the question asks for them. "
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
                f"Evidence support spans:\n{evidence}\n\n"
                "Answer by units when the question has multiple parts. If answerable, write a concise answer. If the answer mode is partial, write the "
                "partial answer and explicitly frame it as based on current materials. Include "
                "support span citations next to relevant claims. If not answerable from support spans, say what "
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


def build_support_spans(hits: list[RetrievalHit], *, max_chars: int = 500) -> list[SupportSpan]:
    spans: list[SupportSpan] = []
    seen_texts: set[str] = set()
    for hit_rank, hit in enumerate(hits[:8], 1):
        for sentence in split_evidence_sentences(hit.text):
            for text in _split_relation_clauses(sentence):
                normalized = normalize_whitespace(text)
                if not normalized:
                    continue
                dedupe_key = normalized.casefold()
                if dedupe_key in seen_texts:
                    continue
                seen_texts.add(dedupe_key)
                if len(normalized) > max_chars:
                    normalized = normalized[:max_chars].rstrip() + "..."
                spans.append(
                    SupportSpan(
                        span_id=f"S{len(spans) + 1}",
                        hit_rank=hit_rank,
                        chunk_id=hit.chunk_id,
                        source_id=hit.source_id,
                        title=hit.title,
                        citation=hit.citation,
                        text=normalized,
                    )
                )
    return spans


def select_support_spans(
    question: str,
    hits: list[RetrievalHit],
    *,
    grounding: AnswerabilityResult,
    limit: int = 8,
) -> list[SupportSpan]:
    spans = build_support_spans(hits)
    if not spans:
        return []
    terms = [
        *evidence_query_terms(question),
        *grounding.matched_cues,
        *answer_unit_terms(question),
    ]
    terms = dedupe_preserve_order([term for term in terms if _is_valid_evidence_query_term(term.casefold())])
    if not terms:
        return spans[:limit]

    scored: list[tuple[int, int, SupportSpan]] = []
    for index, span in enumerate(spans):
        score = _support_span_score(question, span, terms)
        if score:
            scored.append((score, -index, span))
    if not scored:
        return spans[:limit]
    scored.sort(reverse=True)
    selected = [span for _, _, span in scored[:limit]]
    return _expand_support_spans_with_bridge_terms(selected, spans, limit=limit)


def refine_grounding_for_answer_units(
    question: str,
    grounding: AnswerabilityResult,
    support_spans: list[SupportSpan],
) -> AnswerabilityResult:
    if not grounding.answerable or grounding.answer_status != "answered":
        return grounding
    units = split_answer_units(question)
    if len(units) <= 1:
        return grounding
    matched_units = 0
    missing_units = 0
    for unit in units:
        if _matched_span_ids_for_unit(unit, support_spans):
            matched_units += 1
        else:
            missing_units += 1
    if matched_units == 0 or missing_units == 0:
        return grounding
    missing_evidence = dedupe_preserve_order([*grounding.missing_evidence, "answer_unit_evidence"])
    return replace(
        grounding,
        answer_status="partial",
        answer_mode="partial_compound",
        missing_evidence=missing_evidence,
        required_evidence=f"{grounding.required_evidence}; 每个子问题的直接支持片段",
    )


def answer_unit_terms(question: str) -> list[str]:
    terms: list[str] = []
    for unit in split_answer_units(question):
        terms.extend(extract_query_terms(unit))
        terms.extend(tokenize(unit))
    return _valid_evidence_query_terms(terms)


def split_answer_units(question: str) -> list[str]:
    normalized = normalize_whitespace(question)
    parts = [
        part.strip()
        for part in re.split(r"[？?。；;]|(?:，|,)(?=[^，,。；;？?]{1,40}(?:谁|什么|哪些|哪|为什么|为何|为啥|how|what|who|which))", normalized)
        if part.strip()
    ]
    return parts or [normalized]


def validate_grounded_claims(
    answer: str,
    support_spans: list[SupportSpan],
    *,
    grounding: AnswerabilityResult,
) -> list[ClaimCheck]:
    if not answer:
        return []
    if grounding.question_type in {"evaluative", "summary", "list", "procedural"}:
        return []
    checks: list[ClaimCheck] = []
    span_by_id = {span.span_id: span for span in support_spans}
    for claim in split_answer_claims(answer):
        if _is_missing_evidence_statement(claim):
            checks.append(ClaimCheck(claim=claim, support_span_ids=used_support_span_ids(claim, support_spans), supported=True))
            continue
        cited_ids = used_support_span_ids(claim, support_spans)
        if not cited_ids:
            checks.append(
                ClaimCheck(
                    claim=claim,
                    support_span_ids=[],
                    supported=False,
                    reason="missing_support_citation",
                )
            )
            continue
        if grounding.question_type == "exact":
            claim_without_citations = re.sub(r"\[(?:S)?\d+\]", " ", claim, flags=re.IGNORECASE)
            missing_numbers = [
                number
                for number in re.findall(r"\d+(?:\.\d+)?", claim_without_citations)
                if not _term_in_any_cited_span(number, cited_ids, span_by_id)
            ]
            checks.append(
                ClaimCheck(
                    claim=claim,
                    support_span_ids=cited_ids,
                    supported=not missing_numbers,
                    missing_terms=missing_numbers,
                    reason=None if not missing_numbers else "missing_exact_value_in_cited_span",
                )
            )
            continue
        if not _looks_like_relation_claim(claim):
            checks.append(ClaimCheck(claim=claim, support_span_ids=cited_ids, supported=True))
            continue
        terms = claim_key_terms(claim)
        support_span_id = _single_span_supporting_terms(terms, cited_ids, span_by_id)
        checks.append(
            ClaimCheck(
                claim=claim,
                support_span_ids=cited_ids,
                supported=support_span_id is not None,
                missing_terms=[] if support_span_id else _missing_terms_from_single_span(terms, cited_ids, span_by_id),
                reason=None if support_span_id else "relation_terms_not_supported_by_one_span",
            )
        )
    return checks


def build_extractive_answer(
    question: str,
    support_spans: list[SupportSpan],
    grounding: AnswerabilityResult,
) -> tuple[str, list[str]]:
    if not support_spans:
        return NO_EVIDENCE_ANSWER, []
    units = split_answer_units(question)
    used: list[str] = []
    pieces: list[str] = []
    for unit in units:
        unit_terms = _valid_evidence_query_terms([*extract_query_terms(unit), *tokenize(unit)])
        selected = _best_span_for_terms(support_spans, unit_terms, exclude=used)
        if selected is None and used:
            selected = _bridge_span_for_used_spans(support_spans, used)
        if selected:
            used.append(selected.span_id)
            pieces.append(selected.text)
    if not pieces:
        selected = support_spans[0]
        used.append(selected.span_id)
        pieces.append(selected.text)
    missing_note = ""
    if grounding.answer_status == "partial" and _looks_chinese(question):
        missing_note = "但材料不足以完整回答全部问题。"
    elif grounding.answer_status == "partial":
        missing_note = "The materials do not fully answer every part."
    prefix = "根据你的个人知识库，" if _looks_chinese(question) else "Based on your knowledge base, "
    statements = [
        f"{piece.rstrip('。！？!?；;')} [{span_id}]。"
        for piece, span_id in zip(pieces, used, strict=False)
    ]
    if missing_note:
        statements.append(missing_note)
    return f"{prefix}{' '.join(statements)}", used


def _best_span_for_terms(
    support_spans: list[SupportSpan],
    terms: list[str],
    *,
    exclude: list[str],
) -> SupportSpan | None:
    if not terms:
        return next((span for span in support_spans if span.span_id not in exclude), None)
    scored: list[tuple[int, int, SupportSpan]] = []
    for index, span in enumerate(support_spans):
        if span.span_id in exclude:
            continue
        score = _support_span_score("", span, terms)
        if score:
            scored.append((score, -index, span))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def _support_span_score(question: str, span: SupportSpan, terms: list[str]) -> int:
    text = span.text.casefold()
    score = sum(1 for term in terms if term.casefold() in text)
    if question and _contains_any(question.casefold(), RELATION_QUESTION_CUES):
        score += sum(1 for cue in RELATION_ANSWER_CUES if cue.casefold() in text)
    return score


def _bridge_span_for_used_spans(
    support_spans: list[SupportSpan],
    used_ids: list[str],
) -> SupportSpan | None:
    used_spans = [span for span in support_spans if span.span_id in used_ids]
    bridge_terms = _bridge_terms_from_spans(used_spans)
    if not bridge_terms:
        return None
    for span in support_spans:
        if span.span_id in used_ids:
            continue
        if any(_span_is_about_bridge_term(span.text, term) for term in bridge_terms):
            return span
    return None


def _format_support_span(span: SupportSpan) -> str:
    return (
        f"[{span.span_id}] {span.title}\n"
        f"Citation: {span.citation}\n"
        f"Source: {span.source_id}\n"
        f"Chunk: {span.chunk_id}\n"
        f"Text: {span.text}"
    )


def _citations_for_spans(support_spans: list[SupportSpan], used_ids: list[str]) -> list[dict[str, Any]]:
    if not used_ids:
        return []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, span in enumerate(support_spans, 1):
        if span.span_id not in used_ids or span.span_id in seen:
            continue
        output.append(
            {
                "rank": index,
                "title": span.title,
                "citation": span.citation,
                "chunk_id": span.chunk_id,
                "source_id": span.source_id,
                "support_span_id": span.span_id,
                "support_text": span.text,
            }
        )
        seen.add(span.span_id)
    return output


def _grounding_payload(
    grounding: AnswerabilityResult,
    support_spans: list[SupportSpan],
    *,
    question: str | None = None,
    claim_checks: list[ClaimCheck] | None = None,
    unsupported_claims: list[ClaimCheck] | None = None,
    used_span_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload = grounding.as_dict()
    payload.update(
        {
            "answer_units": build_answer_units(question, grounding, support_spans),
            "support_spans": [span.as_dict() for span in support_spans],
            "claim_checks": [check.as_dict() for check in claim_checks or []],
            "unsupported_claims": [check.as_dict() for check in unsupported_claims or []],
            "used_support_span_ids": used_span_ids or [],
        }
    )
    return payload


def build_answer_units(
    question: str | None,
    grounding: AnswerabilityResult,
    support_spans: list[SupportSpan],
) -> list[dict[str, Any]]:
    units = split_answer_units(question or "") if question else [""]
    output: list[dict[str, Any]] = []
    for index, unit in enumerate(units, 1):
        matched_span_ids = _matched_span_ids_for_unit(unit, support_spans)
        if matched_span_ids:
            status = "answered"
            missing = []
        else:
            status = "no_evidence"
            missing = ["unit_evidence"]
        output.append(
            {
                "unit_id": f"U{index}",
                "question": unit,
                "question_type": classify_question_type(unit) if unit else grounding.question_type,
                "status": status,
                "answer_mode": grounding.answer_mode,
                "missing_evidence": missing,
                "matched_support_span_ids": matched_span_ids,
            }
        )
    return output


def _matched_span_ids_for_unit(unit: str, support_spans: list[SupportSpan]) -> list[str]:
    unit_terms = _valid_evidence_query_terms([*extract_query_terms(unit), *tokenize(unit)])
    if not unit_terms:
        return [span.span_id for span in support_spans]
    return [
        span.span_id
        for span in support_spans
        if any(term.casefold() in span.text.casefold() for term in unit_terms)
    ]


def _expand_support_spans_with_bridge_terms(
    selected: list[SupportSpan],
    all_spans: list[SupportSpan],
    *,
    limit: int,
) -> list[SupportSpan]:
    output: list[SupportSpan] = []
    seen: set[str] = set()
    for span in selected:
        if span.span_id in seen:
            continue
        output.append(span)
        seen.add(span.span_id)
        if len(output) >= limit:
            return output

    bridge_terms = _bridge_terms_from_spans(selected)
    if not bridge_terms:
        return output
    selected_hit_ranks = {span.hit_rank for span in selected}
    for span in all_spans:
        if span.span_id in seen or span.hit_rank not in selected_hit_ranks:
            continue
        if any(_span_is_about_bridge_term(span.text, term) for term in bridge_terms):
            output.append(span)
            seen.add(span.span_id)
            if len(output) >= limit:
                break
    return output


def _bridge_terms_from_spans(spans: list[SupportSpan]) -> list[str]:
    terms: list[str] = []
    for span in spans:
        terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", span.text))
    return dedupe_preserve_order([term for term in terms if term.casefold() not in CLAIM_TOKEN_STOPWORDS])


def _span_is_about_bridge_term(text: str, term: str) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized:
        return False
    escaped = re.escape(term)
    return bool(
        re.search(
            rf"^(?:关于|有关|提到)?\s*{escaped}(?:\s|$|[，,。；;：:]|是|的|被|又|也|在)",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def used_support_span_ids(answer: str, support_spans: list[SupportSpan]) -> list[str]:
    valid = {span.span_id for span in support_spans}
    output: list[str] = []
    for match in re.finditer(r"\[(?:S)?(\d+)\]", answer, flags=re.IGNORECASE):
        span_id = f"S{int(match.group(1))}"
        if span_id in valid and span_id not in output:
            output.append(span_id)
    return output


def split_answer_claims(answer: str) -> list[str]:
    normalized = normalize_whitespace(answer)
    normalized = re.sub(
        r"([。！？!?；;])\s*((?:\[(?:S)?\d+\]\s*)+)",
        r" \2\1",
        normalized,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"(?<=[。！？!?；;])\s+|\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _split_relation_clauses(sentence: str) -> list[str]:
    normalized = normalize_whitespace(sentence)
    if "," in normalized and re.search(r"(包括|包含|有|列出)[:：][^。！？!?；;]+,", normalized):
        return [normalized]
    delimiters = "，,"
    if not any(delimiter in normalized for delimiter in delimiters):
        return [normalized]
    parts = [part.strip() for part in re.split(r"[，,]", normalized) if part.strip()]
    if len(parts) <= 1:
        return [normalized]
    relation_parts = sum(1 for part in parts if _looks_like_relation_claim(part) or part.startswith(("因为", "由于", "所以")))
    if relation_parts >= 2:
        return parts
    return [normalized]


def _term_in_any_cited_span(term: str, span_ids: list[str], span_by_id: dict[str, SupportSpan]) -> bool:
    key = term.casefold()
    return any(key in span_by_id[span_id].text.casefold() for span_id in span_ids if span_id in span_by_id)


def _single_span_supporting_terms(
    terms: list[str],
    span_ids: list[str],
    span_by_id: dict[str, SupportSpan],
) -> str | None:
    if not terms:
        return span_ids[0] if span_ids else None
    for span_id in span_ids:
        span = span_by_id.get(span_id)
        if not span:
            continue
        text = span.text.casefold()
        if all(term.casefold() in text for term in terms):
            return span_id
    return None


def _missing_terms_from_single_span(
    terms: list[str],
    span_ids: list[str],
    span_by_id: dict[str, SupportSpan],
) -> list[str]:
    if not terms:
        return []
    best_supported = 0
    best_missing = terms
    for span_id in span_ids:
        span = span_by_id.get(span_id)
        if not span:
            continue
        text = span.text.casefold()
        supported = [term for term in terms if term.casefold() in text]
        missing = [term for term in terms if term.casefold() not in text]
        if len(supported) > best_supported:
            best_supported = len(supported)
            best_missing = missing
    return best_missing


def _is_missing_evidence_statement(claim: str) -> bool:
    normalized = claim.casefold()
    return any(cue.casefold() in normalized for cue in MISSING_EVIDENCE_CUES)


def _looks_like_relation_claim(claim: str) -> bool:
    normalized = claim.casefold()
    return any(cue.casefold() in normalized for cue in CLAIM_RELATION_CUES)


def claim_key_terms(claim: str) -> list[str]:
    clean = re.sub(r"\[(?:S)?\d+\]", " ", claim, flags=re.IGNORECASE)
    for word in sorted(CLAIM_TOKEN_STOPWORDS, key=len, reverse=True):
        clean = clean.replace(word, " ")
    for cue in CLAIM_RELATION_CUES:
        clean = clean.replace(cue, " ")
    terms: list[str] = []
    for match in re.finditer(r"[a-z0-9][a-z0-9_.-]{1,}", clean.casefold()):
        term = match.group(0).strip("_.-")
        if _is_valid_claim_term(term):
            terms.append(term)
    for match in re.finditer(r"[\u4e00-\u9fff]{2,}", clean):
        chunk = match.group(0)
        for part in re.split(r"[，。；、,/的和与及\s]+", chunk):
            if _is_valid_claim_term(part):
                terms.append(part)
    return dedupe_preserve_order(terms)


def _is_valid_claim_term(term: str) -> bool:
    term = term.strip()
    return bool(
        term
        and len(term) >= 2
        and term.casefold() not in STOP_QUERY_TERMS
        and term not in CLAIM_TOKEN_STOPWORDS
        and term not in RELATION_TERMS_FOR_REMOVAL
        and term not in PRONOUN_QUERY_TERMS
    )


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
        {
            "rank": index,
            "title": hit.title,
            "citation": hit.citation,
            "chunk_id": hit.chunk_id,
            "source_id": hit.source_id,
        }
        for index, hit in enumerate(hits[:8], 1)
    ]


def _contextual_definition_extractive_answer(
    question: str,
    support_spans: list[SupportSpan],
    *,
    grounding: AnswerabilityResult,
    relation_only: bool = False,
) -> tuple[str | None, list[str]]:
    if grounding.answer_mode != "contextual_definition" or not grounding.matched_cues:
        return None, []

    cue = grounding.matched_cues[0]
    cue_key = cue.casefold()
    for span in support_spans[:8]:
        if cue_key not in span.text.casefold():
            continue
        sentence = normalize_whitespace(span.text)[:300]
        relation = _contextual_relation_for_cue(cue, sentence)
        if relation:
            if _looks_chinese(question):
                return f"根据你的个人知识库，{relation} [{span.span_id}]", [span.span_id]
            return f"In your personal knowledge base, {relation} [{span.span_id}]", [span.span_id]
        if relation_only:
            continue
        if _looks_chinese(question):
            return f"根据你的个人知识库，{cue}在材料中被描述为：{sentence} [{span.span_id}]", [span.span_id]
        return f"In your personal knowledge base, {cue} is described as: {sentence} [{span.span_id}]", [span.span_id]
    return None, []


def _should_prefer_contextual_definition_extractive_answer(
    question: str,
    grounding: AnswerabilityResult,
) -> bool:
    if grounding.answer_mode != "contextual_definition" or not grounding.matched_cues:
        return False
    cue = grounding.matched_cues[0]
    normalized = re.sub(r"[\s\"'“”‘’《》？?。！!，,、：:；;（）()]+", "", question.casefold())
    cue_key = re.sub(r"[\s\"'“”‘’《》]+", "", cue.casefold())
    return normalized in {
        f"{cue_key}是什么",
        f"什么是{cue_key}",
        f"{cue_key}是啥",
        f"{cue_key}叫啥",
    }


def _contextual_relation_for_cue(cue: str, sentence: str) -> str | None:
    escaped = re.escape(cue)
    patterns = (
        (rf"([\u4e00-\u9fffA-Za-z0-9_.-]{{1,24}})最喜欢的歌是{escaped}", "{cue}是{subject}最喜欢的歌"),
        (rf"([\u4e00-\u9fffA-Za-z0-9_.-]{{1,24}})最爱的歌(?:是|叫){escaped}", "{cue}是{subject}最爱的歌"),
        (rf"([\u4e00-\u9fffA-Za-z0-9_.-]{{1,24}})又被称为{escaped}", "{cue}是{subject}的别名"),
        (rf"{escaped}是([\u4e00-\u9fffA-Za-z0-9_.-]{{1,24}})的兄弟", "{cue}是{subject}的兄弟"),
        (rf"{escaped}是([\u4e00-\u9fffA-Za-z0-9_.-]{{1,24}})的结拜大哥", "{cue}是{subject}的结拜大哥"),
    )
    for pattern, template in patterns:
        match = re.search(pattern, sentence)
        if not match:
            continue
        subject = match.group(1).strip(" ，,。；;：:")
        if not subject:
            continue
        return template.format(cue=cue, subject=subject)
    return None


def _first_sentence_containing(text: str, cue_key: str) -> str | None:
    for sentence in split_evidence_sentences(text):
        if cue_key in sentence.casefold():
            return sentence
    normalized = normalize_whitespace(text)
    if cue_key in normalized.casefold():
        return normalized
    return None


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
