from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from legal_rag.schema import RetrievalHit
from legal_rag.text import normalize_whitespace

INSUFFICIENT_CAUSAL_EVIDENCE = (
    "知识库检索到了相关内容，但没有明确说明原因，不能可靠回答这个“为什么”问题。"
)
INSUFFICIENT_EXACT_EVIDENCE = (
    "知识库检索到了相关内容，但没有明确的数字、日期或精确事实，不能可靠回答这个精确问题。"
)
INSUFFICIENT_SCOPE_EVIDENCE = (
    "知识库检索到了相关内容，但没有覆盖问题中的限定范围，不能可靠回答。"
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
    "insufficient evidence",
    "not enough evidence",
    "not enough information",
    "i don't know",
    "cannot answer",
    "can't answer",
)
RETRYABLE_LLM_REFUSAL_TYPES = {"evaluative", "list", "summary", "procedural"}
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
RELATION_TERMS_FOR_REMOVAL = {"关系", "什么关系"}
PRONOUN_QUERY_TERMS = {"那他", "那她", "那它", "这个人", "那个人", "这人", "那人", "他", "她", "它", "这个", "那个"}
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
    " 是 ",
    "是",
)
MISSING_EVIDENCE_CUES = (
    "缺少",
    "没有",
    "未提到",
    "没有找到",
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

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "api_key_configured": bool(self.api_key),
        }


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

        missing_scope = missing_required_scope_terms(question, hits[:8])
        if missing_scope:
            return AnswerabilityResult(
                question_type=question_type,
                answerable=False,
                checked_hit_count=checked_hit_count,
                required_evidence="覆盖问题限定范围和谓词关系的直接证据",
                answer_mode="refuse",
                answer_status="refused",
                matched_cues=matched_query_terms_in_hits(extract_query_terms(question), hits[:8]),
                missing_evidence=missing_scope,
                reason="missing_required_query_scope",
            )

        matched_terms = matched_query_terms_in_hits(extract_query_terms(question), hits[:8])
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
        if not hits:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "no_evidence",
                "refused": True,
                "answer_status": "refused",
                "answer": NO_EVIDENCE_ANSWER,
                "citations": [],
                "grounding": grounding.as_dict(),
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
                "grounding": grounding.as_dict(),
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
                "grounding": _grounding_payload(grounding, support_spans),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": "base_url and model are required",
            }

        payload = {
            "model": self.config.model,
            "messages": _build_messages(question, support_spans, grounding=grounding),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key.strip():
            headers["Authorization"] = f"Bearer {self.config.api_key.strip()}"

        try:
            response = httpx.post(
                _chat_completions_url(self.config.base_url),
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            answer = _extract_answer_text(data)
            retry_count = 0
            if _should_retry_llm_refusal(grounding, answer):
                retry_count = 1
                retry_payload = {
                    **payload,
                    "messages": _build_messages(question, support_spans, grounding=grounding, retry=True),
                }
                response = httpx.post(
                    _chat_completions_url(self.config.base_url),
                    json=retry_payload,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                answer = _extract_answer_text(data)

            if _looks_like_llm_refusal(answer) and grounding.question_type in RETRYABLE_LLM_REFUSAL_TYPES:
                return {
                    "enabled": True,
                    "skipped": False,
                    "reason": "llm_refused",
                    "refused": True,
                    "answer_status": "llm_refused",
                    "answer": LLM_REFUSED_ANSWER,
                    "citations": [],
                    "grounding": _grounding_payload(grounding, support_spans),
                    "llm": self.config.safe_dict(),
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                    "duration_ms": _elapsed_ms(started),
                    "retry_count": retry_count,
                    "error": None,
                }

            claim_checks = validate_grounded_claims(answer, support_spans, grounding=grounding)
            unsupported_claims = [check for check in claim_checks if not check.supported]
            if unsupported_claims and retry_count == 0:
                retry_count = 1
                retry_payload = {
                    **payload,
                    "messages": _build_messages(
                        question,
                        support_spans,
                        grounding=grounding,
                        retry=True,
                        unsupported_claims=unsupported_claims,
                    ),
                }
                response = httpx.post(
                    _chat_completions_url(self.config.base_url),
                    json=retry_payload,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                answer = _extract_answer_text(data)
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
                        claim_checks=claim_checks,
                        unsupported_claims=unsupported_claims,
                        used_span_ids=used_span_ids,
                    ),
                    "llm": self.config.safe_dict(),
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                    "duration_ms": _elapsed_ms(started),
                    "retry_count": retry_count,
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
                    claim_checks=claim_checks,
                    unsupported_claims=unsupported_claims,
                    used_span_ids=used_span_ids,
                ),
                "llm": self.config.safe_dict(),
                "usage": data.get("usage") if isinstance(data, dict) else None,
                "duration_ms": _elapsed_ms(started),
                "retry_count": retry_count,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - answer layer must fail open.
            return {
                "enabled": True,
                "skipped": True,
                "reason": "llm_error",
                "refused": False,
                "answer_status": "error",
                "answer": None,
                "citations": _citations_for_spans(support_spans, []),
                "grounding": _grounding_payload(grounding, support_spans),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
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

        payload = {
            "model": self.config.model,
            "messages": _build_general_messages(question, fallback_from_kb=fallback_from_kb),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key.strip():
            headers["Authorization"] = f"Bearer {self.config.api_key.strip()}"

        try:
            response = httpx.post(
                _chat_completions_url(self.config.base_url),
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            answer = _extract_answer_text(data)
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
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - general answer layer must fail open as an error.
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
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
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
        answer_status="answered",
    )
    unsupported_note = ""
    if unsupported_claims:
        bad = "; ".join(check.claim for check in unsupported_claims[:3])
        unsupported_note = (
            "The previous answer contained unsupported claims: "
            f"{bad}. Remove any claim not directly supported by a cited support span. "
        )
    retry_note = (
        "The previous answer was too conservative. The evidence is usable for this question "
        "type, so produce the allowed grounded or partial answer instead of replying only "
        "that evidence is insufficient. "
        if retry and not unsupported_claims
        else unsupported_note
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a grounded RAG answer layer. Answer only from the provided support spans, "
                "not from the original chunks and not from general knowledge. Treat the support spans "
                "as the user's personal knowledge base. When the user asks in Chinese, prefer starting "
                "the answer with '根据你的个人知识库，'. First decide whether the support spans directly "
                "answer, partially answer, or cannot answer the question. Use the same language as the "
                "user when possible. Cite every factual claim with bracketed support span ids like [S1] "
                "or [S2]. Every factual claim must be supported by the cited span. Do not cite a span "
                "unless that exact span supports the claim. Do not merge adjacent facts across spans: "
                "aliases, relationships, relatives, achievements, slogans, dates, numbers, and named "
                "entities must come from the same cited support span unless another cited span explicitly "
                "bridges the relation. If the question contains a domain, game, product, jurisdiction, "
                "time period, or other scope constraint, the support spans must explicitly cover that "
                "scope; otherwise state what evidence is missing. For why/reason/causal questions, "
                "answer only when the support spans explicitly state a cause, reason, motive, or causal "
                "chain. If the evidence only repeats a related fact or conclusion without explaining why, "
                "say the evidence is insufficient. For evaluate/assess/opinion-style questions, synthesize "
                "an evidence-backed assessment from descriptive facts in the support spans. Start with "
                "'根据现有材料来看' when the user asks in Chinese. Do not require the evidence to literally "
                "contain an evaluation label, but do not add outside opinions. For list questions, list "
                "only items mentioned in support spans and say they are what the current materials mention. "
                "For comparison questions, compare only sides that have support spans; if one side is "
                "missing, give the supported side and state the gap. For procedural questions, give known "
                "steps only; if the complete procedure is missing, state that the materials only support "
                "a partial procedure. For exact number/date/time questions, answer only when exact evidence "
                "is present. Do not invent policy, law, numbers, timelines, URLs, or commitments. Keep "
                "answers short: use one compact paragraph by default, or at most three bullets when a list "
                f"or procedure is necessary. {retry_note}"
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
                f"Evidence:\n{evidence}\n\n"
                "Answer by units when the question has multiple parts. If answerable, write a concise "
                "answer. If the answer mode is partial, write the partial answer and explicitly frame it "
                "as based on current materials. Cite claims next to the relevant support span ids. If not "
                "answerable from support spans, say what specific evidence is missing."
            ),
        },
    ]


def _build_general_messages(question: str, *, fallback_from_kb: bool = False) -> list[dict[str, str]]:
    fallback_instruction = (
        "用户的问题已经先检索个人知识库，但没有找到直接相关内容。"
        "如果用户使用中文，回答必须以“个人知识库没有找到直接相关内容；根据一般常识，”开头；"
        "如果用户使用其他语言，也要先说明 personal knowledge base did not contain direct evidence. "
        "只能使用通用模型知识回答，不要编造个人知识库事实，不要生成引用。"
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
    for hit_rank, hit in enumerate(hits[:8], 1):
        for raw in split_evidence_sentences(hit.text):
            for text in _split_relation_clauses(raw):
                text = normalize_whitespace(text)
                if not text:
                    continue
                if len(text) > max_chars:
                    text = text[:max_chars].rstrip() + "..."
                span_id = f"S{len(spans) + 1}"
                spans.append(
                    SupportSpan(
                        span_id=span_id,
                        hit_rank=hit_rank,
                        chunk_id=hit.chunk_id,
                        source_id=hit.source_id,
                        title=hit.title,
                        citation=hit.citation,
                        text=text,
                    )
                )
    return spans


def select_support_spans(
    question: str,
    hits: list[RetrievalHit],
    *,
    grounding: AnswerabilityResult | None = None,
    limit: int = 8,
) -> list[SupportSpan]:
    spans = build_support_spans(hits)
    if not spans:
        return []
    terms = extract_query_terms(question)
    if grounding:
        terms.extend(grounding.matched_cues)
    terms.extend(required_scope_terms(question))
    terms = dedupe_preserve_order([term for term in terms if _is_valid_query_term(term)])
    if not terms:
        return spans[:limit]

    scored: list[tuple[int, int, SupportSpan]] = []
    for index, span in enumerate(spans):
        normalized = span.text.casefold()
        score = sum(1 for term in terms if term.casefold() in normalized)
        if score:
            scored.append((score, -index, span))
    if not scored:
        return spans[:limit]
    scored.sort(reverse=True)
    return [span for _, _, span in scored[:limit]]


def validate_grounded_claims(
    answer: str,
    support_spans: list[SupportSpan],
    *,
    grounding: AnswerabilityResult,
) -> list[ClaimCheck]:
    if grounding.question_type not in {"factual", "definition", "exact", "comparison", "causal", "other"}:
        return []
    checks: list[ClaimCheck] = []
    span_by_id = {span.span_id: span for span in support_spans}
    for claim in split_answer_claims(answer):
        cited_ids = used_support_span_ids(claim, support_spans)
        if not cited_ids:
            continue
        if _is_missing_evidence_statement(claim):
            checks.append(ClaimCheck(claim=claim, support_span_ids=cited_ids, supported=True))
            continue
        if grounding.question_type == "exact":
            missing_numbers = [num for num in re.findall(r"\d+(?:\.\d+)?", claim) if not _term_in_spans(num, cited_ids, span_by_id)]
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
        tokens = claim_key_terms(claim)
        missing = [token for token in tokens if not _term_in_spans(token, cited_ids, span_by_id)]
        checks.append(
            ClaimCheck(
                claim=claim,
                support_span_ids=cited_ids,
                supported=not missing,
                missing_terms=missing,
                reason=None if not missing else "missing_claim_term_in_cited_span",
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
    used = [span.span_id for span in support_spans[: min(3, len(support_spans))]]
    pieces = [span.text for span in support_spans if span.span_id in used]
    prefix = "根据你的个人知识库，" if _looks_chinese(question) else "Based on your knowledge base, "
    if grounding.answer_status == "partial":
        gap = "；但材料不足以完整回答全部问题" if _looks_chinese(question) else "; the materials do not fully answer every part"
    else:
        gap = ""
    citations = " ".join(f"[{span_id}]" for span_id in used)
    return f"{prefix}{'；'.join(pieces)}{gap}。{citations}", used


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
    claim_checks: list[ClaimCheck] | None = None,
    unsupported_claims: list[ClaimCheck] | None = None,
    used_span_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload = grounding.as_dict()
    payload.update(
        {
            "answer_units": build_answer_units(grounding),
            "support_spans": [span.as_dict() for span in support_spans],
            "claim_checks": [check.as_dict() for check in claim_checks or []],
            "unsupported_claims": [check.as_dict() for check in unsupported_claims or []],
            "used_support_span_ids": used_span_ids or [],
        }
    )
    return payload


def build_answer_units(grounding: AnswerabilityResult) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": "U1",
            "question_type": grounding.question_type,
            "answer_status": grounding.answer_status,
            "answer_mode": grounding.answer_mode,
            "missing_evidence": grounding.missing_evidence,
        }
    ]


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
    parts = re.split(r"(?<=[。！？!?；;])\s+|\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _split_relation_clauses(sentence: str) -> list[str]:
    normalized = normalize_whitespace(sentence)
    if "，" not in normalized and "," not in normalized:
        return [normalized]
    delimiter = "，" if "，" in normalized else ","
    parts = [part.strip() for part in normalized.split(delimiter) if part.strip()]
    if len(parts) <= 1:
        return [normalized]
    if any(_looks_like_relation_claim(part) for part in parts):
        return parts
    return [normalized]


def required_scope_terms(question: str) -> list[str]:
    text = normalize_whitespace(question)
    terms: list[str] = []
    for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,})的[^？?。；;]*(?:是谁|是什么|有哪些|多少|几个|哪|who|what|which)", text, flags=re.IGNORECASE):
        candidate = match.group(1).strip()
        if _is_valid_query_term(candidate):
            terms.append(candidate)
    return dedupe_preserve_order(terms)


def missing_required_scope_terms(question: str, hits: list[RetrievalHit]) -> list[str]:
    terms = required_scope_terms(question)
    if not terms:
        return []
    haystack = "\n".join(f"{hit.title}\n{hit.citation}\n{hit.text}" for hit in hits).casefold()
    return [term for term in terms if term.casefold() not in haystack]


def _term_in_spans(term: str, span_ids: list[str], span_by_id: dict[str, SupportSpan]) -> bool:
    normalized_term = term.casefold()
    return any(normalized_term in span_by_id[span_id].text.casefold() for span_id in span_ids if span_id in span_by_id)


def _is_missing_evidence_statement(claim: str) -> bool:
    normalized = claim.casefold()
    return any(cue.casefold() in normalized for cue in MISSING_EVIDENCE_CUES)


def _looks_like_relation_claim(claim: str) -> bool:
    normalized = claim.casefold()
    return any(cue.casefold() in normalized for cue in CLAIM_RELATION_CUES)


def claim_key_terms(claim: str) -> list[str]:
    clean = re.sub(r"\[(?:S)?\d+\]", " ", claim, flags=re.IGNORECASE)
    terms: list[str] = []
    for match in re.finditer(r"[a-z0-9][a-z0-9_.-]{1,}", clean.casefold()):
        term = match.group(0).strip("_.-")
        if _is_valid_claim_term(term):
            terms.append(term)
    for match in re.finditer(r"[一-鿿]{2,}", clean):
        chunk = match.group(0)
        parts = [part for part in re.split(r"[，。；、,/的和与及\s]+", chunk) if part]
        for part in parts:
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


def _chat_completions_url(base_url: str) -> str:
    clean = base_url.strip().rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def _extract_answer_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        text = choices[0].get("text") if isinstance(choices[0], dict) else None
        if isinstance(text, str):
            return text.strip()
    return ""


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
    return bool(re.search(r"[一-鿿]", value))


def refusal_answer_for(grounding: AnswerabilityResult) -> str:
    if grounding.reason == "insufficient_causal_evidence":
        return INSUFFICIENT_CAUSAL_EVIDENCE
    if grounding.reason == "insufficient_exact_evidence":
        return INSUFFICIENT_EXACT_EVIDENCE
    if grounding.reason == "missing_required_query_scope":
        return INSUFFICIENT_SCOPE_EVIDENCE
    return NO_EVIDENCE_ANSWER


def _should_retry_llm_refusal(grounding: AnswerabilityResult, answer: str) -> bool:
    return grounding.question_type in RETRYABLE_LLM_REFUSAL_TYPES and _looks_like_llm_refusal(answer)


def _looks_like_llm_refusal(answer: str) -> bool:
    normalized = normalize_whitespace(answer).casefold()
    if not normalized:
        return True
    return any(cue.casefold() in normalized for cue in LLM_REFUSAL_CUES)


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
    query_terms = extract_query_terms(question)
    matched: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        for sentence in split_evidence_sentences(hit.text):
            normalized_sentence = sentence.casefold()
            if query_terms and not any(term.casefold() in normalized_sentence for term in query_terms):
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
    for match in re.finditer(r"[一-鿿]{2,}", normalized):
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


def split_evidence_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    return [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s+|\n+", normalized) if part.strip()]


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
