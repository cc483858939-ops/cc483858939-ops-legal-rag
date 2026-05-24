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
                "grounding": grounding.as_dict(),
                "llm": self.config.safe_dict(),
                "usage": None,
                "duration_ms": _elapsed_ms(started),
                "error": "base_url and model are required",
            }

        payload = {
            "model": self.config.model,
            "messages": _build_messages(question, hits, grounding=grounding),
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
                    "messages": _build_messages(question, hits, grounding=grounding, retry=True),
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
                    "grounding": grounding.as_dict(),
                    "llm": self.config.safe_dict(),
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                    "duration_ms": _elapsed_ms(started),
                    "retry_count": retry_count,
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
                "grounding": grounding.as_dict(),
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
                "citations": _citations(hits),
                "grounding": grounding.as_dict(),
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
    hits: list[RetrievalHit],
    *,
    grounding: AnswerabilityResult | None = None,
    retry: bool = False,
) -> list[dict[str, str]]:
    evidence = "\n\n".join(_format_evidence(index, hit) for index, hit in enumerate(hits[:8], 1))
    grounding = grounding or AnswerabilityGate().evaluate(question, hits)
    retry_note = (
        "The previous answer was too conservative. The evidence is usable for this question "
        "type, so produce the allowed grounded or partial answer instead of replying only "
        "that evidence is insufficient. "
        if retry
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a grounded RAG answer layer. Answer only from the provided evidence. "
                "Treat the evidence as the user's personal knowledge base. When the user asks in "
                "Chinese, prefer starting the answer with '根据你的个人知识库，'. "
                "First decide whether the evidence directly answers, partially answers, or cannot "
                "answer the question. "
                "Use the same language as the user when possible. Cite claims with bracketed "
                "evidence numbers like [1] or [2]. Every factual claim must be supported by the "
                "provided evidence. "
                "For why/reason/causal questions, answer only when the evidence explicitly "
                "states a cause, reason, motive, or causal chain. If the evidence only repeats "
                "a related fact or conclusion without explaining why, say the evidence is "
                "insufficient. "
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
                f"Evidence:\n{evidence}\n\n"
                "If answerable, write a concise answer. If the answer mode is partial, write the "
                "partial answer and explicitly frame it as based on current materials. Include "
                "citations next to relevant claims. If not answerable from evidence, say what "
                "specific evidence is missing."
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
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def refusal_answer_for(grounding: AnswerabilityResult) -> str:
    if grounding.reason == "insufficient_causal_evidence":
        return INSUFFICIENT_CAUSAL_EVIDENCE
    if grounding.reason == "insufficient_exact_evidence":
        return INSUFFICIENT_EXACT_EVIDENCE
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
            if query_terms and not any(term in normalized_sentence for term in query_terms):
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
