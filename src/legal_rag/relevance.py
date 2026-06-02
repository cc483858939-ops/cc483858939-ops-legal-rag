from __future__ import annotations

import re
from dataclasses import dataclass, field

from legal_rag.llm_answer import classify_question_type, extract_query_terms
from legal_rag.schema import RetrievalHit
from legal_rag.text import CJK_RE, tokenize

NO_RELEVANT_EVIDENCE_ANSWER = "知识库没有找到与这个问题直接相关的材料，不能可靠回答。"

MIN_DIRECT_FINAL_SCORE = 0.25
MIN_RERANK_SCORE = 0.35
MIN_STRONG_CJK_DEFINITION_DENSE_SCORE = 0.68
MIN_STRONG_CJK_DEFINITION_RERANK_SCORE = 0.65
WEAK_QUERY_TERMS = {
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


@dataclass(frozen=True)
class RelevanceDecision:
    relevant: bool
    reason: str | None
    matched_terms: list[str] = field(default_factory=list)
    top_score: float = 0.0
    top_dense_score: float = 0.0
    top_bm25_score: float = 0.0
    top_rerank_score: float | None = None
    candidate_hit_count: int = 0
    relevant_hit_count: int = 0

    @property
    def retrieval_status(self) -> str:
        return "retrieved" if self.relevant else "no_relevant_evidence"

    def as_dict(self) -> dict:
        return {
            "relevant": self.relevant,
            "reason": self.reason,
            "matched_terms": self.matched_terms,
            "top_score": self.top_score,
            "top_dense_score": self.top_dense_score,
            "top_bm25_score": self.top_bm25_score,
            "top_rerank_score": self.top_rerank_score,
            "candidate_hit_count": self.candidate_hit_count,
            "relevant_hit_count": self.relevant_hit_count,
        }


class EvidenceRelevanceGate:
    def evaluate(self, query: str, hits: list[RetrievalHit]) -> RelevanceDecision:
        if not hits:
            return RelevanceDecision(relevant=False, reason="no_hits")

        top = hits[0]
        matched_terms = matched_query_terms(query, hits[:5])
        top_rerank = top.rerank_score
        top_score = max(top.final_score, _max_source_hit_score(top))
        candidate_count = len(hits)
        relevant_count = _count_hits_with_terms(hits, matched_terms) if matched_terms else 0

        if matched_terms:
            return RelevanceDecision(
                relevant=True,
                reason="query_term_overlap",
                matched_terms=matched_terms,
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
                candidate_hit_count=candidate_count,
                relevant_hit_count=max(1, relevant_count),
            )

        is_cjk_definition = bool(CJK_RE.search(query)) and classify_question_type(query) == "definition"
        if (
            is_cjk_definition
            and top_rerank is not None
            and top_rerank >= MIN_STRONG_CJK_DEFINITION_RERANK_SCORE
        ):
            return RelevanceDecision(
                relevant=True,
                reason="strong_definition_rerank_score",
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
                candidate_hit_count=candidate_count,
                relevant_hit_count=1,
            )
        if is_cjk_definition and top.dense_score >= MIN_STRONG_CJK_DEFINITION_DENSE_SCORE:
            return RelevanceDecision(
                relevant=True,
                reason="strong_definition_dense_score",
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
                candidate_hit_count=candidate_count,
                relevant_hit_count=1,
            )

        if top_rerank is not None and top_rerank >= MIN_RERANK_SCORE:
            return RelevanceDecision(
                relevant=True,
                reason="rerank_score_threshold",
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
                candidate_hit_count=candidate_count,
                relevant_hit_count=1,
            )
        if (
            not is_cjk_definition
            and top_score >= MIN_DIRECT_FINAL_SCORE
            and not _looks_like_rrf_score(top_score)
        ):
            return RelevanceDecision(
                relevant=True,
                reason="direct_score_threshold",
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
                candidate_hit_count=candidate_count,
                relevant_hit_count=1,
            )
        return RelevanceDecision(
            relevant=False,
            reason="below_relevance_threshold",
            top_score=top_score,
            top_dense_score=top.dense_score,
            top_bm25_score=top.bm25_score,
            top_rerank_score=top_rerank,
            candidate_hit_count=candidate_count,
            relevant_hit_count=0,
        )


def matched_query_terms(query: str, hits: list[RetrievalHit]) -> list[str]:
    terms = _meaningful_terms(extract_query_terms(query))
    matched = _matched_terms_in_hits(terms, hits)
    if matched:
        return matched
    return _matched_terms_in_hits(_meaningful_terms(tokenize(query)), hits)


def _matched_terms_in_hits(terms: list[str], hits: list[RetrievalHit]) -> list[str]:
    if not terms:
        return []
    haystack = "\n".join(
        f"{hit.title}\n{hit.citation}\n{hit.section or ''}\n{hit.text}" for hit in hits
    ).casefold()
    matched: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        if key in haystack or _ascii_token_overlap(key, haystack):
            matched.append(term)
            seen.add(key)
    return matched


def meaningful_query_terms(query: str) -> list[str]:
    return [*_meaningful_terms(extract_query_terms(query)), *_meaningful_terms(tokenize(query))]


def _meaningful_terms(raw_terms: list[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = term.strip().casefold()
        if not _is_meaningful_query_term(normalized) or normalized in seen:
            continue
        terms.append(term.strip())
        seen.add(normalized)
    return terms


def filter_relevant_hits(
    query: str,
    hits: list[RetrievalHit],
    decision: RelevanceDecision,
) -> list[RetrievalHit]:
    if not decision.relevant or not hits:
        return []
    if decision.matched_terms:
        filtered = [hit for hit in hits if _hit_contains_any_term(hit, decision.matched_terms)]
        return filtered or hits[:1]
    return hits[: max(1, decision.relevant_hit_count)]


def _is_meaningful_query_term(term: str) -> bool:
    if len(term) < 2:
        return False
    if term in WEAK_QUERY_TERMS:
        return False
    if term.isdigit():
        return False
    return True


def _hit_contains_any_term(hit: RetrievalHit, terms: list[str]) -> bool:
    haystack = f"{hit.title}\n{hit.citation}\n{hit.section or ''}\n{hit.text}".casefold()
    return any(term.casefold() in haystack or _ascii_token_overlap(term.casefold(), haystack) for term in terms)


def _count_hits_with_terms(hits: list[RetrievalHit], terms: list[str]) -> int:
    return sum(1 for hit in hits if _hit_contains_any_term(hit, terms))


def _ascii_token_overlap(term: str, text: str) -> bool:
    if not re.fullmatch(r"[a-z0-9_.-]+", term):
        return False
    return bool(re.search(rf"(?<![a-z0-9_.-]){re.escape(term)}(?![a-z0-9_.-])", text))


def _looks_like_rrf_score(score: float) -> bool:
    return 0 < score < 0.2


def _max_source_hit_score(hit: RetrievalHit) -> float:
    contributions = hit.rank_explanation.get("query_contributions")
    if not isinstance(contributions, list):
        return 0.0
    scores: list[float] = []
    for item in contributions:
        if not isinstance(item, dict):
            continue
        try:
            scores.append(float(item.get("hit_final_score", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def no_relevant_answer(query: str, decision: RelevanceDecision) -> dict:
    return {
        "enabled": True,
        "skipped": True,
        "reason": "no_relevant_evidence",
        "refused": True,
        "answer_status": "refused",
        "answer": NO_RELEVANT_EVIDENCE_ANSWER,
        "citations": [],
        "grounding": {
            "question_type": "unknown",
            "answerable": False,
            "checked_hit_count": 0,
            "required_evidence": "与问题直接相关的检索证据",
            "answer_mode": "refuse",
            "matched_cues": decision.matched_terms,
            "missing_evidence": ["relevant_evidence"],
        },
        "llm": None,
        "usage": None,
        "duration_ms": 0.0,
        "error": None,
    }
