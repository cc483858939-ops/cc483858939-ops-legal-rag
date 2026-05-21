from __future__ import annotations

import re
from dataclasses import dataclass, field

from legal_rag.llm_answer import extract_query_terms
from legal_rag.schema import RetrievalHit

NO_RELEVANT_EVIDENCE_ANSWER = "知识库没有找到与这个问题直接相关的材料，不能可靠回答。"

MIN_DIRECT_FINAL_SCORE = 0.25
MIN_RERANK_SCORE = 0.35


@dataclass(frozen=True)
class RelevanceDecision:
    relevant: bool
    reason: str | None
    matched_terms: list[str] = field(default_factory=list)
    top_score: float = 0.0
    top_dense_score: float = 0.0
    top_bm25_score: float = 0.0
    top_rerank_score: float | None = None

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
        }


class EvidenceRelevanceGate:
    def evaluate(self, query: str, hits: list[RetrievalHit]) -> RelevanceDecision:
        if not hits:
            return RelevanceDecision(relevant=False, reason="no_hits")

        top = hits[0]
        matched_terms = matched_query_terms(query, hits[:5])
        top_rerank = top.rerank_score
        top_score = max(top.final_score, _max_source_hit_score(top))

        if matched_terms:
            return RelevanceDecision(
                relevant=True,
                reason="query_term_overlap",
                matched_terms=matched_terms,
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
            )
        if top.bm25_score > 0:
            return RelevanceDecision(
                relevant=True,
                reason="bm25_positive",
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
            )
        if top_rerank is not None and top_rerank >= MIN_RERANK_SCORE:
            return RelevanceDecision(
                relevant=True,
                reason="rerank_score_threshold",
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
            )
        if top_score >= MIN_DIRECT_FINAL_SCORE and not _looks_like_rrf_score(top_score):
            return RelevanceDecision(
                relevant=True,
                reason="direct_score_threshold",
                top_score=top_score,
                top_dense_score=top.dense_score,
                top_bm25_score=top.bm25_score,
                top_rerank_score=top_rerank,
            )
        return RelevanceDecision(
            relevant=False,
            reason="below_relevance_threshold",
            top_score=top_score,
            top_dense_score=top.dense_score,
            top_bm25_score=top.bm25_score,
            top_rerank_score=top_rerank,
        )


def matched_query_terms(query: str, hits: list[RetrievalHit]) -> list[str]:
    terms = [term for term in extract_query_terms(query) if len(term) >= 2]
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
