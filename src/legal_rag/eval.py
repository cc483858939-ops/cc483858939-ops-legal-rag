from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from legal_rag.answer import ExtractiveAnswerer
from legal_rag.evidence import (
    retrieve_with_query_extensions_and_summary,
    select_rerank_query,
)
from legal_rag.metadata_filter import normalize_metadata_filters
from legal_rag.query_rewrite import QueryRewriter, build_query_extensions
from legal_rag.retriever import HybridRetriever
from legal_rag.schema import EvalCase, RetrievalEvalCase, RetrievalHit


def load_eval_set(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate_json(line))
    return cases


def load_retrieval_eval_set(path: str | Path) -> list[RetrievalEvalCase]:
    eval_path = Path(path)
    if eval_path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(eval_path.read_text(encoding="utf-8")) or {}
        raw_cases = data.get("queries", data) if isinstance(data, dict) else data
        if not isinstance(raw_cases, list):
            raise ValueError("Retrieval eval YAML must be a list or contain a 'queries' list")
        return [RetrievalEvalCase.model_validate(item) for item in raw_cases]

    cases: list[RetrievalEvalCase] = []
    with eval_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "query" not in payload and "question" in payload:
                payload["query"] = payload.pop("question")
            cases.append(RetrievalEvalCase.model_validate(payload))
    return cases


@dataclass
class EvalRunner:
    retriever: HybridRetriever
    answerer: ExtractiveAnswerer

    def run(self, cases: list[EvalCase], *, top_k: int = 10) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for case in cases:
            hits = self.retriever.retrieve(case.question, top_k=top_k)
            answer = self.answerer.answer(case.question, hits)
            citations = [hit.citation for hit in hits]
            rows.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "expected_citations": case.expected_citations,
                    "retrieved_citations": citations,
                    "answer": answer.answer,
                    "refused": answer.refused,
                    "recall_at_5": citation_recall(case.expected_citations, citations[:5]),
                    "recall_at_10": citation_recall(case.expected_citations, citations[:10]),
                    "mrr_at_10": mrr(case.expected_citations, citations[:10]),
                    "ndcg_at_10": ndcg(case.expected_citations, citations[:10]),
                    "citation_recall_at_5": citation_recall(
                        case.expected_citations,
                        answer.citations[:5],
                    ),
                    "answer_correct": answer_matches(answer.answer, case.expected_answer_patterns),
                    "refusal_correct": (
                        (answer.refused is True)
                        if case.must_refuse
                        else (answer.refused is False)
                    ),
                }
            )
        return summarize(rows)


@dataclass
class RetrievalEvalRunner:
    retriever: HybridRetriever
    rewriter: QueryRewriter
    embedder: Any | None = None
    min_similarity: float = 0.45
    inferred_metadata_filters_enabled: bool = True
    query_rewrite_filter_min_confidence: float = 0.55

    def run(
        self,
        cases: list[RetrievalEvalCase],
        *,
        top_k: int = 8,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for case in cases:
            rewrite = self.rewriter.rewrite(case.query)
            extensions = build_query_extensions(
                rewrite,
                embedder=self.embedder,
                min_similarity=self.min_similarity,
            )
            metadata_filters = normalize_metadata_filters(
                explicit_filters=case.filters,
                rewrite_filters=rewrite.filters,
                rewrite_confidence=rewrite.confidence,
                inferred_enabled=self.inferred_metadata_filters_enabled,
                min_confidence=self.query_rewrite_filter_min_confidence,
            )
            hits, reranker_summary = retrieve_with_query_extensions_and_summary(
                self.retriever,
                extensions.accepted,
                top_k=top_k,
                mode=mode,
                rerank_query=select_rerank_query(rewrite, extensions.accepted),
                metadata_filters=metadata_filters.effective,
            )
            citations = [hit.citation for hit in hits]
            expected_ranks = expected_citation_ranks(case.expected_citations, citations)
            row = {
                "id": case.id,
                "query": case.query,
                "category": case.category,
                "tags": case.tags,
                "expected_citations": case.expected_citations,
                "retrieved_citations": citations,
                "expected_ranks": expected_ranks,
                "hit_count": len(hits),
                "top_citation": citations[0] if citations else None,
                "top_score": hits[0].final_score if hits else 0.0,
                "expected_missing": case.expected_missing,
                "metadata_filters": metadata_filters.as_dict(),
                "rewrite": {
                    "backend": rewrite.backend,
                    "model": rewrite.model,
                    "applied": rewrite.applied,
                    "canonical_query": rewrite.canonical_query,
                    "error": rewrite.error,
                },
                "query_extensions": {
                    "accepted_count": len(extensions.accepted),
                    "generated_accepted_count": max(0, len(extensions.accepted) - 1),
                    "rejected_count": len(extensions.rejected),
                    "accepted": [item.as_dict() for item in extensions.accepted],
                    "rejected": [item.as_dict() for item in extensions.rejected],
                    "error": extensions.error,
                },
                "reranker": reranker_summary,
            }
            if case.expected_citations:
                hit_at_1 = (
                    1.0
                    if citations[:1] and citations[0] in case.expected_citations
                    else 0.0
                )
                row.update(
                    {
                        "hit_at_1": hit_at_1,
                        "recall_at_3": citation_recall(case.expected_citations, citations[:3]),
                        "recall_at_5": citation_recall(case.expected_citations, citations[:5]),
                        "recall_at_k": citation_recall(case.expected_citations, citations[:top_k]),
                        "mrr_at_k": mrr(case.expected_citations, citations[:top_k]),
                        "ndcg_at_k": ndcg(case.expected_citations, citations[:top_k]),
                    }
                )
            rows.append(row)
        return summarize_retrieval_eval(
            rows,
            top_k=top_k,
            mode=mode,
            min_similarity=self.min_similarity,
        )


def citation_recall(expected: list[str], retrieved: list[str]) -> float:
    if not expected:
        return 1.0
    expected_set = set(expected)
    retrieved_set = set(retrieved)
    return len(expected_set & retrieved_set) / len(expected_set)


def mrr(expected: list[str], retrieved: list[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    for index, citation in enumerate(retrieved, start=1):
        if citation in expected_set:
            return 1 / index
    return 0.0


def ndcg(expected: list[str], retrieved: list[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    dcg = 0.0
    for index, citation in enumerate(retrieved, start=1):
        if citation in expected_set:
            dcg += 1 / math.log2(index + 1)
    ideal = sum(
        1 / math.log2(index + 1)
        for index in range(1, min(len(expected), len(retrieved)) + 1)
    )
    return dcg / ideal if ideal else 0.0


def expected_citation_ranks(expected: list[str], retrieved: list[str]) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for citation in expected:
        ranks[citation] = retrieved.index(citation) + 1 if citation in retrieved else None
    return ranks


def answer_matches(answer: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return all(re.search(pattern, answer, re.IGNORECASE) for pattern in patterns)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

    summary = {
        "case_count": len(rows),
        "recall_at_5": avg("recall_at_5"),
        "recall_at_10": avg("recall_at_10"),
        "mrr_at_10": avg("mrr_at_10"),
        "ndcg_at_10": avg("ndcg_at_10"),
        "citation_recall_at_5": avg("citation_recall_at_5"),
        "answer_accuracy": avg("answer_correct"),
        "refusal_precision": avg("refusal_correct"),
        "rows": rows,
    }
    return json.loads(json.dumps(summary, ensure_ascii=False))


def summarize_retrieval_eval(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    mode: str,
    min_similarity: float,
) -> dict[str, Any]:
    positive_rows = [row for row in rows if row["expected_citations"]]
    missing_rows = [row for row in rows if row["expected_missing"]]

    def avg_positive(key: str) -> float:
        if not positive_rows:
            return 0.0
        return sum(float(row[key]) for row in positive_rows) / len(positive_rows)

    def avg_all(key: str) -> float:
        if not rows:
            return 0.0
        return sum(float(row["query_extensions"][key]) for row in rows) / len(rows)

    generated_accepted = sum(
        int(row["query_extensions"]["generated_accepted_count"]) for row in rows
    )
    rejected = sum(int(row["query_extensions"]["rejected_count"]) for row in rows)
    generated_candidates = generated_accepted + rejected
    summary = {
        "case_count": len(rows),
        "positive_case_count": len(positive_rows),
        "expected_missing_count": len(missing_rows),
        "mode": mode,
        "top_k": top_k,
        "query_extension_min_similarity": min_similarity,
        "hit_at_1": avg_positive("hit_at_1"),
        "recall_at_3": avg_positive("recall_at_3"),
        "recall_at_5": avg_positive("recall_at_5"),
        "recall_at_k": avg_positive("recall_at_k"),
        "mrr_at_k": avg_positive("mrr_at_k"),
        "ndcg_at_k": avg_positive("ndcg_at_k"),
        "perfect_recall_at_k": (
            sum(1 for row in positive_rows if float(row["recall_at_k"]) >= 1.0)
            / len(positive_rows)
            if positive_rows
            else 0.0
        ),
        "query_extension": {
            "avg_generated_accepted": avg_all("generated_accepted_count"),
            "avg_rejected": avg_all("rejected_count"),
            "generated_acceptance_rate": (
                generated_accepted / generated_candidates if generated_candidates else 0.0
            ),
        },
        "rows": rows,
    }
    return json.loads(json.dumps(summary, ensure_ascii=False))


def hits_to_jsonable(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    return [hit.model_dump(mode="json") for hit in hits]
