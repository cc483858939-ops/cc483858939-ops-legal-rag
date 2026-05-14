from __future__ import annotations

from pathlib import Path

from legal_rag.config import ROOT_DIR
from legal_rag.eval import (
    EvalRunner,
    RetrievalEvalRunner,
    citation_recall,
    load_eval_set,
    load_retrieval_eval_set,
    mrr,
    ndcg,
)
from legal_rag.query_rewrite import NoopQueryRewriter
from legal_rag.schema import RetrievalEvalCase


def test_answerer_returns_citations(retriever, answerer) -> None:
    hits = retriever.retrieve("What are the fair use factors?", top_k=5)
    answer = answerer.answer("What are the fair use factors?", hits)

    assert answer.refused is False
    assert "17 U.S.C. § 107" in answer.citations
    assert "legal information" in answer.answer


def test_answerer_refuses_without_support(answerer) -> None:
    answer = answerer.answer("What is the Delaware appraisal rule?", [])

    assert answer.refused is True
    assert answer.citations == []


def test_eval_set_has_sixty_cases(eval_path: Path) -> None:
    cases = load_eval_set(eval_path)

    assert len(cases) == 60
    assert sum(case.must_refuse for case in cases) == 5


def test_retrieval_eval_queries_load() -> None:
    cases = load_retrieval_eval_set(ROOT_DIR / "data" / "eval" / "eval_queries.yml")

    assert len(cases) == 14
    assert cases[0].id == "cn-apa-notice-rulemaking"
    assert cases[0].expected_citations == ["5 U.S.C. § 553"]


def test_eval_metrics() -> None:
    expected = ["A", "B"]
    retrieved = ["X", "B", "A"]

    assert citation_recall(expected, retrieved) == 1.0
    assert mrr(expected, retrieved) == 0.5
    assert 0 < ndcg(expected, retrieved) <= 1.0


def test_seed_eval_runs(retriever, answerer, eval_path: Path) -> None:
    cases = load_eval_set(eval_path)[:10]
    result = EvalRunner(retriever, answerer).run(cases, top_k=10)

    assert result["case_count"] == 10
    assert result["recall_at_10"] >= 0.8
    assert "rows" in result


def test_retrieval_eval_runner_scores_citations(retriever, store) -> None:
    cases = [
        RetrievalEvalCase(
            id="fair-use",
            query="What are the statutory fair use factors?",
            expected_citations=["17 U.S.C. § 107"],
        )
    ]

    result = RetrievalEvalRunner(
        retriever,
        NoopQueryRewriter(),
        embedder=store.embedder,
    ).run(cases, top_k=8)

    assert result["case_count"] == 1
    assert result["positive_case_count"] == 1
    assert result["recall_at_k"] == 1.0
    assert result["mrr_at_k"] > 0
    assert result["query_extension"]["avg_generated_accepted"] == 0.0


def test_seed_eval_acceptance_thresholds(retriever, answerer, eval_path: Path) -> None:
    result = EvalRunner(retriever, answerer).run(load_eval_set(eval_path), top_k=10)

    assert result["case_count"] == 60
    assert result["recall_at_10"] >= 0.90
    assert result["citation_recall_at_5"] >= 0.85
    assert result["refusal_precision"] == 1.0
