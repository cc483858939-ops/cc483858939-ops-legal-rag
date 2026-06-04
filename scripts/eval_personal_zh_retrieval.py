from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from legal_rag.config import load_settings
from legal_rag.evidence import (
    retrieve_with_query_extensions_and_summary,
    select_rerank_query,
)
from legal_rag.factory import build_retriever, build_rewriter, build_store
from legal_rag.ingest import ingest_manifest
from legal_rag.query_rewrite import build_query_extensions
from legal_rag.relevance import EvidenceRelevanceGate
from legal_rag.schema import RetrievalHit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chinese personal KB retrieval.")
    parser.add_argument("--eval-set", default="data/eval/personal_zh_bm25_20.yml")
    parser.add_argument("--manifest", default="configs/personal_test_sources.yml")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--api-url", default=None, help="Evaluate a running API endpoint")
    parser.add_argument("--endpoint", choices=["evidence", "answer"], default="evidence")
    parser.add_argument("--llm-base-url", default="http://ollama:11434/v1")
    parser.add_argument("--llm-model", default="qwen3.5:9b")
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=700)
    parser.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cases = data.get("queries", data)
    if not isinstance(cases, list):
        raise ValueError("Eval set must be a list or contain a 'queries' list")
    return cases


def text_contains_all_terms(text: str, terms: list[str]) -> bool:
    normalized = text.casefold()
    return all(term.casefold() in normalized for term in terms)


def hit_text(hit: RetrievalHit | dict[str, Any]) -> str:
    return hit.text if isinstance(hit, RetrievalHit) else str(hit.get("text", ""))


def hit_bm25_score(hit: RetrievalHit | dict[str, Any]) -> float:
    if isinstance(hit, RetrievalHit):
        return float(hit.bm25_score)
    scores = hit.get("scores")
    if isinstance(scores, dict):
        return float(scores.get("bm25", 0.0) or 0.0)
    return 0.0


def hit_final_score(hit: RetrievalHit | dict[str, Any]) -> float:
    if isinstance(hit, RetrievalHit):
        return float(hit.final_score)
    return float(hit.get("final_score", 0.0) or 0.0)


def hit_citation(hit: RetrievalHit | dict[str, Any]) -> str | None:
    return hit.citation if isinstance(hit, RetrievalHit) else hit.get("citation")


def first_relevant_rank(
    hits: list[RetrievalHit] | list[dict[str, Any]],
    terms: list[str],
) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if text_contains_all_terms(hit_text(hit), terms):
            return rank
    return None


def dcg(relevance: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevance))


def build_summary(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
    offline: bool,
    api_url: str | None,
    endpoint: str = "local",
) -> dict[str, Any]:
    positive_top1 = [
        1.0 if row["top_bm25_score"] > 0 else 0.0
        for row in rows
    ]
    ranks = [row["first_relevant_rank"] for row in rows if row["first_relevant_rank"] is not None]

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

    summary = {
        "case_count": len(rows),
        "mode": mode,
        "top_k": top_k,
        "offline": offline,
        "api_url": api_url,
        "endpoint": endpoint,
        "hit_at_1": avg("hit_at_1"),
        "hit_at_3": avg("hit_at_3"),
        f"hit_at_{top_k}": avg(f"hit_at_{top_k}"),
        f"mrr_at_{top_k}": avg(f"mrr_at_{top_k}"),
        f"ndcg_at_{top_k}": avg(f"ndcg_at_{top_k}"),
        "top1_bm25_positive_rate": (
            sum(positive_top1) / len(positive_top1) if positive_top1 else 0.0
        ),
        "mean_top1_bm25_score": (
            sum(float(row["top_bm25_score"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "mean_first_relevant_rank": (
            sum(float(rank) for rank in ranks) / len(ranks) if ranks else None
        ),
        "evidence_relevant_rate": (
            sum(1.0 if row["evidence_relevant"] else 0.0 for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "answer_success_rate": avg("answer_success"),
        "grounded_answer_rate": avg("grounded_answer"),
        "answer_all_terms_rate": avg("answer_all_terms"),
        "answer_any_term_rate": avg("answer_any_term"),
        "citation_present_rate": avg("citation_present"),
        "llm_error_rate": avg("llm_error"),
        "model_fallback_rate": avg("model_fallback"),
        "partial_answer_rate": avg("partial_answer"),
        "forbidden_claim_rate": avg("forbidden_claim_hit"),
        "missing_ids": [row["id"] for row in rows if row["first_relevant_rank"] is None],
        "evidence_blocked_ids": [
            row["id"] for row in rows if row["evidence_relevant"] is not True
        ],
        "answer_failed_ids": [row["id"] for row in rows if row["answer_success"] != 1.0],
        "model_fallback_ids": [row["id"] for row in rows if row["model_fallback"] == 1.0],
        "forbidden_claim_failure_ids": [
            row["id"] for row in rows if row["forbidden_claim_hit"] == 1.0
        ],
        "retrieval_status_counts": count_values(rows, "retrieval_status"),
        "answer_status_counts": count_values(rows, "answer_status"),
        "rows": rows,
    }
    return json.loads(json.dumps(summary, ensure_ascii=False))


def row_from_hits(
    case: dict[str, Any],
    *,
    terms: list[str],
    hits: list[RetrievalHit] | list[dict[str, Any]],
    top_k: int,
    evidence_relevant: bool,
    evidence_reason: str | None,
    accepted_queries: list[str],
    reranker_skipped: bool,
    answer: dict[str, Any] | None = None,
    retrieval_status: str | None = None,
    answer_source: str | None = None,
    intent: str | None = None,
    candidate_hit_count: int | None = None,
    relevant_hit_count: int | None = None,
) -> dict[str, Any]:
    rank = first_relevant_rank(hits, terms)
    binary_relevance = [
        1 if text_contains_all_terms(hit_text(hit), terms) else 0
        for hit in hits
    ]
    ideal = sorted(binary_relevance, reverse=True)
    top_hit = hits[0] if hits else None
    answer = answer or {}
    answer_text = str(answer.get("answer") or "")
    answer_status = str(answer.get("answer_status") or "")
    answer_citations = answer.get("citations") if isinstance(answer.get("citations"), list) else []
    forbidden_patterns = list(case.get("forbidden_claim_patterns") or case.get("forbidden_terms") or [])
    forbidden_claim_hit = any(pattern_matches(answer_text, pattern) for pattern in forbidden_patterns)
    model_fallback = answer_status == "model_fallback" or retrieval_status == "model_fallback"
    answer_success = answer_status in {"answered", "partial", "model_fallback"}
    grounded_answer = bool(
        retrieval_status == "retrieved"
        and answer_status in {"answered", "partial"}
        and answer_citations
    )
    answer_term_matches = [
        term.casefold() in answer_text.casefold()
        for term in terms
    ]
    return {
        "id": case["id"],
        "query": case["query"],
        "category": case.get("category", "general"),
        "expected_terms": terms,
        "first_relevant_rank": rank,
        "hit_at_1": 1.0 if rank == 1 else 0.0,
        "hit_at_3": 1.0 if rank is not None and rank <= 3 else 0.0,
        f"hit_at_{top_k}": 1.0 if rank is not None and rank <= top_k else 0.0,
        f"mrr_at_{top_k}": 1.0 / rank if rank is not None and rank <= top_k else 0.0,
        f"ndcg_at_{top_k}": (
            dcg(binary_relevance) / dcg(ideal) if dcg(ideal) else 0.0
        ),
        "evidence_relevant": evidence_relevant,
        "evidence_reason": evidence_reason,
        "accepted_queries": accepted_queries,
        "accepted_query_count": len(accepted_queries),
        "reranker_skipped": reranker_skipped,
        "top_score": hit_final_score(top_hit) if top_hit is not None else 0.0,
        "top_bm25_score": hit_bm25_score(top_hit) if top_hit is not None else 0.0,
        "top_citation": hit_citation(top_hit) if top_hit is not None else None,
        "top_text_preview": hit_text(top_hit)[:160] if top_hit is not None else "",
        "candidate_hit_count": candidate_hit_count if candidate_hit_count is not None else len(hits),
        "relevant_hit_count": relevant_hit_count if relevant_hit_count is not None else len(hits),
        "retrieval_status": retrieval_status,
        "answer_source": answer_source,
        "intent": intent,
        "answer_status": answer_status,
        "answer_success": 1.0 if answer_success else 0.0,
        "grounded_answer": 1.0 if grounded_answer else 0.0,
        "answer_all_terms": 1.0 if terms and all(answer_term_matches) else 0.0,
        "answer_any_term": 1.0 if any(answer_term_matches) else 0.0,
        "citation_present": 1.0 if answer_citations else 0.0,
        "llm_error": 1.0 if answer_status == "error" or answer.get("error") else 0.0,
        "model_fallback": 1.0 if model_fallback else 0.0,
        "partial_answer": 1.0 if answer_status == "partial" else 0.0,
        "used_citation_count": float(len(answer_citations)),
        "forbidden_claim_patterns": forbidden_patterns,
        "forbidden_claim_hit": 1.0 if forbidden_claim_hit else 0.0,
        "answer_preview": answer_text[:240],
    }


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def pattern_matches(text: str, pattern: str) -> bool:
    if not pattern:
        return False
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return pattern.casefold() in text.casefold()


def run_eval(
    cases: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
    offline: bool,
    manifest: str,
) -> dict[str, Any]:
    settings = load_settings()
    store = build_store(settings, offline=offline)
    ingest_manifest(manifest, store)
    retriever = build_retriever(settings, store)
    rewriter = build_rewriter(settings)
    relevance_gate = EvidenceRelevanceGate()

    rows: list[dict[str, Any]] = []
    for case in cases:
        terms = list(case["expected_terms"])
        rewrite = rewriter.rewrite(case["query"])
        extensions = build_query_extensions(
            rewrite,
            embedder=getattr(store, "embedder", None),
            min_similarity=settings.query_extension_min_similarity,
        )
        hits, reranker_summary = retrieve_with_query_extensions_and_summary(
            retriever,
            extensions.accepted,
            top_k=top_k,
            mode=mode,
            rerank_query=select_rerank_query(rewrite, extensions.accepted),
        )
        evidence_relevance = relevance_gate.evaluate(case["query"], hits)
        rows.append(
            row_from_hits(
                case,
                terms=terms,
                hits=hits,
                top_k=top_k,
                evidence_relevant=evidence_relevance.relevant,
                evidence_reason=evidence_relevance.reason,
                accepted_queries=[query.query for query in extensions.accepted],
                reranker_skipped=bool(reranker_summary.get("skipped")),
                candidate_hit_count=len(hits),
                relevant_hit_count=evidence_relevance.relevant_hit_count,
            )
        )

    return build_summary(rows, mode=mode, top_k=top_k, offline=offline, api_url=None)


def request_api(
    api_url: str,
    *,
    endpoint: str,
    query: str,
    mode: str,
    top_k: int,
    llm_base_url: str,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "top_k": top_k,
        "mode": mode,
        "filters": {},
        "context": [],
    }
    if endpoint == "answer":
        payload["llm"] = {
            "provider": "openai_compatible",
            "base_url": llm_base_url,
            "api_key": None,
            "model": llm_model,
            "temperature": llm_temperature,
            "max_tokens": llm_max_tokens,
            "timeout_seconds": llm_timeout_seconds,
        }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url.rstrip("/") + f"/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=llm_timeout_seconds + 60) as response:
        return json.loads(response.read().decode("utf-8"))


def run_api_eval(
    cases: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
    api_url: str,
    endpoint: str,
    llm_base_url: str,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_timeout_seconds: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        response = request_api(
            api_url,
            endpoint=endpoint,
            query=case["query"],
            mode=mode,
            top_k=top_k,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_timeout_seconds=llm_timeout_seconds,
        )
        relevance = response.get("relevance") or {}
        extensions = response.get("query_extensions") or {}
        reranker = response.get("reranker") or {}
        answer = response.get("answer") if endpoint == "answer" else None
        rows.append(
            row_from_hits(
                case,
                terms=list(case["expected_terms"]),
                hits=response.get("hits") or [],
                top_k=top_k,
                evidence_relevant=bool(relevance.get("relevant")),
                evidence_reason=relevance.get("reason"),
                accepted_queries=[
                    item.get("query", "")
                    for item in extensions.get("accepted", [])
                    if isinstance(item, dict)
                ],
                reranker_skipped=bool(reranker.get("skipped")),
                answer=answer,
                retrieval_status=response.get("retrieval_status"),
                answer_source=response.get("answer_source"),
                intent=response.get("intent"),
                candidate_hit_count=response.get("candidate_hit_count"),
                relevant_hit_count=response.get("relevant_hit_count"),
            )
        )

    return build_summary(
        rows,
        mode=mode,
        top_k=top_k,
        offline=False,
        api_url=api_url,
        endpoint=endpoint,
    )


def main() -> None:
    args = parse_args()
    cases = load_cases(args.eval_set)
    if args.api_url:
        result = run_api_eval(
            cases,
            mode=args.mode,
            top_k=args.top_k,
            api_url=args.api_url,
            endpoint=args.endpoint,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            llm_timeout_seconds=args.llm_timeout_seconds,
        )
    else:
        result = run_eval(
            cases,
            mode=args.mode,
            top_k=args.top_k,
            offline=args.offline,
            manifest=args.manifest,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
