from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from legal_rag.embeddings import DenseEmbedder, cosine_similarity

SYSTEM_PROMPT = """You rewrite personal knowledge-base retrieval queries. Do not answer the question.
Return one compact JSON object only. No markdown.
The JSON schema is:
{
  "canonical_query": "short normalized retrieval query or null",
  "search_queries": ["2-4 concise retrieval queries"],
  "aliases": ["entity aliases, translations, abbreviations"],
  "filters": {},
  "confidence": 0.0
}
Rules:
- Preserve the user's intent.
- Expand Chinese aliases into likely entity names, translations, abbreviations, and domain terms.
- Prefer concrete entity names, document titles, proper nouns, and user-provided keywords over generic terms.
- If uncertain, keep expansions broad and set confidence below 0.5.
- Do not invent that a document exists in the corpus.
- Do not provide an answer, analysis, or citations as facts.
Examples:
- 辛普森案 -> O.J. Simpson murder trial; People v. Simpson; Orenthal James Simpson.
- 版权合理使用 -> fair use; 17 U.S.C. § 107; Campbell v. Acuff-Rose.
- 退款多久到账 -> refund processing time; refund timeline.
- 如何评价炫神 -> 炫神; Xuan Shen; evaluation.
"""


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    retrieval_query: str
    backend: str = "none"
    model: str | None = None
    applied: bool = False
    canonical_query: str | None = None
    search_queries: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    error: str | None = None

    @classmethod
    def original(
        cls,
        query: str,
        *,
        backend: str = "none",
        error: str | None = None,
    ) -> QueryRewriteResult:
        return cls(
            original_query=query,
            retrieval_query=query,
            backend=backend,
            applied=False,
            error=error,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "retrieval_query": self.retrieval_query,
            "backend": self.backend,
            "model": self.model,
            "applied": self.applied,
            "canonical_query": self.canonical_query,
            "search_queries": self.search_queries,
            "aliases": self.aliases,
            "filters": self.filters,
            "confidence": self.confidence,
            "error": self.error,
        }


@dataclass(frozen=True)
class AcceptedQueryExtension:
    query: str
    source: str
    similarity: float
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "source": self.source,
            "similarity": self.similarity,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class RejectedQueryExtension:
    query: str
    source: str
    similarity: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "source": self.source,
            "similarity": self.similarity,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QueryExtensionResult:
    min_similarity: float
    accepted: list[AcceptedQueryExtension]
    rejected: list[RejectedQueryExtension] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_similarity": self.min_similarity,
            "accepted": [item.as_dict() for item in self.accepted],
            "rejected": [item.as_dict() for item in self.rejected],
            "error": self.error,
        }

    @property
    def retrieval_query(self) -> str:
        return " | ".join(item.query for item in self.accepted)


class QueryRewriter:
    def rewrite(self, query: str) -> QueryRewriteResult:
        raise NotImplementedError


class NoopQueryRewriter(QueryRewriter):
    def rewrite(self, query: str) -> QueryRewriteResult:
        return QueryRewriteResult.original(query)


class OllamaQueryRewriter(QueryRewriter):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 8.0,
        max_queries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_queries = max_queries

    def rewrite(self, query: str) -> QueryRewriteResult:
        query = query.strip()
        if not query:
            return QueryRewriteResult.original(query, backend="ollama")

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 384,
                    },
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps({"query": query}, ensure_ascii=False),
                        },
                    ],
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("message", {}).get("content", "")
            parsed = parse_json_object(content)
            return result_from_model_payload(
                query,
                parsed,
                backend="ollama",
                model=self.model,
                max_queries=self.max_queries,
            )
        except Exception as exc:  # noqa: BLE001 - fail open so retrieval remains available.
            return QueryRewriteResult.original(
                query,
                backend="ollama",
                error=f"{type(exc).__name__}: {str(exc)[:220]}",
            )


def build_query_rewriter(settings: Any) -> QueryRewriter:
    backend = settings.query_rewrite_backend.strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return NoopQueryRewriter()
    if backend == "ollama":
        return OllamaQueryRewriter(
            base_url=settings.query_rewrite_base_url,
            model=settings.query_rewrite_model,
            timeout_seconds=settings.query_rewrite_timeout_seconds,
            max_queries=settings.query_rewrite_max_queries,
        )
    raise ValueError(f"Unsupported QUERY_REWRITE_BACKEND={settings.query_rewrite_backend!r}")


def result_from_model_payload(
    original_query: str,
    payload: dict[str, Any],
    *,
    backend: str,
    model: str | None,
    max_queries: int,
) -> QueryRewriteResult:
    canonical_query = clean_optional_text(payload.get("canonical_query"))
    search_queries = clean_text_list(payload.get("search_queries"), limit=max_queries)
    aliases = clean_text_list(payload.get("aliases"), limit=8)
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    confidence = parse_confidence(payload.get("confidence"))
    retrieval_query = build_retrieval_query(
        original_query,
        canonical_query=canonical_query,
        search_queries=search_queries,
        aliases=aliases,
    )
    return QueryRewriteResult(
        original_query=original_query,
        retrieval_query=retrieval_query,
        backend=backend,
        model=model,
        applied=retrieval_query != original_query,
        canonical_query=canonical_query,
        search_queries=search_queries,
        aliases=aliases,
        filters=filters,
        confidence=confidence,
    )


def build_query_extensions(
    rewrite: QueryRewriteResult,
    *,
    embedder: DenseEmbedder | None,
    min_similarity: float = 0.45,
) -> QueryExtensionResult:
    original_query = normalize_query_part(rewrite.original_query)
    accepted = [
        AcceptedQueryExtension(
            query=original_query,
            source="original",
            similarity=1.0,
            weight=1.0,
        )
    ]
    candidates = collect_extension_candidates(rewrite)
    if not candidates:
        return QueryExtensionResult(min_similarity=min_similarity, accepted=accepted)

    if embedder is None:
        rejected = [
            RejectedQueryExtension(
                query=query,
                source=source,
                similarity=None,
                reason="embedding_unavailable",
            )
            for query, source in candidates
        ]
        return QueryExtensionResult(
            min_similarity=min_similarity,
            accepted=accepted,
            rejected=rejected,
            error="Dense embedder is unavailable; using original query only.",
        )

    try:
        original_vector = embedder.embed_query(original_query)
        canonical_vector = (
            embedder.embed_query(rewrite.canonical_query) if rewrite.canonical_query else None
        )
        rejected: list[RejectedQueryExtension] = []
        for query, source in candidates:
            candidate_vector = embedder.embed_query(query)
            original_similarity = cosine_similarity(candidate_vector, original_vector)
            canonical_similarity = (
                cosine_similarity(candidate_vector, canonical_vector)
                if canonical_vector is not None
                else 0.0
            )
            similarity = max(original_similarity, canonical_similarity)
            if similarity >= min_similarity:
                accepted.append(
                    AcceptedQueryExtension(
                        query=query,
                        source=source,
                        similarity=similarity,
                        weight=similarity,
                    )
                )
            else:
                rejected.append(
                    RejectedQueryExtension(
                        query=query,
                        source=source,
                        similarity=similarity,
                        reason="below_similarity_threshold",
                    )
                )
        return QueryExtensionResult(
            min_similarity=min_similarity,
            accepted=accepted,
            rejected=rejected,
        )
    except Exception as exc:  # noqa: BLE001 - query extension must fail open.
        rejected = [
            RejectedQueryExtension(
                query=query,
                source=source,
                similarity=None,
                reason="extension_filter_error",
            )
            for query, source in candidates
        ]
        return QueryExtensionResult(
            min_similarity=min_similarity,
            accepted=accepted,
            rejected=rejected,
            error=f"{type(exc).__name__}: {str(exc)[:220]}",
        )


def collect_extension_candidates(rewrite: QueryRewriteResult) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if rewrite.canonical_query:
        candidates.append((rewrite.canonical_query, "canonical"))
    candidates.extend((query, "search_query") for query in rewrite.search_queries)
    candidates.extend((query, "alias") for query in rewrite.aliases)

    output: list[tuple[str, str]] = []
    seen = {normalize_query_part(rewrite.original_query).casefold()}
    for query, source in candidates:
        clean = normalize_query_part(query)
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        output.append((clean, source))
    return output


def build_retrieval_query(
    original_query: str,
    *,
    canonical_query: str | None,
    search_queries: list[str],
    aliases: list[str],
) -> str:
    parts = [original_query, canonical_query, *search_queries, *aliases]
    unique_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = normalize_query_part(part)
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            unique_parts.append(clean)
    return " ".join(unique_parts) or original_query


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("query rewrite response must be a JSON object")
    return parsed


def clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    clean = normalize_query_part(value)
    return clean or None


def clean_text_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        clean = clean_optional_text(item)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
        if len(output) >= limit:
            break
    return output


def normalize_query_part(value: Any) -> str:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    if clean.casefold() in {"", "none", "null", "nil", "n/a", "na"}:
        return ""
    return clean


def parse_confidence(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))
