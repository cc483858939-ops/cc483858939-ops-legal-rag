from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from legal_rag.schema import DocumentChunk
from legal_rag.text import CJK_RE, tokenize


@dataclass
class BM25Index:
    """Small in-memory BM25 with a jieba-backed path for Chinese queries."""

    k1: float = 1.5
    b: float = 0.75
    chunks: list[DocumentChunk] = field(default_factory=list)
    term_freqs: list[Counter[str]] = field(default_factory=list)
    doc_freqs: Counter[str] = field(default_factory=Counter)
    doc_lengths: list[int] = field(default_factory=list)
    avg_doc_length: float = 0.0
    jieba_index: Any | None = field(default=None, init=False, repr=False)
    jieba_available: bool = field(default=False, init=False)

    def fit(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self._fit_python_bm25(self.chunks)
        self._fit_jieba_bm25(self.chunks)

    def search(self, query: str, *, top_k: int) -> list[tuple[DocumentChunk, float]]:
        if _contains_cjk(query):
            jieba_results = self._search_jieba(query, top_k=top_k)
            if jieba_results:
                return jieba_results
        return self._search_python(query, top_k=top_k)

    def _fit_python_bm25(self, chunks: list[DocumentChunk]) -> None:
        self.term_freqs = []
        self.doc_freqs = Counter()
        self.doc_lengths = []

        for chunk in chunks:
            terms = tokenize(chunk.text)
            freqs = Counter(terms)
            self.term_freqs.append(freqs)
            self.doc_lengths.append(len(terms))
            for term in freqs:
                self.doc_freqs[term] += 1

        total_length = sum(self.doc_lengths)
        self.avg_doc_length = total_length / len(self.doc_lengths) if self.doc_lengths else 0.0

    def _fit_jieba_bm25(self, chunks: list[DocumentChunk]) -> None:
        self.jieba_index = None
        self.jieba_available = False
        if not chunks:
            return
        try:
            from bm25_jieba import BM25
        except ImportError:
            return

        index = BM25(k1=self.k1, b=self.b, lowercase=True)
        index.fit([chunk.text for chunk in chunks], ids=list(range(len(chunks))))
        self.jieba_index = index
        self.jieba_available = True

    def _search_jieba(self, query: str, *, top_k: int) -> list[tuple[DocumentChunk, float]]:
        if self.jieba_index is None or not self.chunks:
            return []
        results = self.jieba_index.search(query, top_k=top_k)
        scored: list[tuple[DocumentChunk, float]] = []
        for doc_id, score in results:
            try:
                index = int(doc_id)
                value = float(score)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(self.chunks) and value > 0:
                scored.append((self.chunks[index], value))
        return scored[:top_k]

    def _search_python(self, query: str, *, top_k: int) -> list[tuple[DocumentChunk, float]]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scored: list[tuple[DocumentChunk, float]] = []
        for index, chunk in enumerate(self.chunks):
            score = self._score(query_terms, index)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        total_docs = len(self.chunks)
        doc_len = self.doc_lengths[doc_index]
        freqs = self.term_freqs[doc_index]
        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1))
            score += idf * (tf * (self.k1 + 1) / denom)
        return score


def _contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))
