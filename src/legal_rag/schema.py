from __future__ import annotations

from datetime import date as Date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_id: str
    doc_type: str
    title: str
    citation: str
    jurisdiction: str = "US"
    court: str | None = None
    date: Date | None = None
    section: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_id: str
    doc_type: str
    title: str
    citation: str
    jurisdiction: str
    court: str | None = None
    date: Date | None = None
    section: str | None = None
    text: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float | None = None
    rank_explanation: dict[str, Any] = Field(default_factory=dict)

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fusion_score

    @classmethod
    def from_chunk(
        cls,
        chunk: DocumentChunk,
        *,
        dense_score: float = 0.0,
        bm25_score: float = 0.0,
        fusion_score: float = 0.0,
        rerank_score: float | None = None,
        rank_explanation: dict[str, Any] | None = None,
    ) -> RetrievalHit:
        return cls(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            doc_type=chunk.doc_type,
            title=chunk.title,
            citation=chunk.citation,
            jurisdiction=chunk.jurisdiction,
            court=chunk.court,
            date=chunk.date,
            section=chunk.section,
            text=chunk.text,
            dense_score=dense_score,
            bm25_score=bm25_score,
            fusion_score=fusion_score,
            rerank_score=rerank_score,
            rank_explanation=rank_explanation or {},
        )


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_citations: list[str] = Field(default_factory=list)
    expected_answer_patterns: list[str] = Field(default_factory=list)
    answer_type: Literal["statute", "case", "cross_document", "no_answer"] = "statute"
    filters: dict[str, Any] = Field(default_factory=dict)
    must_refuse: bool = False


class RetrievalEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    expected_citations: list[str] = Field(default_factory=list)
    category: str = "general"
    tags: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    expected_missing: bool = False


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    citations: list[str]
    hits: list[RetrievalHit]
    refused: bool = False
