from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from legal_rag.schema import RetrievalHit
from legal_rag.text import tokenize

ONNX_CROSS_ENCODER_MODELS = {"cross-encoder/ms-marco-MiniLM-L6-v2"}


class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Return hits ordered by reranker score."""


@dataclass
class LexicalReranker:
    """Deterministic reranker for tests."""

    model_name: str = "fake-lexical-reranker"

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        query_terms = set(tokenize(query))
        reranked: list[RetrievalHit] = []
        for hit in hits:
            terms = set(tokenize(hit.text))
            score = len(query_terms & terms) / max(len(query_terms), 1)
            reranked.append(
                hit.model_copy(
                    update={
                        "rerank_score": float(score),
                        "rank_explanation": {
                            **hit.rank_explanation,
                            "reranker": {
                                "model": self.model_name,
                                "score": float(score),
                            },
                        },
                    }
                )
            )
        reranked.sort(key=lambda item: (item.rerank_score or 0.0, item.fusion_score), reverse=True)
        return reranked


@dataclass
class UnavailableReranker:
    model_name: str
    error: str

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        return hits


class SentenceTransformerReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "RERANKER_MODEL requires the optional sentence-transformers dependency. "
                "Build with INSTALL_MODEL_EXTRAS=true or leave RERANKER_MODEL empty to skip rerank."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, hit.text) for hit in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        reranked = [
            hit.model_copy(
                update={
                    "rerank_score": float(score),
                    "rank_explanation": {
                        **hit.rank_explanation,
                        "reranker": {
                            "model": self.model_name,
                            "score": float(score),
                        },
                    },
                }
            )
            for hit, score in zip(hits, scores, strict=True)
        ]
        reranked.sort(key=lambda item: (item.rerank_score or 0.0, item.fusion_score), reverse=True)
        return reranked


class OnnxCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 4,
        max_length: int = 512,
        onnx_file: str = "onnx/model_quint8_avx2.onnx",
    ) -> None:
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession
        from tokenizers import Tokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file(
            hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        )
        self.session = InferenceSession(
            hf_hub_download(repo_id=model_name, filename=onnx_file),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []
        scores: list[float] = []
        for offset in range(0, len(hits), self.batch_size):
            batch = hits[offset : offset + self.batch_size]
            scores.extend(self._predict_batch(query, batch))
        reranked = [
            hit.model_copy(
                update={
                    "rerank_score": score,
                    "rank_explanation": {
                        **hit.rank_explanation,
                        "reranker": {
                            "model": self.model_name,
                            "backend": "onnx",
                            "score": score,
                        },
                    },
                }
            )
            for hit, score in zip(hits, scores, strict=True)
        ]
        reranked.sort(key=lambda item: (item.rerank_score or 0.0, item.fusion_score), reverse=True)
        return reranked

    def _predict_batch(self, query: str, hits: list[RetrievalHit]) -> list[float]:
        encodings = [self.tokenizer.encode(query, hit.text) for hit in hits]
        max_length = min(self.max_length, max(len(encoding.ids) for encoding in encodings))
        pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        input_ids: list[list[int]] = []
        token_type_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for encoding in encodings:
            ids = encoding.ids[:max_length]
            type_ids = encoding.type_ids[:max_length]
            mask = encoding.attention_mask[:max_length]
            pad_count = max_length - len(ids)
            input_ids.append(ids + [pad_id] * pad_count)
            token_type_ids.append(type_ids + [0] * pad_count)
            attention_mask.append(mask + [0] * pad_count)

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.asarray(input_ids, dtype=np.int64)
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.asarray(token_type_ids, dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.asarray(attention_mask, dtype=np.int64)
        output = self.session.run(None, feed)[0]
        return [float(score) for score in np.asarray(output).reshape(-1)]


def build_optional_reranker(
    model_name: str | None,
    *,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
    onnx_file: str = "onnx/model_quint8_avx2.onnx",
    fail_open: bool = True,
) -> Reranker | None:
    if not model_name or not model_name.strip():
        return None
    clean_model_name = model_name.strip()
    if clean_model_name.lower() in {"none", "off", "disabled"}:
        return None
    if clean_model_name == "fake-lexical-reranker":
        return LexicalReranker()
    try:
        if clean_model_name in ONNX_CROSS_ENCODER_MODELS:
            return OnnxCrossEncoderReranker(
                clean_model_name,
                batch_size=batch_size,
                max_length=max_length,
                onnx_file=onnx_file,
            )
        return SentenceTransformerReranker(
            clean_model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    except Exception as exc:
        if not fail_open:
            raise
        return UnavailableReranker(
            model_name=clean_model_name,
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )


def rerank_hits(
    reranker: Reranker | None,
    query: str,
    hits: list[RetrievalHit],
    *,
    top_n: int,
) -> tuple[list[RetrievalHit], dict]:
    started = time.perf_counter()
    summary = {
        "enabled": reranker is not None,
        "skipped": reranker is None,
        "model": getattr(reranker, "model_name", None),
        "query": query,
        "input_count": len(hits),
        "reranked_count": 0,
        "top_n": top_n,
        "duration_ms": 0.0,
        "error": None,
        "reason": None,
    }
    if reranker is None:
        summary["reason"] = "reranker_not_configured"
        summary["duration_ms"] = elapsed_ms(started)
        return hits, summary

    if isinstance(reranker, UnavailableReranker):
        summary["skipped"] = True
        summary["reason"] = "reranker_unavailable"
        summary["error"] = reranker.error
        summary["duration_ms"] = elapsed_ms(started)
        return hits, summary

    try:
        rerank_count = min(max(1, top_n), len(hits))
        candidates = hits[:rerank_count]
        tail = hits[rerank_count:]
        reranked = reranker.rerank(query, candidates) + tail
        summary["skipped"] = False
        summary["reranked_count"] = rerank_count
        summary["duration_ms"] = elapsed_ms(started)
        return reranked, summary
    except Exception as exc:  # noqa: BLE001 - rerank must fail open.
        summary["skipped"] = True
        summary["reason"] = "reranker_error"
        summary["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        summary["duration_ms"] = elapsed_ms(started)
        return hits, summary


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
