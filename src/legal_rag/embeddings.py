from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from legal_rag.text import tokenize


class DenseEmbedder(Protocol):
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""


@dataclass
class HashingEmbedder:
    """Deterministic local embedder for tests and offline smoke runs."""

    dimensions: int = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=float)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dimensions
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if math.isclose(float(norm), 0.0):
            return vector.tolist()
        return (vector / norm).tolist()


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": device} if device else {}
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)
        self.dimensions = self._detect_dimensions()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embedding.tolist() for embedding in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _detect_dimensions(self) -> int:
        for meta in self.model.list_supported_models():
            if meta.get("model") == self.model_name:
                return int(meta["dim"])
        return len(self.embed_query("dimension probe"))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_arr = np.array(left, dtype=float)
    right_arr = np.array(right, dtype=float)
    denom = np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
    if math.isclose(float(denom), 0.0):
        return 0.0
    return float(np.dot(left_arr, right_arr) / denom)
