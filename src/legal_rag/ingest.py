from __future__ import annotations

from pathlib import Path

from legal_rag.loaders import load_source
from legal_rag.manifest import load_manifest
from legal_rag.schema import DocumentChunk
from legal_rag.store import LegalVectorStore


def load_chunks_from_manifest(
    manifest_path: str | Path,
    *,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for item in load_manifest(manifest_path):
        chunks.extend(load_source(item, max_words=max_words, overlap_words=overlap_words))
    return chunks


def ingest_manifest(
    manifest_path: str | Path,
    store: LegalVectorStore,
    *,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    chunks = load_chunks_from_manifest(
        manifest_path,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    store.upsert_chunks(chunks)
    return chunks
