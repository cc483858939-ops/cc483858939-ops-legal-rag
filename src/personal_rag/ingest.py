from __future__ import annotations

from pathlib import Path

from personal_rag.loaders import load_source
from personal_rag.manifest import load_manifest
from personal_rag.schema import DocumentChunk
from personal_rag.store import PersonalVectorStore


def load_chunks_from_manifest(
    manifest_path: str | Path,
    *,
    max_words: int = 300,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for item in load_manifest(manifest_path):
        chunks.extend(load_source(item, max_words=max_words, overlap_words=overlap_words))
    return chunks


def ingest_manifest(
    manifest_path: str | Path,
    store: PersonalVectorStore,
    *,
    max_words: int = 300,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    chunks = load_chunks_from_manifest(
        manifest_path,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    store.upsert_chunks(chunks)
    return chunks
