from __future__ import annotations

from pathlib import Path

from legal_rag.ingest import load_chunks_from_manifest
from legal_rag.manifest import load_manifest


def test_manifest_loads_legal_sources(manifest_path: Path) -> None:
    sources = load_manifest(manifest_path)

    assert len(sources) >= 20
    assert sources[0].source_id == "usc-5-551"
    assert sources[0].citation == "5 U.S.C. § 551"
    assert sources[0].path is not None


def test_ingest_builds_cited_chunks(manifest_path: Path) -> None:
    chunks = load_chunks_from_manifest(manifest_path, max_words=80, overlap_words=10)

    citations = {chunk.citation for chunk in chunks}
    assert "5 U.S.C. § 553" in citations
    assert "Marbury v. Madison, 5 U.S. (1 Cranch) 137 (1803)" in citations
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_section_chunking_preserves_metadata(manifest_path: Path) -> None:
    chunks = load_chunks_from_manifest(manifest_path, max_words=80, overlap_words=10)
    apa_notice = [chunk for chunk in chunks if chunk.source_id == "usc-5-553"]

    assert apa_notice
    assert all(chunk.doc_type == "statute" for chunk in apa_notice)
    assert any("Federal Register" in chunk.text for chunk in apa_notice)
