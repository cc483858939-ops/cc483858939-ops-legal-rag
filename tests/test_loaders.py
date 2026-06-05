from __future__ import annotations

from pathlib import Path

from personal_rag.config import ROOT_DIR, SourceItem
from personal_rag.ingest import load_chunks_from_manifest
from personal_rag.loaders import build_chunks
from personal_rag.manifest import load_manifest


def test_manifest_loads_personal_sources() -> None:
    sources = load_manifest(ROOT_DIR / "configs" / "personal_test_sources.yml")

    assert len(sources) == 1
    assert sources[0].source_id == "personal-test-note"
    assert sources[0].doc_type == "note"
    assert sources[0].source_ref == "Personal Test Note v1, Section personal-test"
    assert sources[0].path is not None


def test_ingest_builds_personal_chunks() -> None:
    chunks = load_chunks_from_manifest(
        ROOT_DIR / "configs" / "personal_test_sources.yml",
        max_words=80,
        overlap_words=10,
    )

    assert chunks
    assert {chunk.source_ref for chunk in chunks} == {"Personal Test Note v1, Section personal-test"}
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_section_heading_line_does_not_absorb_body_text() -> None:
    item = SourceItem(
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        source_ref="Personal Test Note v1",
        date="2026-05-19",
        section="personal-test",
        path=None,
        url=None,
        metadata={},
    )

    chunks = build_chunks(
        item,
        "Section personal-test.\nAgent memory belongs in the body.",
        max_words=80,
        overlap_words=10,
    )

    assert chunks[0].section == "Section personal-test."
    assert chunks[0].source_ref == "Personal Test Note v1, Section personal-test."


def test_manifest_resolves_relative_path() -> None:
    source = load_manifest(ROOT_DIR / "configs" / "personal_test_sources.yml")[0]

    assert Path(source.path or "").name == "my_note.txt"
