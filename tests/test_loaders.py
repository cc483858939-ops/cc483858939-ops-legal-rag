from __future__ import annotations

from pathlib import Path

from legal_rag.config import SourceItem
from legal_rag.ingest import load_chunks_from_manifest
from legal_rag.loaders import build_chunks
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


def test_section_heading_line_does_not_absorb_body_text() -> None:
    item = SourceItem(
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        court=None,
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
    assert chunks[0].citation == "Personal Test Note v1, Section personal-test."


def test_support_manifest_loads_customer_service_metadata() -> None:
    support_manifest = Path(__file__).resolve().parents[1] / "configs" / "support_sources.yml"
    sources = load_manifest(support_manifest)

    assert len(sources) == 7
    returns = next(source for source in sources if source.source_id == "support-returns-refunds")
    assert returns.doc_type == "policy"
    assert returns.metadata["domain"] == "customer_support"
    assert returns.metadata["category"] == "returns"
    assert returns.path is not None


def test_support_ingest_builds_bilingual_chunks() -> None:
    support_manifest = Path(__file__).resolve().parents[1] / "configs" / "support_sources.yml"
    chunks = load_chunks_from_manifest(support_manifest, max_words=80, overlap_words=10)
    refund_chunks = [chunk for chunk in chunks if chunk.source_id == "support-returns-refunds"]

    assert refund_chunks
    assert any("退款多久到账" in chunk.text for chunk in refund_chunks)
    assert all(chunk.metadata["domain"] == "customer_support" for chunk in refund_chunks)
    assert any("returns.refund_timing" in chunk.citation for chunk in refund_chunks)
