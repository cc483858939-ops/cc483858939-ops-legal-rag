from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from legal_rag.config import SourceItem
from legal_rag.schema import DocumentChunk
from legal_rag.text import chunk_text, normalize_whitespace, stable_id


def load_source(
    item: SourceItem,
    *,
    max_words: int = 300,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    if not item.path:
        raise ValueError(f"Source {item.source_id} has no local path. Fetch it before ingesting.")

    path = Path(item.path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".xml"}:
        text = _read_xml_text(path)
    else:
        text = path.read_text(encoding="utf-8")

    return build_chunks(item, text, max_words=max_words, overlap_words=overlap_words)


def build_chunks(
    item: SourceItem,
    text: str,
    *,
    max_words: int = 300,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    parsed_date = date.fromisoformat(item.date) if item.date else None
    section_blocks = _split_sections(text)
    chunks: list[DocumentChunk] = []

    for section_index, (section, section_text) in enumerate(section_blocks):
        chunk_texts = chunk_text(section_text, max_words=max_words, overlap_words=overlap_words)
        for chunk_index, content in enumerate(chunk_texts):
            chunk_id = stable_id(
                [item.source_id, section or "", str(section_index), str(chunk_index), content]
            )
            citation = _section_citation(item.citation, section)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_id=item.source_id,
                    doc_type=item.doc_type,  # type: ignore[arg-type]
                    title=item.title,
                    citation=citation,
                    jurisdiction=item.jurisdiction,
                    court=item.court,
                    date=parsed_date,
                    section=section or item.section,
                    text=content,
                    metadata={
                        **item.metadata,
                        "source_path": item.path,
                        "source_url": item.url,
                        "chunk_index": chunk_index,
                        "section_index": section_index,
                    },
                )
            )
    return chunks


def _read_xml_text(path: Path) -> str:
    root = ElementTree.parse(path).getroot()
    parts: list[str] = []
    for element in root.iter():
        if element.text and element.text.strip():
            parts.append(element.text.strip())
    return normalize_whitespace(" ".join(parts))


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    normalized = text.replace("\r\n", "\n")
    pattern = re.compile(
        r"(?m)^[^\S\r\n]*(Section|§)[^\S\r\n]+([A-Za-z0-9_.-]+)\.?[^\S\r\n]*(.*)$"
    )
    matches = list(pattern.finditer(normalized))
    if not matches:
        return [(None, normalize_whitespace(normalized))]

    blocks: list[tuple[str | None, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        heading = " ".join(part for part in match.groups() if part)
        blocks.append((heading, normalize_whitespace(normalized[start:end])))
    return blocks


def _section_citation(base_citation: str, section: str | None) -> str:
    if not section:
        return base_citation
    section_number = re.search(
        r"\b([A-Za-z0-9_.-]+)\b",
        section.replace("Section", "").replace("§", ""),
    )
    if section in base_citation or (section_number and section_number.group(1) in base_citation):
        return base_citation
    return f"{base_citation}, {section}"
