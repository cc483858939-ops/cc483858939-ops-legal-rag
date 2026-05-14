from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from legal_rag.config import SourceItem


def load_manifest(path: str | Path) -> list[SourceItem]:
    manifest_path = Path(path)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    sources = data.get("sources", [])
    items: list[SourceItem] = []
    for raw in sources:
        items.append(
            SourceItem(
                source_id=str(raw["source_id"]),
                doc_type=str(raw.get("doc_type", "other")),
                title=str(raw["title"]),
                citation=str(raw["citation"]),
                jurisdiction=str(raw.get("jurisdiction", "US")),
                court=raw.get("court"),
                date=str(raw["date"]) if raw.get("date") else None,
                section=raw.get("section"),
                path=_resolve_relative(manifest_path.parent, raw.get("path")),
                url=raw.get("url"),
            )
        )
    return items


def _resolve_relative(base_dir: Path, value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())
