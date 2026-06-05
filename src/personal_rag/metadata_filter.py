from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from personal_rag.schema import DocumentChunk

VALUE_FILTER_FIELDS = frozenset(
    {
        "doc_type",
        "source_id",
        "source_ref",
        "section",
    }
)
METADATA_VALUE_FILTER_FIELDS = frozenset(
    {
        "domain",
        "collection",
        "category",
        "topic",
        "tags",
        "locale",
        "language",
    }
)
DATE_FILTER_FIELDS = frozenset({"date_from", "date_to"})
ALLOWED_FILTER_FIELDS = VALUE_FILTER_FIELDS | METADATA_VALUE_FILTER_FIELDS | DATE_FILTER_FIELDS
SAFE_FILTER_VALUE = re.compile(r"^[\w .:/§&(),+\-]+$")


@dataclass(frozen=True)
class MetadataFilters:
    values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    date_from: date | None = None
    date_to: date | None = None

    @property
    def is_empty(self) -> bool:
        return not self.values and self.date_from is None and self.date_to is None

    def as_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {key: list(value) for key, value in self.values.items()}
        if self.date_from is not None:
            output["date_from"] = self.date_from.isoformat()
        if self.date_to is not None:
            output["date_to"] = self.date_to.isoformat()
        return output


@dataclass(frozen=True)
class DiscardedMetadataFilter:
    source: str
    key: str
    value: Any
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "key": self.key,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MetadataFilterDecision:
    explicit: MetadataFilters
    rewrite: MetadataFilters
    rewrite_accepted: MetadataFilters
    effective: MetadataFilters
    discarded: list[DiscardedMetadataFilter]
    inferred_enabled: bool
    min_confidence: float
    rewrite_confidence: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "explicit_filters": self.explicit.as_dict(),
            "rewrite_filters": self.rewrite.as_dict(),
            "rewrite_accepted_filters": self.rewrite_accepted.as_dict(),
            "effective_filters": self.effective.as_dict(),
            "discarded_filters": [item.as_dict() for item in self.discarded],
            "inferred_filters_enabled": self.inferred_enabled,
            "query_rewrite_filter_min_confidence": self.min_confidence,
            "rewrite_confidence": self.rewrite_confidence,
            "inferred_filters_applied": not self.rewrite_accepted.is_empty,
            "qdrant_filter": qdrant_filter_as_dict(self.effective),
        }


def normalize_metadata_filters(
    *,
    explicit_filters: dict[str, Any] | None,
    rewrite_filters: dict[str, Any] | None,
    rewrite_confidence: float | None,
    inferred_enabled: bool = True,
    min_confidence: float = 0.55,
) -> MetadataFilterDecision:
    explicit, explicit_discarded = _normalize_filter_dict(
        explicit_filters or {},
        source="explicit",
    )
    rewrite, rewrite_discarded = _normalize_filter_dict(
        rewrite_filters or {},
        source="rewrite",
    )
    discarded = [*explicit_discarded, *rewrite_discarded]

    if not inferred_enabled:
        discarded.extend(
            _discard_all(rewrite, source="rewrite", reason="inferred_filters_disabled")
        )
        rewrite_accepted = MetadataFilters()
    elif (rewrite_confidence or 0.0) < min_confidence:
        discarded.extend(
            _discard_all(rewrite, source="rewrite", reason="below_confidence_threshold")
        )
        rewrite_accepted = MetadataFilters()
    else:
        rewrite_accepted, conflict_discarded = _merge_rewrite_filters(explicit, rewrite)
        discarded.extend(conflict_discarded)

    effective = _combine_filters(explicit, rewrite_accepted)
    return MetadataFilterDecision(
        explicit=explicit,
        rewrite=rewrite,
        rewrite_accepted=rewrite_accepted,
        effective=effective,
        discarded=discarded,
        inferred_enabled=inferred_enabled,
        min_confidence=min_confidence,
        rewrite_confidence=rewrite_confidence,
    )


def matches_metadata_filters(chunk: DocumentChunk, filters: MetadataFilters | None) -> bool:
    if filters is None or filters.is_empty:
        return True
    for key, values in filters.values.items():
        value = _chunk_filter_value(chunk, key)
        if value is None or str(value) not in values:
            return False
    if filters.date_from is not None:
        if chunk.date is None or chunk.date < filters.date_from:
            return False
    if filters.date_to is not None:
        if chunk.date is None or chunk.date > filters.date_to:
            return False
    return True


def build_qdrant_filter(filters: MetadataFilters | None):
    if filters is None or filters.is_empty:
        return None

    from qdrant_client.http import models

    must = []
    for key, values in filters.values.items():
        payload_key = f"metadata.{key}" if key in METADATA_VALUE_FILTER_FIELDS else key
        match = (
            models.MatchValue(value=values[0])
            if len(values) == 1
            else models.MatchAny(any=list(values))
        )
        must.append(models.FieldCondition(key=payload_key, match=match))

    if filters.date_from is not None or filters.date_to is not None:
        must.append(
            models.FieldCondition(
                key="date",
                range=models.DatetimeRange(
                    gte=filters.date_from.isoformat() if filters.date_from else None,
                    lte=filters.date_to.isoformat() if filters.date_to else None,
                ),
            )
        )

    return models.Filter(must=must) if must else None


def qdrant_filter_as_dict(filters: MetadataFilters | None) -> dict | None:
    qdrant_filter = build_qdrant_filter(filters)
    return qdrant_filter.model_dump(mode="json") if qdrant_filter is not None else None


def _normalize_filter_dict(
    raw_filters: dict[str, Any],
    *,
    source: str,
) -> tuple[MetadataFilters, list[DiscardedMetadataFilter]]:
    values: dict[str, tuple[str, ...]] = {}
    date_from: date | None = None
    date_to: date | None = None
    discarded: list[DiscardedMetadataFilter] = []

    for key, raw_value in raw_filters.items():
        if key not in ALLOWED_FILTER_FIELDS:
            discarded.append(
                DiscardedMetadataFilter(source, key, raw_value, "unsupported_filter_field")
            )
            continue
        if key in VALUE_FILTER_FIELDS:
            normalized_values, value_discards = _normalize_values(key, raw_value, source=source)
            discarded.extend(value_discards)
            if normalized_values:
                values[key] = tuple(normalized_values)
            continue
        if key in METADATA_VALUE_FILTER_FIELDS:
            normalized_values, value_discards = _normalize_values(key, raw_value, source=source)
            discarded.extend(value_discards)
            if normalized_values:
                values[key] = tuple(normalized_values)
            continue
        parsed_date = _parse_date_filter(key, raw_value, source=source)
        if isinstance(parsed_date, DiscardedMetadataFilter):
            discarded.append(parsed_date)
        elif key == "date_from":
            date_from = parsed_date
        else:
            date_to = parsed_date

    if date_from and date_to and date_from > date_to:
        discarded.extend(
            [
                DiscardedMetadataFilter(
                    source,
                    "date_from",
                    date_from.isoformat(),
                    "invalid_date_range",
                ),
                DiscardedMetadataFilter(
                    source,
                    "date_to",
                    date_to.isoformat(),
                    "invalid_date_range",
                ),
            ]
        )
        date_from = None
        date_to = None

    return MetadataFilters(values=values, date_from=date_from, date_to=date_to), discarded


def _normalize_values(
    key: str,
    raw_value: Any,
    *,
    source: str,
) -> tuple[list[str], list[DiscardedMetadataFilter]]:
    raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
    output: list[str] = []
    seen: set[str] = set()
    discarded: list[DiscardedMetadataFilter] = []
    for item in raw_values:
        if not isinstance(item, str):
            discarded.append(DiscardedMetadataFilter(source, key, item, "invalid_filter_value"))
            continue
        value = item.strip()
        if key == "doc_type":
            value = value.lower()
        elif key in {"locale", "language"}:
            value = value.lower()
        if not value:
            discarded.append(DiscardedMetadataFilter(source, key, item, "empty_filter_value"))
            continue
        if len(value) > 120 or not SAFE_FILTER_VALUE.match(value):
            discarded.append(DiscardedMetadataFilter(source, key, item, "unsafe_filter_value"))
            continue
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output, discarded


def _chunk_filter_value(chunk: DocumentChunk, key: str) -> Any:
    if key in METADATA_VALUE_FILTER_FIELDS:
        return chunk.metadata.get(key)
    return getattr(chunk, key, None)


def _parse_date_filter(
    key: str,
    raw_value: Any,
    *,
    source: str,
) -> date | DiscardedMetadataFilter:
    if not isinstance(raw_value, str):
        return DiscardedMetadataFilter(source, key, raw_value, "invalid_date_value")
    value = raw_value.strip()
    if not value:
        return DiscardedMetadataFilter(source, key, raw_value, "empty_date_value")
    try:
        return date.fromisoformat(value)
    except ValueError:
        return DiscardedMetadataFilter(source, key, raw_value, "invalid_iso_date")


def _merge_rewrite_filters(
    explicit: MetadataFilters,
    rewrite: MetadataFilters,
) -> tuple[MetadataFilters, list[DiscardedMetadataFilter]]:
    accepted_values: dict[str, tuple[str, ...]] = {}
    discarded: list[DiscardedMetadataFilter] = []
    for key, values in rewrite.values.items():
        explicit_values = explicit.values.get(key)
        if explicit_values is None:
            accepted_values[key] = values
        elif set(explicit_values) != set(values):
            discarded.append(
                DiscardedMetadataFilter(
                    "rewrite",
                    key,
                    list(values),
                    "conflicts_with_explicit_filter",
                )
            )

    accepted_date_from = _merge_rewrite_date(
        "date_from",
        explicit.date_from,
        rewrite.date_from,
        discarded,
    )
    accepted_date_to = _merge_rewrite_date(
        "date_to",
        explicit.date_to,
        rewrite.date_to,
        discarded,
    )
    return (
        MetadataFilters(
            values=accepted_values,
            date_from=accepted_date_from,
            date_to=accepted_date_to,
        ),
        discarded,
    )


def _merge_rewrite_date(
    key: str,
    explicit_value: date | None,
    rewrite_value: date | None,
    discarded: list[DiscardedMetadataFilter],
) -> date | None:
    if rewrite_value is None:
        return None
    if explicit_value is None:
        return rewrite_value
    if explicit_value != rewrite_value:
        discarded.append(
            DiscardedMetadataFilter(
                "rewrite",
                key,
                rewrite_value.isoformat(),
                "conflicts_with_explicit_filter",
            )
        )
    return None


def _combine_filters(explicit: MetadataFilters, rewrite: MetadataFilters) -> MetadataFilters:
    return MetadataFilters(
        values={**rewrite.values, **explicit.values},
        date_from=explicit.date_from or rewrite.date_from,
        date_to=explicit.date_to or rewrite.date_to,
    )


def _discard_all(
    filters: MetadataFilters,
    *,
    source: str,
    reason: str,
) -> list[DiscardedMetadataFilter]:
    discarded = [
        DiscardedMetadataFilter(source, key, list(values), reason)
        for key, values in filters.values.items()
    ]
    if filters.date_from is not None:
        discarded.append(
            DiscardedMetadataFilter(
                source,
                "date_from",
                filters.date_from.isoformat(),
                reason,
            )
        )
    if filters.date_to is not None:
        discarded.append(
            DiscardedMetadataFilter(
                source,
                "date_to",
                filters.date_to.isoformat(),
                reason,
            )
        )
    return discarded
