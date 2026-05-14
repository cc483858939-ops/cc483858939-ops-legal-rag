from __future__ import annotations

from datetime import date

from legal_rag.metadata_filter import (
    MetadataFilters,
    build_qdrant_filter,
    matches_metadata_filters,
    normalize_metadata_filters,
)
from legal_rag.schema import DocumentChunk


def test_explicit_filters_are_hard_filters() -> None:
    decision = normalize_metadata_filters(
        explicit_filters={"doc_type": "case", "jurisdiction": "us"},
        rewrite_filters={},
        rewrite_confidence=None,
    )

    assert decision.effective.as_dict() == {
        "doc_type": ["case"],
        "jurisdiction": ["US"],
    }


def test_low_confidence_rewrite_filters_are_not_applied() -> None:
    decision = normalize_metadata_filters(
        explicit_filters={},
        rewrite_filters={"doc_type": "case", "jurisdiction": "US"},
        rewrite_confidence=0.2,
        min_confidence=0.55,
    )

    assert decision.effective.as_dict() == {}
    assert decision.rewrite.as_dict() == {"doc_type": ["case"], "jurisdiction": ["US"]}
    assert {item.reason for item in decision.discarded} == {"below_confidence_threshold"}


def test_rewrite_filters_conflicting_with_explicit_filters_are_discarded() -> None:
    decision = normalize_metadata_filters(
        explicit_filters={"doc_type": "statute"},
        rewrite_filters={"doc_type": "case", "jurisdiction": "US"},
        rewrite_confidence=0.9,
    )

    assert decision.effective.as_dict() == {
        "doc_type": ["statute"],
        "jurisdiction": ["US"],
    }
    assert decision.rewrite_accepted.as_dict() == {"jurisdiction": ["US"]}
    assert decision.discarded[0].reason == "conflicts_with_explicit_filter"


def test_invalid_fields_and_values_are_discarded() -> None:
    decision = normalize_metadata_filters(
        explicit_filters={
            "doc_type": ["case", "bad"],
            "date_from": "not-a-date",
            "topic": "copyright",
        },
        rewrite_filters={},
        rewrite_confidence=None,
    )

    assert decision.effective.as_dict() == {"doc_type": ["case"]}
    assert {item.reason for item in decision.discarded} == {
        "invalid_doc_type",
        "invalid_iso_date",
        "unsupported_filter_field",
    }


def test_canonical_qdrant_filter_uses_match_any_and_date_range() -> None:
    qdrant_filter = build_qdrant_filter(
        MetadataFilters(
            values={"doc_type": ("case", "statute"), "jurisdiction": ("US",)},
            date_from=date(2000, 1, 1),
            date_to=date(2024, 12, 31),
        )
    )

    payload = qdrant_filter.model_dump(mode="json")
    assert payload["must"][0]["key"] == "doc_type"
    assert payload["must"][0]["match"]["any"] == ["case", "statute"]
    assert payload["must"][1]["key"] == "jurisdiction"
    assert payload["must"][1]["match"]["value"] == "US"
    assert payload["must"][2]["key"] == "date"
    assert payload["must"][2]["range"]["gte"] == "2000-01-01T00:00:00"


def test_in_memory_filter_matching_uses_top_level_metadata() -> None:
    chunk = DocumentChunk(
        chunk_id="loper-1",
        source_id="loper-bright",
        doc_type="case",
        title="Loper Bright",
        citation="Loper Bright Enterprises v. Raimondo, 603 U.S. 369 (2024)",
        jurisdiction="US",
        court="Supreme Court",
        date=date(2024, 6, 28),
        text="Chevron was overruled.",
    )

    assert matches_metadata_filters(
        chunk,
        MetadataFilters(
            values={"doc_type": ("case",), "court": ("Supreme Court",)},
            date_from=date(2024, 1, 1),
        ),
    )
    assert not matches_metadata_filters(
        chunk,
        MetadataFilters(values={"doc_type": ("statute",)}),
    )
