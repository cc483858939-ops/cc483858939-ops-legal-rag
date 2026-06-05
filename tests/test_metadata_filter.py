from __future__ import annotations

from datetime import date

from personal_rag.metadata_filter import (
    MetadataFilters,
    build_qdrant_filter,
    matches_metadata_filters,
    normalize_metadata_filters,
)
from personal_rag.schema import DocumentChunk


def test_explicit_filters_are_hard_filters() -> None:
    decision = normalize_metadata_filters(
        explicit_filters={"doc_type": "note", "source_ref": "Personal Test Note v1"},
        rewrite_filters={},
        rewrite_confidence=None,
    )

    assert decision.effective.as_dict() == {
        "doc_type": ["note"],
        "source_ref": ["Personal Test Note v1"],
    }


def test_low_confidence_rewrite_filters_are_not_applied() -> None:
    decision = normalize_metadata_filters(
        explicit_filters={},
        rewrite_filters={"doc_type": "note", "topic": "personal_test"},
        rewrite_confidence=0.2,
        min_confidence=0.55,
    )

    assert decision.effective.as_dict() == {}
    assert decision.rewrite.as_dict() == {"doc_type": ["note"], "topic": ["personal_test"]}
    assert {item.reason for item in decision.discarded} == {"below_confidence_threshold"}


def test_rewrite_filters_conflicting_with_explicit_filters_are_discarded() -> None:
    decision = normalize_metadata_filters(
        explicit_filters={"doc_type": "note"},
        rewrite_filters={"doc_type": "document", "topic": "personal_test"},
        rewrite_confidence=0.9,
    )

    assert decision.effective.as_dict() == {
        "doc_type": ["note"],
        "topic": ["personal_test"],
    }
    assert decision.rewrite_accepted.as_dict() == {"topic": ["personal_test"]}
    assert decision.discarded[0].reason == "conflicts_with_explicit_filter"


def test_invalid_fields_and_values_are_discarded() -> None:
    decision = normalize_metadata_filters(
        explicit_filters={
            "doc_type": ["note", "<script>"],
            "date_from": "not-a-date",
            "legacy_field": "legacy",
        },
        rewrite_filters={},
        rewrite_confidence=None,
    )

    assert decision.effective.as_dict() == {"doc_type": ["note"]}
    assert {item.reason for item in decision.discarded} == {
        "unsafe_filter_value",
        "invalid_iso_date",
        "unsupported_filter_field",
    }


def test_canonical_qdrant_filter_uses_match_any_and_date_range() -> None:
    qdrant_filter = build_qdrant_filter(
        MetadataFilters(
            values={"doc_type": ("note", "document"), "source_ref": ("Personal Test Note v1",)},
            date_from=date(2026, 1, 1),
            date_to=date(2026, 12, 31),
        )
    )

    payload = qdrant_filter.model_dump(mode="json")
    assert payload["must"][0]["key"] == "doc_type"
    assert payload["must"][0]["match"]["any"] == ["note", "document"]
    assert payload["must"][1]["key"] == "source_ref"
    assert payload["must"][1]["match"]["value"] == "Personal Test Note v1"
    assert payload["must"][2]["key"] == "date"


def test_qdrant_filter_uses_nested_metadata_payload_keys() -> None:
    qdrant_filter = build_qdrant_filter(
        MetadataFilters(values={"domain": ("personal_kb",), "topic": ("personal_test",)})
    )

    payload = qdrant_filter.model_dump(mode="json")
    assert payload["must"][0]["key"] == "metadata.domain"
    assert payload["must"][1]["key"] == "metadata.topic"


def test_in_memory_filter_matching_uses_top_level_and_nested_metadata() -> None:
    chunk = DocumentChunk(
        chunk_id="doc-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        source_ref="Personal Test Note v1",
        date=date(2026, 5, 19),
        section="personal-test",
        text="Personal note text.",
        metadata={"domain": "personal_kb", "topic": "personal_test", "language": "zh"},
    )

    assert matches_metadata_filters(
        chunk,
        MetadataFilters(
            values={"doc_type": ("note",), "topic": ("personal_test",)},
            date_from=date(2026, 1, 1),
        ),
    )
    assert not matches_metadata_filters(chunk, MetadataFilters(values={"doc_type": ("document",)}))
