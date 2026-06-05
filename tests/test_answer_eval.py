from __future__ import annotations

from personal_rag.eval import source_recall


def test_source_recall_scores_expected_sources() -> None:
    assert source_recall(["Personal Test Note v1"], ["Personal Test Note v1"]) == 1.0
    assert source_recall(["Personal Test Note v1"], ["Other Note"]) == 0.0
    assert source_recall([], ["Other Note"]) == 1.0
