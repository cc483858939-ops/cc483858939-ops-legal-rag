from __future__ import annotations

from personal_rag.relevance import EvidenceRelevanceGate, matched_query_terms
from personal_rag.schema import RetrievalHit


def personal_note_hit(text: str | None = None, *, bm25_score: float = 0.0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id="personal-test-note-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        source_ref="Personal Test Note v1",
        text=text
        or (
            "炫神，又被称为炫狗。他的父亲包括：Faker, ShowMaker, TheShy。"
            "大司马是他的结拜大哥。炫神最喜欢的歌是打火机。"
        ),
        fusion_score=0.1,
        dense_score=0.0,
        bm25_score=bm25_score,
    )


def test_relevance_matches_person_entity_query() -> None:
    decision = EvidenceRelevanceGate().evaluate("炫神是谁", [personal_note_hit()])

    assert decision.relevant is True
    assert decision.reason == "query_term_overlap"
    assert "炫神" in decision.matched_terms


def test_relevance_matches_relation_query_terms_after_splitting() -> None:
    hit = personal_note_hit()

    assert "父亲" in matched_query_terms("炫神的父亲有哪些", [hit])
    assert EvidenceRelevanceGate().evaluate("大司马和炫神是什么关系", [hit]).relevant is True


def test_relevance_does_not_treat_ambiguous_pronoun_as_entity() -> None:
    assert matched_query_terms("那他是谁", [personal_note_hit()]) == []


def test_relevance_does_not_accept_bm25_only_for_unrelated_chinese_definition() -> None:
    hit = personal_note_hit(
        "个人知识库助手会记录记忆分类、RAG 指标、ShowMaker 和炫神的测试描述。",
        bm25_score=3.2,
    )

    decision = EvidenceRelevanceGate().evaluate("灰度域和彩色域是什么", [hit])

    assert decision.relevant is False
    assert decision.reason == "below_relevance_threshold"
    assert decision.top_bm25_score == 3.2


def test_relevance_uses_meaningful_chinese_token_overlap_when_full_phrase_differs() -> None:
    hit = personal_note_hit("打火机歌词里有一句：吉隆坡的天气，他是翻云又覆雨。")

    terms = matched_query_terms("吉隆坡天气那句歌词后面是什么", [hit])

    assert "天气" in terms or "歌词" in terms
    assert EvidenceRelevanceGate().evaluate("吉隆坡天气那句歌词后面是什么", [hit]).relevant is True
