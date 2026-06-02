from legal_rag.relevance import EvidenceRelevanceGate, matched_query_terms
from legal_rag.schema import RetrievalHit


def personal_note_hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="personal-test-note-1",
        source_id="personal-test-note",
        doc_type="note",
        title="personal-test-note",
        citation="personal-test-note",
        jurisdiction="PERSONAL",
        text=(
            "炫神，又被称为炫狗。他的父亲非常多，但很多就是半孝不孝。"
            "他的父亲包括：Faker, Showmaker, Theshy。"
            "大司马（马老师）是他的结拜大哥。"
        ),
        fusion_score=0.1,
        dense_score=0.0,
        bm25_score=0.0,
    )


def test_relevance_matches_person_entity_query() -> None:
    decision = EvidenceRelevanceGate().evaluate("炫神是谁", [personal_note_hit()])

    assert decision.relevant is True
    assert decision.reason == "query_term_overlap"
    assert decision.matched_terms == ["炫神"]


def test_relevance_matches_relation_query_terms_after_splitting() -> None:
    hit = personal_note_hit()

    assert matched_query_terms("炫神的父亲有哪些", [hit]) == ["炫神", "父亲"]
    assert matched_query_terms("大司马和炫神是什么关系", [hit]) == ["大司马", "炫神"]

    assert EvidenceRelevanceGate().evaluate("炫神的父亲有哪些", [hit]).relevant is True
    assert EvidenceRelevanceGate().evaluate("大司马和炫神是什么关系", [hit]).relevant is True


def test_relevance_keeps_strict_kb_scope_out_of_entity_terms() -> None:
    decision = EvidenceRelevanceGate().evaluate("我的笔记里炫神是谁", [personal_note_hit()])

    assert decision.relevant is True
    assert decision.matched_terms == ["炫神"]


def test_relevance_does_not_treat_ambiguous_pronoun_as_entity() -> None:
    assert matched_query_terms("那他是谁", [personal_note_hit()]) == []

def test_relevance_does_not_accept_bm25_only_for_unrelated_chinese_definition() -> None:
    hit = RetrievalHit(
        chunk_id="unrelated-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="个人知识库助手会记录记忆分类、RAG 指标、ShowMaker 和炫神的个人测试描述。",
        fusion_score=0.05,
        dense_score=0.0,
        bm25_score=3.2,
    )

    decision = EvidenceRelevanceGate().evaluate("灰度域和彩色域是什么", [hit])

    assert decision.relevant is False
    assert decision.reason == "below_relevance_threshold"
    assert decision.top_bm25_score == 3.2
    assert decision.candidate_hit_count == 1
    assert decision.relevant_hit_count == 0


def test_relevance_uses_meaningful_chinese_token_overlap_when_full_phrase_differs() -> None:
    hit = RetrievalHit(
        chunk_id="song-1",
        source_id="personal-test-note",
        doc_type="note",
        title="Personal Test Note",
        citation="Personal Test Note v1",
        jurisdiction="PERSONAL",
        text="打火机歌词里有一句：吉隆坡的天气，他是翻云又覆雨。",
        fusion_score=0.05,
        dense_score=0.0,
        bm25_score=0.0,
    )

    terms = matched_query_terms("吉隆坡天气那句歌词后面是什么", [hit])

    assert "天气" in terms or "歌词" in terms
    assert EvidenceRelevanceGate().evaluate("吉隆坡天气那句歌词后面是什么", [hit]).relevant is True
