from __future__ import annotations

from personal_rag.text import chunk_text, tokenize


def test_tokenize_preserves_mixed_personal_kb_terms() -> None:
    tokens = tokenize("ShowMaker 是世界第一中单，炫神最喜欢的歌是打火机。")

    assert "showmaker" in tokens
    assert "炫神" in tokens
    assert "火" in tokens


def test_chunk_text_keeps_short_note_as_one_chunk() -> None:
    chunks = chunk_text("个人知识库助手需要先检索证据，再生成回答。", max_words=20, overlap_words=3)

    assert chunks == ["个人知识库助手需要先检索证据，再生成回答。"]
