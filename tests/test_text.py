from __future__ import annotations

from legal_rag.text import chunk_text, tokenize


def test_tokenize_preserves_english_legal_tokens() -> None:
    assert tokenize("5 U.S.C. 553 notice") == ["usc", "553", "notice"]


def test_tokenize_adds_cjk_bigrams_for_chinese_queries() -> None:
    tokens = tokenize("退款多久到账")

    assert "退" in tokens
    assert "退款" in tokens
    assert "多久" in tokens
    assert "到账" in tokens


def test_chunk_text_splits_chinese_on_sentence_boundaries() -> None:
    text = (
        "Agent 记忆可以分成短期会话记忆、长期事实记忆和任务轨迹记忆。"
        "RAG 回答质量不能只看最终答案，还要看检索是否命中。"
        "个人知识库助手和智能客服很像，都需要先检索知识库。"
        "炫神最喜欢的歌是打火机，因为其中的一句歌词是吉隆坡的天气。"
    )

    chunks = chunk_text(text, max_words=40, overlap_words=0)

    assert len(chunks) >= 3
    assert all(chunk.endswith(("。", "！", "？", "!", "?", ";", "；")) for chunk in chunks)
    assert any("炫神最喜欢的歌是打火机" in chunk for chunk in chunks)
