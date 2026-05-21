from __future__ import annotations

from legal_rag.text import tokenize


def test_tokenize_preserves_english_legal_tokens() -> None:
    assert tokenize("5 U.S.C. 553 notice") == ["usc", "553", "notice"]


def test_tokenize_adds_cjk_bigrams_for_chinese_queries() -> None:
    tokens = tokenize("退款多久到账")

    assert "退款" in tokens
    assert "多久" in tokens
    assert "到账" in tokens
