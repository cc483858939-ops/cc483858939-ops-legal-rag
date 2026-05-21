from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    normalized = re.sub(r"\bu\.s\.c\.", "usc", normalized)
    normalized = re.sub(r"\bu\.s\.", "us", normalized)
    tokens: list[str] = []
    for token in TOKEN_RE.findall(normalized):
        if CJK_RE.search(token):
            tokens.extend(_cjk_bigrams(token))
        elif len(token) > 1:
            tokens.append(token)
    return tokens


def _cjk_bigrams(token: str) -> list[str]:
    chars = [char for char in token if CJK_RE.match(char)]
    if len(chars) <= 1:
        return chars
    return ["".join(chars[index : index + 2]) for index in range(len(chars) - 1)]


def chunk_text(text: str, *, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    words = normalize_whitespace(text).split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    step = max(1, max_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def stable_id(parts: Iterable[str]) -> str:
    import hashlib

    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]
