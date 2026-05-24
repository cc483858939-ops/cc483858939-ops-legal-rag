from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CHUNK_BOUNDARY_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?|\n+")
CJK_STOP_TOKENS = {
    "什",
    "么",
    "什么",
    "怎",
    "怎么",
    "哪",
    "哪些",
    "谁",
    "是",
    "的",
    "了",
    "吗",
    "呢",
    "啊",
    "和",
    "与",
    "或",
    "在",
    "里",
    "中",
    "最",
    "喜欢",
}


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
    grams = [char for char in chars if char not in CJK_STOP_TOKENS]
    grams.extend(
        gram
        for gram in ("".join(chars[index : index + 2]) for index in range(len(chars) - 1))
        if gram not in CJK_STOP_TOKENS
    )
    return grams


def chunk_text(text: str, *, max_words: int = 300, overlap_words: int = 40) -> list[str]:
    segments = _sentence_segments(text)
    if not segments:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_units = 0

    for segment in segments:
        segment_units = _chunk_units(segment)
        if segment_units > max_words:
            if current:
                chunks.append(_join_chunk_segments(current))
                current = []
                current_units = 0
            chunks.extend(
                _split_long_segment(
                    segment,
                    max_units=max_words,
                    overlap_units=overlap_words,
                )
            )
            continue

        if current and current_units + segment_units > max_words:
            chunks.append(_join_chunk_segments(current))
            current, current_units = _overlap_segments(current, overlap_words)
            if current and current_units + segment_units > max_words:
                current = []
                current_units = 0

        current.append(segment)
        current_units += segment_units

    if current:
        chunks.append(_join_chunk_segments(current))

    return [chunk for chunk in chunks if chunk]


def _sentence_segments(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    if not normalized:
        return []
    return [
        segment
        for segment in (match.group(0).strip() for match in CHUNK_BOUNDARY_RE.finditer(normalized))
        if segment
    ]


def _chunk_units(text: str) -> int:
    units = 0
    for token in TOKEN_RE.findall(text):
        if CJK_RE.search(token):
            units += len([char for char in token if CJK_RE.match(char)])
        else:
            units += 1
    return units


def _join_chunk_segments(segments: list[str]) -> str:
    joined = " ".join(segment.strip() for segment in segments if segment.strip())
    return normalize_whitespace(joined)


def _overlap_segments(segments: list[str], overlap_units: int) -> tuple[list[str], int]:
    if overlap_units <= 0:
        return [], 0
    selected: list[str] = []
    total = 0
    for segment in reversed(segments):
        units = _chunk_units(segment)
        if selected and total + units > overlap_units:
            break
        selected.append(segment)
        total += units
        if total >= overlap_units:
            break
    selected.reverse()
    return selected, total


def _split_long_segment(segment: str, *, max_units: int, overlap_units: int) -> list[str]:
    spans = _unit_spans(segment)
    if not spans:
        return [normalize_whitespace(segment)] if segment.strip() else []

    chunks: list[str] = []
    start = 0
    step = max(1, max_units - max(0, overlap_units))
    while start < len(spans):
        end = min(len(spans), start + max_units)
        char_start = spans[start][0]
        char_end = spans[end - 1][1]
        chunk = normalize_whitespace(segment[char_start:char_end])
        if chunk:
            chunks.append(chunk)
        if end == len(spans):
            break
        start += step
    return chunks


def _unit_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if CJK_RE.search(token):
            spans.extend((index, index + 1) for index in range(match.start(), match.end()))
        else:
            spans.append(match.span())
    return spans


def stable_id(parts: Iterable[str]) -> str:
    import hashlib

    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]
