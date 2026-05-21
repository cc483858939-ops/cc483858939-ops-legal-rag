from __future__ import annotations

import re
from dataclasses import dataclass

from legal_rag.schema import Answer, RetrievalHit
from legal_rag.text import normalize_whitespace, tokenize

REFUSAL = (
    "I do not have enough retrieved legal authority in the configured corpus to answer "
    "that. This is legal information only, not legal advice."
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "say",
    "says",
    "the",
    "this",
    "to",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
}

GENERIC_LEGAL_TERMS = {
    "act",
    "advice",
    "answer",
    "case",
    "citation",
    "claim",
    "court",
    "current",
    "law",
    "legal",
    "question",
    "rights",
    "state",
    "statute",
}

OUT_OF_CORPUS_TERMS = {
    "alien",
    "asylum",
    "california",
    "ccpa",
    "delaware",
    "election",
    "invent",
    "ny",
    "rate",
    "tax",
    "tomorrow",
    "york",
}

QUESTION_CONCEPTS = {
    "chevron": {"chevron", "agency", "interpretation", "statute"},
    "chenery": {"chenery", "agency", "grounds", "invoked", "explanation", "review"},
    "iqbal": {"iqbal", "plausible", "plausibly", "conclusory", "pleading"},
    "marbury": {"marbury", "judicial", "review", "federal", "law"},
    "monell": {"monell", "municipal", "liability", "policy", "custom", "1983"},
    "twombly": {"twombly", "plausible", "pleading", "complaint"},
}


@dataclass
class ExtractiveAnswerer:
    min_overlap: int = 1
    min_overlap_ratio: float = 0.35

    def answer(self, question: str, hits: list[RetrievalHit], *, max_citations: int = 4) -> Answer:
        supported_hits = self._supported_hits(question, hits)
        if not supported_hits:
            return Answer(question=question, answer=REFUSAL, citations=[], hits=hits, refused=True)

        selected = supported_hits[:max_citations]
        sentences = self._select_sentences(question, selected)
        if not sentences:
            sentences = [normalize_whitespace(selected[0].text)[:500]]
        citations = list(dict.fromkeys(hit.citation for hit in selected))
        body = " ".join(sentences)
        citation_text = "; ".join(citations)
        return Answer(
            question=question,
            answer=f"{body} Sources: {citation_text}. This is legal information, not legal advice.",
            citations=citations,
            hits=hits,
            refused=False,
        )

    def _has_support(self, question: str, hits: list[RetrievalHit]) -> bool:
        return bool(self._supported_hits(question, hits))

    def _supported_hits(self, question: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if self._looks_out_of_corpus(question):
            return []
        query_terms = self._meaningful_terms(question)
        if not query_terms:
            return hits
        required_authorities = self._authority_terms(question)
        concept_terms = self._concept_terms(question)
        supported: list[RetrievalHit] = []
        for hit in hits[:5]:
            support_terms = self._hit_terms(hit)
            overlap = query_terms & support_terms
            if required_authorities & support_terms:
                supported.append(hit)
                continue
            if concept_terms and concept_terms & support_terms:
                supported.append(hit)
                continue
            overlap_ratio = len(overlap) / len(query_terms)
            if len(overlap) >= self.min_overlap and overlap_ratio >= self.min_overlap_ratio:
                supported.append(hit)
        return supported

    def _select_sentences(self, question: str, hits: list[RetrievalHit]) -> list[str]:
        query_terms = self._meaningful_terms(question)
        authority_terms = self._authority_terms(question)
        concept_terms = self._concept_terms(question)
        scored: list[tuple[int, int, str]] = []
        for hit in hits:
            hit_terms = self._hit_terms(hit)
            for sentence in re.split(r"(?<=[.!?])\s+", normalize_whitespace(hit.text)):
                terms = set(tokenize(sentence))
                score = len(query_terms & terms)
                if concept_terms:
                    score += 3 * len(concept_terms & terms)
                if authority_terms & hit_terms:
                    score += 2
                if not terms - set(tokenize(hit.title)) and score <= 2:
                    continue
                if score > 0:
                    scored.append((score, len(sentence), sentence))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for _, _, sentence in scored:
            if sentence not in seen:
                selected.append(sentence)
                seen.add(sentence)
            if len(selected) >= 3:
                break
        return selected

    def _hit_terms(self, hit: RetrievalHit) -> set[str]:
        return set(tokenize(f"{hit.title} {hit.citation} {hit.section or ''} {hit.text}"))

    def _meaningful_terms(self, text: str) -> set[str]:
        return {
            term
            for term in tokenize(text)
            if len(term) > 1 and term not in STOPWORDS and term not in GENERIC_LEGAL_TERMS
        }

    def _authority_terms(self, text: str) -> set[str]:
        terms = set()
        terms.update(re.findall(r"\b\d+\b", text.lower()))
        for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+v\.\s+[A-Z][a-z]+)?\b", text):
            for token in tokenize(phrase):
                if token not in STOPWORDS and token not in GENERIC_LEGAL_TERMS:
                    terms.add(token)
        return terms

    def _concept_terms(self, text: str) -> set[str]:
        query_terms = set(tokenize(text))
        concepts: set[str] = set()
        for trigger, terms in QUESTION_CONCEPTS.items():
            if trigger in query_terms:
                concepts.update(terms)
        return concepts

    def _looks_out_of_corpus(self, question: str) -> bool:
        terms = set(tokenize(question))
        if not terms & OUT_OF_CORPUS_TERMS:
            return False
        return not re.search(r"\b\d+\s+u\.?s\.?c\.?\b|\bv\.\b", question, re.IGNORECASE)
