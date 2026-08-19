"""Lightweight lexical retrieval used by the hybrid RAG pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
    "work",
    "works",
    "postgresql",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


@dataclass(frozen=True)
class LexicalChunk:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    embedding: Sequence[float]


@dataclass(frozen=True)
class LexicalMatch:
    chunk: LexicalChunk
    score: float
    matched_terms: int
    query_terms: int

    @property
    def coverage(self) -> float:
        return self.matched_terms / self.query_terms if self.query_terms else 0.0


class Bm25Index:
    """In-memory BM25 index with postings lists for a small local corpus."""

    def __init__(
        self,
        chunks: list[LexicalChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.document_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for index, chunk in enumerate(chunks):
            frequencies = Counter(tokenize(chunk.document))
            self.document_lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                self.postings[term].append((index, frequency))
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 1.0
        )

    def search(self, query: str, limit: int) -> list[LexicalMatch]:
        if not self.chunks or limit <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []
        matched_terms: dict[int, set[str]] = defaultdict(set)
        corpus_size = len(self.chunks)
        for term, query_frequency in query_terms.items():
            postings = self.postings.get(term, [])
            document_frequency = len(postings)
            if not document_frequency:
                continue
            inverse_document_frequency = math.log(
                1
                + (corpus_size - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for index, term_frequency in postings:
                matched_terms[index].add(term)
                length = self.document_lengths[index]
                normalization = term_frequency + self.k1 * (
                    1
                    - self.b
                    + self.b * length / self.average_document_length
                )
                scores[index] += (
                    query_frequency
                    * inverse_document_frequency
                    * term_frequency
                    * (self.k1 + 1)
                    / normalization
                )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            LexicalMatch(
                chunk=self.chunks[index],
                score=score,
                matched_terms=len(matched_terms[index]),
                query_terms=len(query_terms),
            )
            for index, score in ranked[:limit]
        ]
