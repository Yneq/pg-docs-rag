"""Reusable retrieval and grounded-generation service."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import re
from threading import Lock
import time
from typing import Any, Callable

from app.inference import (
    GenerationResult,
    InferenceBackend,
    InferenceConfig,
    create_inference_backend,
)
from app.retrieval import Bm25Index, LexicalChunk


REFUSAL_MESSAGE = (
    "I could not find relevant information in the PostgreSQL documentation."
)

QUERY_TERM_EXPANSIONS = {
    "MVCC": "multiversion concurrency control",
    "WAL": "write-ahead logging",
    "CTE": "common table expression",
    "PITR": "point-in-time recovery",
    "HOT": "heap-only tuple",
}


def expand_postgresql_terms(question: str) -> str:
    """Add full forms for common PostgreSQL acronyms before embedding."""
    expanded = question
    lowered = question.lower()
    for acronym, full_form in QUERY_TERM_EXPANSIONS.items():
        if full_form in lowered:
            continue
        expanded = re.sub(
            rf"\b{re.escape(acronym)}\b",
            f"{acronym} ({full_form})",
            expanded,
            flags=re.IGNORECASE,
        )
    return expanded


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RagSettings:
    chroma_path: str = "./chroma"
    collection_name: str = "pg_docs"
    embedding_model: str = "nomic-embed-text"
    distance_threshold: float = 0.6
    hybrid_enabled: bool = True
    semantic_candidates: int = 50
    lexical_candidates: int = 20
    rrf_k: int = 60
    lexical_weight: float = 2.0
    lexical_guardrail_coverage: float = 0.8
    lexical_guardrail_min_terms: int = 2

    @classmethod
    def from_env(cls) -> "RagSettings":
        return cls(
            chroma_path=os.getenv("CHROMA_PATH", "./chroma"),
            collection_name=os.getenv("CHROMA_COLLECTION", "pg_docs"),
            embedding_model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
            ),
            distance_threshold=float(
                os.getenv("RAG_DISTANCE_THRESHOLD", "0.6")
            ),
            hybrid_enabled=_env_bool("RAG_HYBRID_ENABLED", True),
            semantic_candidates=int(os.getenv("RAG_SEMANTIC_CANDIDATES", "50")),
            lexical_candidates=int(os.getenv("RAG_LEXICAL_CANDIDATES", "20")),
            rrf_k=int(os.getenv("RAG_RRF_K", "60")),
            lexical_weight=float(os.getenv("RAG_LEXICAL_WEIGHT", "2.0")),
            lexical_guardrail_coverage=float(
                os.getenv("RAG_LEXICAL_GUARDRAIL_COVERAGE", "0.8")
            ),
            lexical_guardrail_min_terms=int(
                os.getenv("RAG_LEXICAL_GUARDRAIL_MIN_TERMS", "2")
            ),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    document: str
    distance: float
    source: str | None = None
    title: str | None = None
    chunk_index: int | None = None
    chunk_id: str | None = None
    lexical_score: float | None = None
    lexical_matched_terms: int = 0
    lexical_query_terms: int = 0
    lexical_coverage: float = 0.0
    fusion_score: float | None = None


@dataclass(frozen=True)
class RagResult:
    answer: str
    grounded: bool
    chunks: list[RetrievedChunk]
    retrieval_seconds: float
    generation_seconds: float
    generation: GenerationResult | None = None

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.generation_seconds


class RagService:
    """Coordinates retrieval and generation without depending on an HTTP layer."""

    def __init__(
        self,
        *,
        collection: Any,
        embed_query: Callable[[str], list[float]],
        inference: InferenceBackend,
        settings: RagSettings,
        backend_summary: str,
    ) -> None:
        self.collection = collection
        self.embed_query = embed_query
        self.inference = inference
        self.settings = settings
        self.backend_summary = backend_summary
        # Direct Transformers generation mutates cache/state and should not run
        # concurrently on a single model instance. This also limits GPU pressure.
        self._generation_lock = Lock()
        self._index_lock = Lock()
        self._lexical_index: Bm25Index | None = None

    def collection_count(self) -> int:
        return int(self.collection.count())

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        expanded_question = expand_postgresql_terms(question)
        query_embedding = self.embed_query(expanded_question)
        if not self.settings.hybrid_enabled:
            return self._semantic_search(query_embedding, top_k)
        return self._hybrid_search(expanded_question, query_embedding, top_k)

    def _semantic_search(
        self, query_embedding: list[float], limit: int
    ) -> list[RetrievedChunk]:
        collection_size = self.collection_count()
        if collection_size == 0:
            return []
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection_size),
            include=["documents", "distances", "metadatas"],
        )
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for index, (document, distance) in enumerate(zip(documents, distances)):
            metadata = metadatas[index] if index < len(metadatas) else None
            metadata = metadata or {}
            chunk_index = metadata.get("chunk_index")
            chunks.append(
                RetrievedChunk(
                    document=document,
                    distance=float(distance),
                    source=metadata.get("source"),
                    title=metadata.get("title"),
                    chunk_index=(
                        int(chunk_index) if chunk_index is not None else None
                    ),
                    chunk_id=ids[index] if index < len(ids) else None,
                )
            )
        return chunks

    def _get_lexical_index(self) -> Bm25Index:
        if self._lexical_index is not None:
            return self._lexical_index
        with self._index_lock:
            if self._lexical_index is not None:
                return self._lexical_index
            response = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            embeddings = response.get("embeddings")
            if embeddings is None:
                raise RuntimeError("Chroma did not return embeddings")
            chunks = []
            for chunk_id, document, metadata, embedding in zip(
                response["ids"],
                response["documents"],
                response["metadatas"],
                embeddings,
            ):
                chunks.append(
                    LexicalChunk(
                        chunk_id=chunk_id,
                        document=document,
                        metadata=metadata or {},
                        embedding=embedding,
                    )
                )
            self._lexical_index = Bm25Index(chunks)
            return self._lexical_index

    def _hybrid_search(
        self,
        question: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        semantic = self._semantic_search(
            query_embedding,
            max(top_k, self.settings.semantic_candidates),
        )
        lexical = self._get_lexical_index().search(
            question,
            self.settings.lexical_candidates,
        )
        candidates = {
            chunk.chunk_id: chunk for chunk in semantic if chunk.chunk_id is not None
        }
        fusion_scores: dict[str, float] = {}
        for rank, chunk in enumerate(semantic, start=1):
            if chunk.chunk_id is not None:
                fusion_scores[chunk.chunk_id] = 1 / (self.settings.rrf_k + rank)

        for rank, match in enumerate(lexical, start=1):
            record = match.chunk
            fusion_scores[record.chunk_id] = fusion_scores.get(
                record.chunk_id, 0.0
            ) + self.settings.lexical_weight / (self.settings.rrf_k + rank)
            existing = candidates.get(record.chunk_id)
            if existing is not None:
                candidates[record.chunk_id] = replace(
                    existing,
                    lexical_score=match.score,
                    lexical_matched_terms=match.matched_terms,
                    lexical_query_terms=match.query_terms,
                    lexical_coverage=match.coverage,
                )
                continue
            metadata = record.metadata
            chunk_index = metadata.get("chunk_index")
            distance = sum(
                (float(query_value) - float(document_value)) ** 2
                for query_value, document_value in zip(
                    query_embedding, record.embedding
                )
            )
            candidates[record.chunk_id] = RetrievedChunk(
                document=record.document,
                distance=distance,
                source=metadata.get("source"),
                title=metadata.get("title"),
                chunk_index=(
                    int(chunk_index) if chunk_index is not None else None
                ),
                chunk_id=record.chunk_id,
                lexical_score=match.score,
                lexical_matched_terms=match.matched_terms,
                lexical_query_terms=match.query_terms,
                lexical_coverage=match.coverage,
            )

        ranked = sorted(
            candidates.values(),
            key=lambda chunk: (
                fusion_scores.get(chunk.chunk_id or "", 0.0),
                -chunk.distance,
            ),
            reverse=True,
        )
        return [
            replace(
                chunk,
                fusion_score=fusion_scores.get(chunk.chunk_id or "", 0.0),
            )
            for chunk in ranked[:top_k]
        ]

    def is_relevant(self, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        best = chunks[0]
        semantic_match = best.distance <= self.settings.distance_threshold
        lexical_match = (
            best.lexical_matched_terms
            >= self.settings.lexical_guardrail_min_terms
            and best.lexical_coverage
            >= self.settings.lexical_guardrail_coverage
        )
        return semantic_match or lexical_match

    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GenerationResult:
        context_sections = []
        for rank, chunk in enumerate(chunks, start=1):
            label = chunk.title or chunk.source or "PostgreSQL documentation"
            context_sections.append(
                f"[Source {rank}] {label}\n{chunk.document}"
            )

        context = "\n\n".join(context_sections)
        prompt = (
            "You are a PostgreSQL expert. Answer the question using ONLY the "
            "provided context. Every factual answer must contain at least one "
            "citation using the exact format [Source N], for example [Source 1]. "
            "Do not put titles or other text inside the brackets. Include important "
            "safety caveats when they are relevant. If the context does not contain "
            "the answer, say that you could not find it in the PostgreSQL "
            "documentation.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\nAnswer:"
        )
        with self._generation_lock:
            return self.inference.generate_with_metrics(prompt)

    def query(self, question: str, top_k: int = 3) -> RagResult:
        retrieval_started = time.perf_counter()
        chunks = self.retrieve(question, top_k)
        retrieval_seconds = time.perf_counter() - retrieval_started

        if not self.is_relevant(chunks):
            return RagResult(
                answer=REFUSAL_MESSAGE,
                grounded=False,
                chunks=chunks,
                retrieval_seconds=retrieval_seconds,
                generation_seconds=0.0,
            )

        generation_started = time.perf_counter()
        generation = self.generate(question, chunks)
        generation_seconds = time.perf_counter() - generation_started
        return RagResult(
            answer=generation.text,
            grounded=True,
            chunks=chunks,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            generation=generation,
        )


def create_rag_service() -> RagService:
    """Build the production service lazily from environment configuration."""
    import chromadb
    from chromadb.config import Settings
    import ollama

    settings = RagSettings.from_env()
    inference_config = InferenceConfig.from_env()
    chroma = chromadb.Client(
        Settings(
            persist_directory=settings.chroma_path,
            is_persistent=True,
        )
    )
    collection = chroma.get_or_create_collection(settings.collection_name)

    def embed_query(question: str) -> list[float]:
        response = ollama.embed(
            model=settings.embedding_model,
            input=f"search_query: {question}",
        )
        return response["embeddings"][0]

    return RagService(
        collection=collection,
        embed_query=embed_query,
        inference=create_inference_backend(inference_config),
        settings=settings,
        backend_summary=inference_config.summary(),
    )
