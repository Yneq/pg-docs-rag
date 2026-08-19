"""Reusable retrieval and grounded-generation service."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class RagSettings:
    chroma_path: str = "./chroma"
    collection_name: str = "pg_docs"
    embedding_model: str = "nomic-embed-text"
    distance_threshold: float = 0.6

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
        )


@dataclass(frozen=True)
class RetrievedChunk:
    document: str
    distance: float
    source: str | None = None
    title: str | None = None
    chunk_index: int | None = None


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

    def collection_count(self) -> int:
        return int(self.collection.count())

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        response = self.collection.query(
            query_embeddings=[self.embed_query(question)],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
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
                )
            )
        return chunks

    def is_relevant(self, chunks: list[RetrievedChunk]) -> bool:
        return bool(chunks) and chunks[0].distance <= self.settings.distance_threshold

    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GenerationResult:
        context_sections = []
        for rank, chunk in enumerate(chunks, start=1):
            label = chunk.title or chunk.source or "PostgreSQL documentation"
            context_sections.append(
                f"[Source {rank}: {label}]\n{chunk.document}"
            )

        context = "\n\n".join(context_sections)
        prompt = (
            "You are a PostgreSQL expert. Answer the question using ONLY the "
            "provided context. Cite supporting passages with [Source N]. If the "
            "context does not contain the answer, say that you could not find it "
            "in the PostgreSQL documentation.\n\n"
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
            input=f"search_query: {expand_postgresql_terms(question)}",
        )
        return response["embeddings"][0]

    return RagService(
        collection=collection,
        embed_query=embed_query,
        inference=create_inference_backend(inference_config),
        settings=settings,
        backend_summary=inference_config.summary(),
    )
