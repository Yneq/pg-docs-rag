"""FastAPI entry point for the PostgreSQL documentation RAG service."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.rag import RagService, create_rag_service


app = FastAPI(
    title="PostgreSQL Docs RAG API",
    description="A local, source-grounded PostgreSQL documentation assistant.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceResponse(BaseModel):
    rank: int
    title: str | None
    source: str | None
    chunk_index: int | None
    distance: float


class MetricsResponse(BaseModel):
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    prompt_tokens: int | None
    generated_tokens: int | None
    tokens_per_second: float | None


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    backend: str
    sources: list[SourceResponse]
    metrics: MetricsResponse


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    backend: str
    collection: str
    indexed_chunks: int


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return create_rag_service()


@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health(
    service: RagService = Depends(get_rag_service),
) -> HealthResponse:
    try:
        indexed_chunks = await run_in_threadpool(service.collection_count)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The vector store is unavailable.",
        ) from exc

    return HealthResponse(
        status="ok" if indexed_chunks > 0 else "degraded",
        backend=service.backend_summary,
        collection=service.settings.collection_name,
        indexed_chunks=indexed_chunks,
    )


@app.post("/api/query", response_model=QueryResponse, tags=["rag"])
async def query(
    request: QueryRequest,
    service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    try:
        result = await run_in_threadpool(
            service.query, request.question, request.top_k
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The local inference service is unavailable.",
        ) from exc

    generation = result.generation
    generated_tokens = generation.generated_tokens if generation else None
    prompt_tokens = generation.prompt_tokens if generation else None
    tokens_per_second = (
        generated_tokens / result.generation_seconds
        if generated_tokens is not None and result.generation_seconds > 0
        else None
    )
    return QueryResponse(
        answer=result.answer,
        grounded=result.grounded,
        backend=service.backend_summary,
        sources=[
            SourceResponse(
                rank=rank,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                distance=round(chunk.distance, 6),
            )
            for rank, chunk in enumerate(result.chunks, start=1)
        ],
        metrics=MetricsResponse(
            retrieval_seconds=round(result.retrieval_seconds, 3),
            generation_seconds=round(result.generation_seconds, 3),
            total_seconds=round(result.total_seconds, 3),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            tokens_per_second=(
                round(tokens_per_second, 3)
                if tokens_per_second is not None
                else None
            ),
        ),
    )
