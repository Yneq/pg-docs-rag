"""Benchmark one configured RAG generation backend and save reproducible data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure retrieval and generation for the selected LLM_BACKEND."
    )
    parser.add_argument(
        "--question",
        default="How does PostgreSQL MVCC work?",
        help="Question sent through the RAG pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path, for example results/ollama.json.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one unmeasured generation first to exclude model startup cost.",
    )
    return parser.parse_args()


def _run_diagnostic(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output or None


def gpu_snapshot() -> str | None:
    return _run_diagnostic(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )


def main() -> None:
    args = parse_args()

    # Import project dependencies only after argument parsing so `--help` stays
    # usable before the optional runtime is installed.
    from app.inference import InferenceConfig
    from app.rag import create_rag_service

    config = InferenceConfig.from_env()
    service = create_rag_service()
    started_at = datetime.now(timezone.utc).isoformat()
    gpu_before = gpu_snapshot()

    retrieval_start = time.perf_counter()
    chunks = service.retrieve(args.question)
    retrieval_seconds = time.perf_counter() - retrieval_start

    if args.warmup:
        print("Warming up generation backend...", file=sys.stderr)
        service.generate(args.question, chunks)

    generation_start = time.perf_counter()
    generation = service.generate(args.question, chunks)
    generation_seconds = time.perf_counter() - generation_start
    total_seconds = retrieval_seconds + generation_seconds
    tokens_per_second = (
        generation.generated_tokens / generation_seconds
        if generation.generated_tokens is not None and generation_seconds > 0
        else None
    )

    result = {
        "timestamp_utc": started_at,
        "platform": platform.platform(),
        "backend": config.backend,
        "configuration": config.summary(),
        "question": args.question,
        "warmup": args.warmup,
        "retrieval_seconds": round(retrieval_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "prompt_tokens": generation.prompt_tokens,
        "generated_tokens": generation.generated_tokens,
        "tokens_per_second": (
            round(tokens_per_second, 3)
            if tokens_per_second is not None
            else None
        ),
        "distances": [chunk.distance for chunk in chunks],
        "sources": [
            {
                "title": chunk.title,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ],
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "ollama_ps": _run_diagnostic(["ollama", "ps"]),
        "answer": generation.text,
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nSaved benchmark: {output_path}")


if __name__ == "__main__":
    main()
