"""Build the persistent Chroma index from PostgreSQL documentation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.documents import chunk_document, parse_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/postgresql-18.4"),
        help="An HTML/text file or directory of documentation files.",
    )
    parser.add_argument("--collection", default="pg_docs")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the selected Chroma collection.",
    )
    return parser.parse_args()


def source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Documentation input not found: {source}")
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".htm", ".txt"}
    )


def stable_chunk_id(relative_source: str, chunk_index: int, text: str) -> str:
    value = f"{relative_source}\0{chunk_index}\0{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    import chromadb
    from chromadb.config import Settings
    import ollama

    files = source_files(args.input)
    if not files:
        raise RuntimeError(f"No HTML or text documents found under {args.input}")

    chroma = chromadb.Client(
        Settings(persist_directory="./chroma", is_persistent=True)
    )
    if args.reset:
        existing_names = {
            item if isinstance(item, str) else item.name
            for item in chroma.list_collections()
        }
        if args.collection in existing_names:
            chroma.delete_collection(args.collection)
    collection = chroma.get_or_create_collection(args.collection)

    pending_ids: list[str] = []
    pending_documents: list[str] = []
    pending_metadata: list[dict[str, str | int]] = []
    indexed = 0

    def flush() -> None:
        nonlocal indexed
        if not pending_ids:
            return
        response = ollama.embed(
            model=args.embedding_model,
            input=[f"search_document: {text}" for text in pending_documents],
        )
        collection.upsert(
            ids=pending_ids,
            documents=pending_documents,
            embeddings=response["embeddings"],
            metadatas=pending_metadata,
        )
        indexed += len(pending_ids)
        pending_ids.clear()
        pending_documents.clear()
        pending_metadata.clear()
        print(f"Indexed {indexed} chunks...", flush=True)

    print(f"Found {len(files)} source files. Building embeddings with {args.embedding_model}...")
    for file_number, path in enumerate(files, start=1):
        document = parse_document(path)
        chunks = chunk_document(document, args.chunk_size, args.overlap)
        relative_source = str(path.relative_to(args.input) if args.input.is_dir() else path.name)

        for chunk_index, chunk in enumerate(chunks):
            pending_ids.append(stable_chunk_id(relative_source, chunk_index, chunk))
            pending_documents.append(chunk)
            pending_metadata.append(
                {
                    "source": relative_source,
                    "title": document.title,
                    "chunk_index": chunk_index,
                }
            )
            if len(pending_ids) >= args.batch_size:
                flush()

        print(f"Processed {file_number}/{len(files)}: {relative_source}", flush=True)

    flush()
    print(f"Done. Collection '{args.collection}' contains {collection.count()} chunks.")


if __name__ == "__main__":
    main()
