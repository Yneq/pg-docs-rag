from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import create_rag_service


if __name__ == "__main__":
    question = "What does SELECT do in PostgreSQL?"
    chunks = create_rag_service().retrieve(question)
    print([chunk.distance for chunk in chunks])

    for i, chunk in enumerate(chunks):
        print(f"\n--- Result {i+1} ---\n")
        print(chunk.document[:500])
