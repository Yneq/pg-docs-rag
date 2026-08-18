from pathlib import Path
import sys

# Keep `python scripts/chat.py` working while importing project modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import create_rag_service


def chat():

    service = create_rag_service()

    print("\nPostgreSQL Docs RAG Chat")
    print("Type 'exit' to quit\n")

    while True:

        question = input("Ask a PostgreSQL question: ")

        if question.lower() in ["exit", "quit"]:
            break

        result = service.query(question)

        print("\nRetrieved Chunks Distance:")
        print([chunk.distance for chunk in result.chunks])

        print("\nAnswer:\n")
        print(result.answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    chat()
