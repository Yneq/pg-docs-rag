from pathlib import Path
import sys

# Keep `python scripts/chat.py` working while importing project modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama
import chromadb
from chromadb.config import Settings
from app.inference import GenerationResult, create_inference_backend

# 連接 Chroma
chroma = chromadb.Client(Settings(
    persist_directory="./chroma",
    is_persistent=True
))

collection = chroma.get_or_create_collection("pg_docs")
inference = create_inference_backend()


def retrieve(query, k=3):
    # query embedding
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )

    query_embedding = response["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances"]
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    return docs, distances


def generate_answer(question, context_chunks):

    return generate_answer_with_metrics(question, context_chunks).text


def generate_answer_with_metrics(question, context_chunks) -> GenerationResult:

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a PostgreSQL expert.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    return inference.generate_with_metrics(prompt)


def chat():

    print("\nPostgreSQL Docs RAG Chat")
    print("Type 'exit' to quit\n")

    while True:

        question = input("Ask a PostgreSQL question: ")

        if question.lower() in ["exit", "quit"]:
            break

        docs, distances = retrieve(question)

        print("\nRetrieved Chunks Distance:")
        print(distances)

        best_distance = distances[0]

        # Guardrail
        if best_distance > 250:
            print("\nAnswer:\n")
            print("I could not find relevant information in the PostgreSQL documentation.")
            print("\n" + "-" * 60 + "\n")
            continue

        answer = generate_answer(question, docs)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    chat()
