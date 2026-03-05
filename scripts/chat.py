import ollama
import chromadb
from chromadb.config import Settings

# 連接 Chroma
chroma = chromadb.Client(Settings(
    persist_directory="./chroma",
    is_persistent=True
))

collection = chroma.get_or_create_collection("pg_docs")


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

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a PostgreSQL expert.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


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