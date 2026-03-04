import ollama
import chromadb
from chromadb.config import Settings

chroma = chromadb.Client(Settings(
    persist_directory="./chroma",
    is_persistent=True
))

collection = chroma.get_or_create_collection("pg_docs")

def retrieve(query, k=3):
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
    
    print(results["distances"])
    return results["documents"][0]


if __name__ == "__main__":
    question = "What does SELECT do in PostgreSQL?"
    docs = retrieve(question)

    for i, doc in enumerate(docs):
        print(f"\n--- Result {i+1} ---\n")
        print(doc[:500])