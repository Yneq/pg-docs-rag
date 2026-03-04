import ollama
import chromadb
from chromadb.config import Settings

chroma = chromadb.Client(Settings(
    persist_directory="./chroma",
    is_persistent=True
))

collection = chroma.get_or_create_collection("pg_docs")


def retrieve(query, k=1):
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

    # 印相似度看一下 demo
    print("Distances:", results["distances"])

    return results["documents"][0]


def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)[:1500]

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": 150  # 限制輸出 token
        }
    )

    return response["message"]["content"]


if __name__ == "__main__":
    question = "What does SELECT do in PostgreSQL?"
    
    docs = retrieve(question)
    answer = generate_answer(question, docs)

    print("\n====== FINAL ANSWER ======\n")
    print(answer)
    print("Collection count:", collection.count())