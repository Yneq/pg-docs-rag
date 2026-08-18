from pathlib import Path
import sys

# Keep `python scripts/demo_rag.py` working while importing project modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama
import chromadb
from chromadb.config import Settings
from app.inference import create_inference_backend

# 初始化 ChromaDB
chroma = chromadb.Client(Settings(
    persist_directory="./chroma",
    is_persistent=True
))

collection = chroma.get_or_create_collection("pg_docs")
inference = create_inference_backend()


def translate_to_english(query):
    return inference.generate(
        f"Translate this PostgreSQL question to clear, formal English, keep technical terms: {query}"
    )


def translate_to_chinese(text):
    return inference.generate(
        f"請將下列 PostgreSQL 技術回答翻譯成繁體中文（台灣用語），保持技術術語正確，並使用正式、易讀的文字：\n\n{text}"
    )


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

    docs = results["documents"][0]
    distances = results["distances"][0]

    print("Distances:", distances)

    return docs


def generate_answer(question, docs):

    context = "\n\n".join(docs)

    prompt = f"""
You are a PostgreSQL expert.

Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    return inference.generate(prompt)


if __name__ == "__main__":

    while True:

        question = input("\nAsk a PostgreSQL question (Chinese or English, q to quit): ")

        if question.lower() == "q":
            break

        # 中文 → 英文
        english_query = translate_to_english(question)
        print("\nTranslated query:", english_query)

        # retrieval
        docs = retrieve(english_query)

        # LLM answer
        answer_en = generate_answer(english_query, docs)

        # 翻譯回中文
        answer_zh = translate_to_chinese(answer_en)

        print("\n====== FINAL ANSWER ======\n")
        print(answer_zh)
