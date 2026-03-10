import ollama
import chromadb
from chromadb.config import Settings

# 初始化 ChromaDB
chroma = chromadb.Client(Settings(
    persist_directory="./chroma",
    is_persistent=True
))

collection = chroma.get_or_create_collection("pg_docs")


def translate_to_english(query):
    response = ollama.generate(
        model="llama3.2",
        prompt=f"Translate this PostgreSQL question to clear, formal English, keep technical terms: {query}",
    )
    return response["response"].strip()


def translate_to_chinese(text):
    response = ollama.generate(
        model="llama3.2",
        prompt=f"請將下列 PostgreSQL 技術回答翻譯成繁體中文（台灣用語），保持技術術語正確，並使用正式、易讀的文字：\n\n{text}",
    )
    return response["response"].strip()


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

    response = ollama.generate(
        model="llama3.2",
        prompt=prompt
    )

    return response["response"]


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