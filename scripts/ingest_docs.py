import os
import uuid
import ollama
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup

chroma = chromadb.Client(Settings(
    persist_directory="./chroma",
    is_persistent=True
))

collection = chroma.get_or_create_collection("pg_docs")


def extract_text_from_html(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # 只抓 body 文字
    body = soup.body
    text = body.get_text(separator="\n")

    return text


def chunk_by_sections(text):
    chunks = []
    current_chunk = []

    lines = text.split("\n")

    for line in lines:
        # 如果是大標題（全大寫或短句）
        if line.strip().isupper() and len(line.strip()) < 50:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []

        current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def ingest_file(filepath):
    text = extract_text_from_html(filepath)

    chunks = chunk_by_sections(text)
    print(f"共 {len(chunks)} 個 chunk，開始處理...")

    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        embedding = response["embedding"]

        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": filepath}]
        )

        if i % 10 == 0:
            print(f"進度：{i+1}/{len(chunks)}")

    print("✅ 完成！")


if __name__ == "__main__":
    ingest_file("data/raw/postgres_docs.txt")