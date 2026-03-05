pg-docs-rag

A local Retrieval-Augmented Generation (RAG) system built using PostgreSQL official documentation as the knowledge base.

This project demonstrates an end-to-end RAG pipeline running fully locally (no external APIs), focusing on system design clarity rather than UI complexity.

🎯 Project Goal

Build a minimal but production-style RAG system that:

Uses real technical documentation as knowledge base

Runs entirely locally (zero API cost)

Demonstrates vector search + LLM grounding

Is easy to explain in an interview

🧠 System Architecture
Document → Clean → Chunk → Embed → Store → Retrieve → Ground → Generate

Pipeline:

Document ingestion
↓
Chunking
↓
Embeddings (nomic-embed-text)
↓
Vector DB (Chroma)
↓
Similarity Search
↓
Guardrail
↓
LLM (llama3.2)
↓
Answer
1️⃣ Data Ingestion

Source: PostgreSQL official documentation

HTML cleaned using BeautifulSoup

Text chunked into fixed-size segments

Embeddings generated using nomic-embed-text via Ollama

Stored in ChromaDB (persistent vector store)

2️⃣ Retrieval

User query embedded using the same embedding model

Top-K similar chunks retrieved via vector similarity search

Similarity scores inspected for debugging

3️⃣ Grounded Generation

Retrieved chunks injected into prompt

LLM (llama3.2 via Ollama) generates answer

Prompt constrains model to use only retrieved context

4️⃣ CLI Chat Demo

Run the chatbot:

python scripts/chat.py

Example:

Ask a PostgreSQL question:

What does SELECT do in PostgreSQL?

Answer:

The SELECT statement retrieves rows from a table or view...

5️⃣ Guardrails

To reduce hallucinations, the system checks vector similarity before sending context to the LLM.

If the closest chunk distance exceeds a threshold, the system refuses to answer.

This prevents the model from generating responses unrelated to the PostgreSQL documentation.

🏗 Tech Stack
Component	Technology
Language	Python
Embedding	Ollama (nomic-embed-text)
LLM	Ollama (llama3.2)
Vector DB	ChromaDB (persistent mode)
Data Source	PostgreSQL Documentation

📂 Project Structure
```
pg-docs-rag/
├── data/
│   └── raw/                 # Raw PostgreSQL documentation
├── scripts/
│   ├── ingest_docs.py       # Build vector database from docs
│   ├── demo_rag.py          # Ask questions using RAG
│   └── chat.py              # CLI interactive chatbot
├── chroma/                  # Chroma vector store
├── requirements.txt
└── README.md
```

🚀 How It Works
Ingestion
python scripts/ingest_docs.py

Steps performed:

Parse HTML

Extract readable text

Chunk text

Generate embeddings

Persist into ChromaDB

Run Demo
python scripts/demo_rag.py

Example query:

What does SELECT do in PostgreSQL?

System flow:

Embed query
↓
Retrieve top matching chunks
↓
Inject context into prompt
↓
Generate grounded answer
🧩 Design Decisions

Why Local Models?

Avoid API cost

Remove external dependency

Easier to test and debug

Demonstrates understanding of model deployment

Why ChromaDB?

Lightweight

Simple persistent storage

Good for local RAG prototype

Why PostgreSQL Docs?

Technical, structured content

Suitable for semantic search

Easier to discuss in backend/system interviews

🛠 Engineering Considerations

Persistent vector storage

HTML cleaning before embedding

Chunk overlap strategy

Distance inspection for debugging

Context size control to reduce latency

Simple guardrail to reduce hallucination

📊 Current Status

~1500+ chunks indexed

Retrieval working with similarity scores

Fully local RAG pipeline

CLI demo functional

🔮 Potential Improvements

Similarity threshold guardrail

Metadata filtering (section-based search)

Hybrid search (keyword + vector)

FastAPI endpoint

Multi-document ingestion

Evaluation dataset

Token usage tracking

💡 What This Project Demonstrates

Understanding of RAG architecture

Embedding workflow

Vector search fundamentals

Prompt grounding strategy

Trade-offs between latency and context size

Local LLM deployment workflow

📝 Notes

Fully local

CPU inference (slower on lightweight machines)

Designed for clarity and interview discussion rather than production scale


專案目標

建立一個 最小可行但具 production 思維的 RAG 系統：

使用真實技術文件作為知識庫

完全本地執行（不依賴 API）

展示向量檢索 + LLM grounding

🧠 系統架構
文件 → 清理 → Chunk → Embedding → 儲存 → 檢索 → Ground → 生成回答

流程圖：

Document ingestion
↓
Chunking
↓
Embeddings (nomic-embed-text)
↓
Vector DB (Chroma)
↓
Similarity Search
↓
Guardrail (距離檢查)
↓
LLM (llama3.2)
↓
Answer

1️⃣ 文件收集與處理（Data Ingestion）

資料來源：PostgreSQL 官方文件

HTML 清理：使用 BeautifulSoup 去除標籤

Chunk 分段：固定長度 + overlap

Embedding：使用 nomic-embed-text via Ollama

儲存：ChromaDB（持久化向量資料庫）

2️⃣ 檢索（Retrieval）

User query 也會產生 embedding

Top-K 相似 chunks 透過向量相似度搜尋

顯示距離方便調試與 guardrail

3️⃣ Grounded Generation

將檢索到的 chunks 注入 prompt

使用 LLM (llama3.2 via Ollama) 生成回答

Prompt 強制模型只能使用 context 內容

4️⃣ CLI 互動 Demo

執行：
```bash
python scripts/chat.py
```
範例互動：

Ask a PostgreSQL question:
> What does SELECT do in PostgreSQL?

Answer:
The SELECT statement retrieves rows from a table or view...

5️⃣ Guardrail（防止亂回答）

檢查向量距離（distance）

若距離超過 threshold → 拒答

避免 LLM 回答與文件無關的問題
```bash
if best_distance > 250:
    print("I could not find relevant information in the PostgreSQL documentation.")
    continue
```

🏗 技術堆疊（Tech Stack）
元件	技術
語言	Python
Embedding	Ollama (nomic-embed-text)
LLM	Ollama (llama3.2)
Vector DB	ChromaDB (持久化模式)
資料來源	PostgreSQL Documentation

📂 專案目錄
```bash
pg-docs-rag/
├── data/
│   └── raw/                 # 原始 PostgreSQL 文件
├── scripts/
│   ├── ingest_docs.py       # 建立向量資料庫
│   ├── demo_rag.py          # 單次問題測試
│   └── chat.py              # CLI 互動問答
├── chroma/                  # Chroma 向量資料庫
├── requirements.txt
└── README.md
```

🚀 使用說明
1️⃣ 建立向量資料庫
```bash
python scripts/ingest_docs.py
```
步驟：

解析 HTML

提取可讀文字

分段 chunk

生成 embeddings

儲存到 ChromaDB

2️⃣ 問問題 Demo
```bash
python scripts/demo_rag.py
```
範例：

What does SELECT do in PostgreSQL?

流程：

Embed query
↓
檢索 top-K 相關 chunks
↓
注入 context 到 prompt
↓
LLM 生成回答

3️⃣ CLI 互動問答
```bash
python scripts/chat.py
```
即時問問題

顯示距離、答案

Guardrail 自動拒答不相關問題

🧩 設計決策

為什麼使用本地模型？

無 API 成本

減少外部依賴

測試、除錯方便

展示部署概念

為什麼選 ChromaDB？

輕量

支援持久化

適合本地 RAG 原型

為什麼用 PostgreSQL 文件？

技術性強、結構化

適合語意檢索

面試講解方便

🛠 工程細節

持久化向量存儲

HTML 清理後再嵌入向量

Chunk overlap 策略

檢索距離可檢查

控制上下文大小降低延遲

Guardrail 減少 hallucination

📊 專案現狀

約 1500+ chunks 已索引

檢索功能正常，可顯示距離

完全本地 RAG pipeline

CLI 互動 Demo 可用

🔮 潛在升級

Guardrail 閾值微調

Metadata 篩選（section-based search）

Hybrid search（關鍵字 + vector）

FastAPI API endpoint

多文件 ingestion

評估數據集

Token usage 追蹤