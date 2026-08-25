[English](README.md) | [繁體中文](README.zh-TW.md)

# pg-docs-rag

[![CI](https://github.com/Yneq/pg-docs-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Yneq/pg-docs-rag/actions/workflows/ci.yml)

這是一套完全在本機執行的檢索增強生成（Retrieval-Augmented Generation，RAG）系統，以 PostgreSQL 官方文件作為知識庫。專案重點是呈現清楚、適合面試說明的處理流程，而非複雜的 UI 或外部 API。

## 系統架構

```text
PostgreSQL 文件 → 清理 → 分塊 → 加前綴的 Ollama embeddings → ChromaDB
                                                              ↓
問題 → Ollama embedding + BM25 → RRF 融合 → guardrail → grounded prompt
                                                              ↓
                                      Ollama 或 Transformers 生成
                                                              ↓
                                          CLI 或 FastAPI JSON 回應
```

Ollama 是預設後端，且一律負責 `nomic-embed-text` embeddings。文字生成則可選擇使用 Hugging Face Transformers、PyTorch 與 CUDA。只要索引是用目前的 embedding pipeline 建立，同一份 Chroma 索引就能搭配任一生成後端，切換生成器時不需要重新匯入資料。

## 本機推論環境（Local Inference Environment）

以下是執行本專案時使用的部署／效能環境。這段資訊記錄實際測試過的本機配置，並不代表本專案另有一個獨立的「GPU 版本」。

| 元件 | 環境 |
|---|---|
| CPU | Intel Core i5-12400 |
| RAM | 32 GB |
| GPU | NVIDIA GeForce GTX 1660 Ti 6 GB |
| Ollama | 0.32.14 |
| 模型 | `llama3.2:latest` |
| GPU offload | 經 `ollama ps` 驗證為 `100% GPU` |
| Context | 4096 tokens |

### Mac 執行驗證

完整 API pipeline 也已在 Apple M1 Mac 上完成驗證；環境為 16 GB unified memory、Python 3.12.3、Ollama 0.17.5，以及同一份包含 4,871 個 chunks 的 PostgreSQL 18.4 corpus。對 `How does PostgreSQL MVCC work?` 發送 warm `POST /api/query` 請求後得到：

| Retrieval | Generation | Total | Generated tokens | Tokens/s |
|---:|---:|---:|---:|---:|
| 0.039 s | 12.051 s | 12.090 s | 198 | 16.430 |

回答依據 `mvcc-intro.html`、`routine-vacuuming.html` 與 `different-replication-solutions.html`。一個與 PostgreSQL 無關的巧克力蛋糕問題則在 0.128 秒內被拒答，且沒有呼叫生成模型。這些 Mac 數據用於驗證 HTTP pipeline 與 guardrail；因為硬體及生成 token 數不同，不能直接與下方 CUDA benchmark 比較。

## 實測效能

以下是在上述環境於 2026-08-18 測得的 warm-run 結果；兩者使用相同問題（`How does PostgreSQL MVCC work?`）、相同的三個 retrieved chunks，以及包含 4,871 個 chunks 的 PostgreSQL 18.4 索引：

| 生成後端 | Retrieval | Generation | Total | Generated tokens | Tokens/s |
|---|---:|---:|---:|---:|---:|
| Ollama — `llama3.2:latest` | 0.133 s | 1.150 s | 1.283 s | 84 | 73.060 |
| Transformers — Qwen2.5 3B, 4-bit | 1.273 s | 18.363 s | 19.636 s | 136 | 7.406 |

這次測試中，Ollama 的生成 throughput 約高出 9.9 倍。Transformers 的回答多生成了約 62% tokens，因此只看總耗時會放大底層 throughput 的差距。其 GPU 快照在高使用率時仍維持約 3.2 GiB / 6 GiB，證明這個 4-bit 模型可以放進目標顯示卡，但通用的 PyTorch／bitsandbytes 路徑速度較慢。

兩種設定的 retrieval 都使用相同的 Ollama embedding 後端，因此 retrieval 時間差異來自 runtime／cache 波動，而不是生成後端的優勢。人工檢查發現 Transformers 的回答與 retrieved MVCC context 更一致，但仍需要可重複執行的題目集與 groundedness 評分，才能對品質下結論。

這些後端數據是在導入正規化、加上 task prefix 的 embeddings 之前取得。由於該次測試中兩個後端使用相同的 retrieved chunks，數據仍可作為生成 throughput 的比較；若要發布新的端到端數據，應重新執行 ingestion 與兩個 benchmark。

## 安裝設定

建議使用 Python 3.12。預設 Ollama 後端可在 Apple Silicon 執行；本專案的可選直接 Transformers 後端目前需要 NVIDIA CUDA，無法在 macOS 啟用。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2
```

下載版本固定為 PostgreSQL 18.4 的官方 HTML 文件壓縮檔，再建立本機向量資料庫。下載程式只會解壓 HTML 文件，並在 manifest 中記錄來源 URL：

```bash
python scripts/download_docs.py
ollama pull nomic-embed-text
python scripts/ingest_docs.py --reset
```

Ingestion 可能需要一些時間，因為每個 chunk 的 embedding 都在本機生成；執行期間會顯示進度與最終 collection 數量。確定性的 chunk ID 與 Chroma `upsert` 讓未使用 reset 的重跑保持安全；只有在刻意重建整個 `pg_docs` collection 時才使用 `--reset`。

## 執行 FastAPI 服務

API 與 CLI 共用同一套 retrieval、guardrail、prompt 與 inference 實作。建立索引後，在 repository root 執行：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

開啟 `http://127.0.0.1:8000/docs` 可使用互動式 Swagger 文件。服務提供：

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/health` | 回報後端設定與已建立索引的 chunk 數量 |
| `POST` | `/api/query` | 檢索來源、套用 guardrail 並生成回答 |

請求範例：

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does PostgreSQL MVCC work?","top_k":3}'
```

回應包含 `grounded`、retrieved source metadata、retrieval 與 generation latency、token 數，以及 generation tokens/second。同步的 Chroma 與本機模型工作會在 FastAPI event loop 之外執行。每個 process 會循序執行 generation，避免同時存取同一個 Transformers 模型，並避免對 6 GB GPU 造成過大壓力。

這個本機 demo 不包含驗證或 rate limiting。除非前方已設置具驗證機制的 reverse proxy，否則請綁定至 `127.0.0.1`。

不需要 Ollama、Chroma 資料或 GPU，即可執行離線 unit tests 與 API contract tests：

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

## Retrieval 品質評估

納入版本控制的評估集包含 15 個 PostgreSQL 問題與五個 out-of-domain 問題。它會測量前三筆結果的 source hit rate、mean reciprocal rank（MRR）與 guardrail accuracy，而且不需要花時間生成答案：

```bash
python scripts/evaluate_retrieval.py \
  --output results/retrieval-eval.json
```

Baseline 特別包含較困難的 CTE、`EXPLAIN ANALYZE` 與 HOT 問題，而不是只挑選已知會成功的題目。若 retrieval hit@3 低於 75%，或 guardrail accuracy 低於 90%，評估會以非零 exit code 結束。GitHub Actions 會在每次 push 與 pull request 驗證 dataset 並執行所有 offline tests；真正的 retrieval evaluation 仍保留在本機執行，因為它需要 Ollama 與包含 4,871 個 chunks 的 Chroma 索引。

目前 Mac 測試結果顯示，hybrid retrieval 在相同固定 dataset 與索引上的效果如下：

| Retrieval strategy | Hit@3 | MRR@3 | Guardrail accuracy |
|---|---:|---:|---:|
| Semantic only | 80.0% (12/15) | 0.767 | 95.0% (19/20) |
| Semantic + BM25 + RRF | 100.0% (15/15) | 0.922 | 100.0% (20/20) |

Hybrid run 找回了全部三個 challenge cases，同時沒有失去任何一個 out-of-domain refusal。這些數據是此專案的小型 regression baseline，不代表通用的模型品質 benchmark。

## 回答與引用評估

Retrieval 成功不代表生成的回答一定完整，也不代表引用一定可用。因此第二份納入版本控制的 dataset 會執行完整 RAG pipeline，並檢查三項確定性質：

- 回答中必須出現預期的 PostgreSQL concepts；
- 每個回答至少包含一個有效的 `[Source N]` citation，且引用必須指向該次請求所回傳的 source；
- out-of-domain 問題必須在呼叫生成模型之前被拒答。

```bash
python scripts/evaluate_answers.py \
  --output results/answer-eval-ollama.json
```

目前在 Mac 環境使用 Ollama 的結果：

| Concept coverage | Citation accuracy | Guardrail accuracy | Result |
|---:|---:|---:|---|
| 100.0% (15/15) | 100.0% (5/5) | 100.0% (7/7) | PASS |

第一次執行時發現了一個真實的 prompt contract 缺陷：回答使用 `[Source 1: title]`，而不是文件規定的 `[Source 1]` 格式，因此有效 citation 為 0/5。在 context 中將 source number 與 title 分開，並明確規定精確的 citation contract 後，同一份測試集提升至 5/5。

這項評估刻意定位為 regression smoke test，而不是宣稱回答品質完美。Keyword groups 可以驗證必要 concepts 與 source numbers，卻無法證明每一句話都有事實依據。在將系統用於較高風險的結論前，仍適合加入人工審查或經獨立校正的 judge。

### 使用 Docker 執行 API

讓 Ollama 繼續在 Mac host 執行，並啟動 API container：

```bash
docker compose up --build
```

Compose 設定會掛載現有的 `./chroma` 索引，並透過 `host.docker.internal` 將 container 連接到 host 上的 Ollama。這條路徑使用預設 Ollama generation backend。直接使用 CUDA 的 Transformers 選項需要 NVIDIA 環境，無法在這台 Apple Silicon host 上使用。

## 使用 Ollama 執行（預設）

不需要設定後端：

```bash
python scripts/chat.py
```

也提供雙語 demo：

```bash
python scripts/demo_rag.py
```

若要選擇另一個由 Ollama 提供服務的模型：

```bash
export OLLAMA_LLM_MODEL="llama3.2:latest"
python scripts/chat.py
```

## 使用 Transformers + CUDA 執行（可選）

請依照已安裝的 NVIDIA driver／CUDA 環境，使用對應的 PyTorch build 安裝可選依賴。如果已安裝支援 CUDA 的 PyTorch build，執行：

```bash
pip install -r requirements-transformers.txt
```

預設的直接 Transformers 模型是 `Qwen/Qwen2.5-3B-Instruct`，使用 bitsandbytes 4-bit NF4 quantization 載入，以便更穩定地容納於 6 GB VRAM。第一次執行會從 Hugging Face 下載模型檔案。

```bash
export LLM_BACKEND="transformers"
export HF_MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
export HF_LOAD_IN_4BIT="true"
python scripts/check_inference.py
python scripts/chat.py
```

設定變數：

| 變數 | 預設值 | 用途 |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` 或 `transformers` |
| `OLLAMA_LLM_MODEL` | `llama3.2:latest` | Ollama 生成模型 |
| `HF_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face model ID 或本機模型路徑 |
| `HF_LOAD_IN_4BIT` | `true` | 啟用 bitsandbytes 4-bit 載入 |
| `HF_MAX_NEW_TOKENS` | `512` | 每次請求最多生成的 tokens |
| `HF_TEMPERATURE` | `0.0` | `0` 代表 deterministic generation；正值會啟用 sampling |
| `CHROMA_PATH` | `./chroma` | 持久化 Chroma 目錄 |
| `CHROMA_COLLECTION` | `pg_docs` | Chroma collection 名稱 |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding 模型 |
| `RAG_DISTANCE_THRESHOLD` | `0.6` | 可接受的最大 normalized L2 distance |
| `RAG_HYBRID_ENABLED` | `true` | 結合 semantic 與 BM25 retrieval |
| `RAG_SEMANTIC_CANDIDATES` | `50` | RRF 納入考量的 semantic candidates 數量 |
| `RAG_LEXICAL_CANDIDATES` | `20` | RRF 納入考量的 BM25 candidates 數量 |
| `RAG_RRF_K` | `60` | Reciprocal Rank Fusion smoothing constant |
| `RAG_LEXICAL_WEIGHT` | `2.0` | BM25 相對於 semantic rank 的貢獻權重 |
| `RAG_LEXICAL_GUARDRAIL_COVERAGE` | `0.8` | Lexical guardrail evidence 所需的最低 query-term coverage |
| `RAG_LEXICAL_GUARDRAIL_MIN_TERMS` | `2` | Lexical guardrail evidence 所需的最少 matched terms |

`scripts/check_inference.py` 不會下載或載入模型，只驗證後端選擇。實際的 chat command 會驗證 CUDA，並在第一次 generation request 時載入模型。若要嘗試完整 FP16，請設定 `HF_LOAD_IN_4BIT=false`，但 3B 模型加上 runtime／KV-cache overhead 可能超過 6 GB 顯示卡容量；建議從 4-bit 開始。

## Ollama 與 Transformers 比較

| 面向 | Ollama | Transformers + PyTorch |
|---|---|---|
| 專案預設 | 是 | 可選 |
| 模型生命週期 | Ollama 管理載入與 serving | Python process 直接載入模型 |
| 安裝 | 較簡單 | 需要 CUDA／PyTorch 與模型依賴 |
| Quantization | 由 Ollama model package 管理 | 明確設定 bitsandbytes 4-bit |
| 控制能力 | 便利的 high-level API | 直接控制 tokenizer、dtype、device 與 generation |
| 本專案的 embeddings | `nomic-embed-text` | 仍使用 Ollama，以保留既有索引 |
| 在本專案最適合的用途 | 穩定的日常本機 RAG | 學習、實驗與後端比較 |

## 比較兩個後端的 Benchmark

一次只執行一個後端，避免兩個生成模型爭用 GTX 1660 Ti 的 6 GB VRAM。兩次執行必須使用完全相同的問題。

```bash
# 1. Ollama
export LLM_BACKEND="ollama"
python scripts/benchmark_rag.py \
  --question "How does PostgreSQL MVCC work?" \
  --warmup \
  --output results/ollama.json

# 載入 Transformers 前，先釋放 Ollama 生成模型。
ollama stop llama3.2

# 2. Transformers
export LLM_BACKEND="transformers"
export HF_LOAD_IN_4BIT="true"
python scripts/benchmark_rag.py \
  --question "How does PostgreSQL MVCC work?" \
  --warmup \
  --output results/transformers.json

# 3. 印出比較表格
python scripts/compare_benchmarks.py \
  results/ollama.json results/transformers.json
```

每個結果都會記錄 retrieval、generation 與 total wall-clock time、prompt 與 generated token 數、generation tokens/second、回答、retrieval distances、`ollama ps`，以及 NVIDIA GPU memory／utilization 快照。Ollama 使用其 API 的 `eval_count`；Transformers 則直接計算生成的 token IDs。使用 `--warmup` 時，計時前會先執行一次不計入測量的 generation 來載入模型；若要測量 cold-start cost，請移除此 flag。Transformers 第一次執行也可能下載模型檔案，這段時間應與 inference latency 分開回報。

在加入 token metrics 前建立的既有 benchmark JSON 仍可讀取，相關欄位會顯示 `N/A`。若要公平比較 tokens/second，請重新執行兩個後端。

## 專案結構

```text
pg-docs-rag/
├── app/
│   ├── documents.py             # PostgreSQL 文件解析／分塊
│   ├── inference.py             # Ollama／Transformers 後端抽象層
│   ├── rag.py                   # 共用 retrieval／guardrail／generation service
│   └── main.py                  # FastAPI endpoints 與 Pydantic contracts
├── data/raw/                    # 本機 PostgreSQL 文件（不提交）
├── scripts/
│   ├── check_inference.py       # 輕量後端設定檢查
│   ├── benchmark_rag.py         # 測量單一後端並擷取 GPU diagnostics
│   ├── compare_benchmarks.py    # 比較已儲存的 benchmark runs
│   ├── download_docs.py         # 取得固定版本的 PostgreSQL 官方 HTML 文件
│   ├── ingest_docs.py
│   ├── demo_rag.py
│   └── chat.py
├── chroma/                      # 持久化本機 vector store（不提交）
├── evals/                       # 納入版本控制的 retrieval／guardrail 題目集
├── tests/                       # 離線核心與 API contract tests
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── requirements-dev.txt
├── requirements-test.txt
└── requirements-transformers.txt
```

## Guardrail 與設計說明

共用的 RAG service 使用加權 Reciprocal Rank Fusion（RRF），結合 Chroma semantic search 與 in-memory BM25 inverted index。生成前，guardrail 會接受足夠接近的 normalized vector match，或至少涵蓋兩個 query terms 的強 lexical evidence。同時要求 lexical coverage 與最低 term count，可避免單一偶然相符的詞繞過 distance threshold。Retrieved chunks 會在 prompt 中加上標記；prompt 要求模型只能根據提供的 PostgreSQL context 回答，並以 `[Source N]` 格式引用。API 會回傳每個 source 的 title、file、chunk index、distance、lexical score、query-term coverage 與 fusion score，讓 client 可以檢查其排序原因。

`nomic-embed-text` 會收到其要求的 retrieval task prefixes：ingestion 時使用 `search_document:`，retrieval 時使用 `search_query:`。目前 Ollama `/api/embed` endpoint 會回傳 normalized vectors，使 distance threshold 保持穩定且容易解讀。MVCC、WAL、CTE、PITR 與 HOT 等常見 PostgreSQL 縮寫會在 query embedding 前展開，使簡短技術問題也能檢索到包含完整名稱的文件。使用 API 前，請透過 `python scripts/ingest_docs.py --reset` 重建由舊版專案建立的索引。

本專案仍是一個本機 RAG prototype：它展示文件清理、分塊、embeddings、vector retrieval、grounded generation、可選本機 inference、HTTP contract，以及封裝式 runtime 與直接模型控制之間的部署取捨。
