# pg-docs-rag

A fully local Retrieval-Augmented Generation (RAG) system using the official
PostgreSQL documentation as its knowledge base. The project focuses on a clear,
interview-friendly pipeline rather than UI complexity or external APIs.

## Architecture

```text
PostgreSQL docs → clean → chunk → Ollama embeddings → ChromaDB
                                                        ↓
Question → Ollama embedding → retrieve → guardrail → grounded prompt
                                                        ↓
                                  Ollama or Transformers generation
```

Ollama remains the default and is always used for `nomic-embed-text`
embeddings. Text generation can optionally use Hugging Face Transformers with
PyTorch and CUDA. Existing Chroma data therefore works with either generation
backend and does not need to be re-ingested.

## Local Inference Environment

The following is the deployment/performance environment used to run the
project. It documents the tested local setup; it is not a separate "GPU version"
of the project.

| Component | Environment |
|---|---|
| CPU | Intel Core i5-12400 |
| RAM | 32 GB |
| GPU | NVIDIA GeForce GTX 1660 Ti 6 GB |
| Ollama | 0.32.14 |
| Model | `llama3.2:latest` |
| GPU offload | `ollama ps` verified `100% GPU` |
| Context | 4096 tokens |

## Measured Performance

Warm-run results measured on the environment above on 2026-08-18, using the
same question (`How does PostgreSQL MVCC work?`), the same three retrieved
chunks, and a 4,871-chunk PostgreSQL 18.4 index:

| Generation backend | Retrieval | Generation | Total | Generated tokens | Tokens/s |
|---|---:|---:|---:|---:|---:|
| Ollama — `llama3.2:latest` | 0.133 s | 1.150 s | 1.283 s | 84 | 73.060 |
| Transformers — Qwen2.5 3B, 4-bit | 1.273 s | 18.363 s | 19.636 s | 136 | 7.406 |

Ollama delivered about 9.9× higher generation throughput in this run. The
Transformers answer contained about 62% more generated tokens, so wall-clock
time alone overstated the underlying throughput gap. Its GPU snapshot remained
around 3.2 GiB / 6 GiB at high utilization, confirming that the 4-bit model fits
the target card even though its generic PyTorch/bitsandbytes path is slower.

Retrieval uses the same Ollama embedding backend in both configurations, so its
timing difference is runtime/cache variation rather than a generation-backend
advantage. Manual inspection found the Transformers answer more closely aligned
with the retrieved MVCC context, but a repeatable question set and groundedness
scoring are still needed for a quality conclusion.

## Setup

Python 3.10–3.12 is recommended, especially for CUDA/PyTorch package
compatibility on Windows.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2
```

Download the pinned PostgreSQL 18.4 official HTML documentation archive, then
build the local vector store. The downloader extracts only HTML documentation
members and records the source URL in a manifest:

```powershell
python scripts/download_docs.py
ollama pull nomic-embed-text
python scripts/ingest_docs.py --reset
```

Ingestion may take a while because embeddings are generated locally for every
chunk. Progress and the final collection count are printed as it runs.
Deterministic chunk IDs and Chroma `upsert` make a non-reset rerun safe; use
`--reset` when intentionally rebuilding the entire `pg_docs` collection.

## Run with Ollama (default)

No backend setting is required:

```powershell
python scripts/chat.py
```

The bilingual demo is also available:

```powershell
python scripts/demo_rag.py
```

To choose another model served by Ollama:

```powershell
$env:OLLAMA_LLM_MODEL = "llama3.2:latest"
python scripts/chat.py
```

## Run with Transformers + CUDA (optional)

Install the optional dependencies using the PyTorch build appropriate for the
installed NVIDIA driver/CUDA environment. If a CUDA-enabled PyTorch build is
already installed, run:

```powershell
pip install -r requirements-transformers.txt
```

The default direct-Transformers model is
`Qwen/Qwen2.5-3B-Instruct`, loaded with bitsandbytes 4-bit NF4 quantization to
fit a 6 GB VRAM target more comfortably. The first run downloads model files
from Hugging Face.

```powershell
$env:LLM_BACKEND = "transformers"
$env:HF_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
$env:HF_LOAD_IN_4BIT = "true"
python scripts/check_inference.py
python scripts/chat.py
```

Configuration variables:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `transformers` |
| `OLLAMA_LLM_MODEL` | `llama3.2:latest` | Ollama generation model |
| `HF_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face model ID or local model path |
| `HF_LOAD_IN_4BIT` | `true` | Enable bitsandbytes 4-bit loading |
| `HF_MAX_NEW_TOKENS` | `512` | Maximum generated tokens per request |
| `HF_TEMPERATURE` | `0.0` | `0` for deterministic generation; positive values enable sampling |

`scripts/check_inference.py` validates backend selection without downloading or
loading the model. The actual chat command verifies CUDA and loads the model on
its first generation request. To try full FP16, set `HF_LOAD_IN_4BIT=false`, but
a 3B model plus runtime/KV-cache overhead may exceed a 6 GB card; 4-bit is the
recommended starting point.

## Ollama vs. Transformers

| Area | Ollama | Transformers + PyTorch |
|---|---|---|
| Project default | Yes | Optional |
| Model lifecycle | Ollama manages loading and serving | Python process loads model directly |
| Setup | Simpler | CUDA/PyTorch and model dependencies required |
| Quantization | Managed by Ollama model package | Explicit bitsandbytes 4-bit configuration |
| Control | Convenient high-level API | Direct tokenizer, dtype, device, and generation control |
| Embeddings in this project | `nomic-embed-text` | Still uses Ollama to preserve the existing index |
| Best use here | Stable everyday local RAG | Learning, experimentation, and backend comparison |

## Benchmark Both Backends

Run one backend at a time so the two generation models do not compete for the
GTX 1660 Ti's 6 GB VRAM. Use exactly the same question for both runs.

```powershell
# 1. Ollama
$env:LLM_BACKEND = "ollama"
python scripts/benchmark_rag.py `
  --question "How does PostgreSQL MVCC work?" `
  --warmup `
  --output results/ollama.json

# Free the Ollama generation model before loading Transformers.
ollama stop llama3.2

# 2. Transformers
$env:LLM_BACKEND = "transformers"
$env:HF_LOAD_IN_4BIT = "true"
python scripts/benchmark_rag.py `
  --question "How does PostgreSQL MVCC work?" `
  --warmup `
  --output results/transformers.json

# 3. Print a comparison table
python scripts/compare_benchmarks.py `
  results/ollama.json results/transformers.json
```

Each result records retrieval, generation and total wall-clock time, prompt and
generated token counts, generation tokens/second, the answer, retrieval
distances, `ollama ps`, and NVIDIA GPU memory/utilization snapshots. Ollama uses
its API `eval_count`; Transformers counts the generated token IDs directly.
With `--warmup`, one unmeasured generation loads the model before timing begins.
Remove this flag when measuring cold-start cost. The first Transformers run may
also download model files, which should be reported separately from inference
latency.

Existing benchmark JSON created before token metrics was added remains readable
and displays `N/A` for those columns. Rerun both backends to produce a fair
tokens/second comparison.

## Project Structure

```text
pg-docs-rag/
├── app/
│   ├── db/session.py
│   └── inference.py             # Ollama/Transformers backend abstraction
├── data/raw/                    # Local PostgreSQL docs (not committed)
├── scripts/
│   ├── check_inference.py       # Lightweight backend configuration check
│   ├── benchmark_rag.py         # Time one backend and capture GPU diagnostics
│   ├── compare_benchmarks.py    # Compare saved benchmark runs
│   ├── download_docs.py         # Fetch pinned official PostgreSQL HTML docs
│   ├── ingest_docs.py
│   ├── demo_rag.py
│   └── chat.py
├── chroma/                      # Persistent local vector store (not committed)
├── requirements.txt
└── requirements-transformers.txt
```

## Guardrail and Design Notes

The chat checks the closest Chroma distance before generating an answer and
refuses unrelated questions. Retrieved chunks are placed in a prompt that tells
the model to answer only from the supplied PostgreSQL context.

This remains a local RAG prototype: it demonstrates document cleaning,
chunking, embeddings, vector retrieval, grounded generation, selectable local
inference, and the deployment trade-offs between a packaged runtime and direct
model control.
