# pg-docs-rag

[![CI](https://github.com/Yneq/pg-docs-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Yneq/pg-docs-rag/actions/workflows/ci.yml)

A fully local Retrieval-Augmented Generation (RAG) system using the official
PostgreSQL documentation as its knowledge base. The project focuses on a clear,
interview-friendly pipeline rather than UI complexity or external APIs.

## Architecture

```text
PostgreSQL docs → clean → chunk → prefixed Ollama embeddings → ChromaDB
                                                        ↓
Question → Ollama embedding → retrieve → guardrail → grounded prompt
                                                        ↓
                                  Ollama or Transformers generation
                                                        ↓
                                      CLI or FastAPI JSON response
```

Ollama remains the default and is always used for `nomic-embed-text`
embeddings. Text generation can optionally use Hugging Face Transformers with
PyTorch and CUDA. Once built with the current embedding pipeline, the same
Chroma index works with either generation backend and does not need to be
re-ingested when switching generators.

## Reference GPU Benchmark Environment

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

### Mac Runtime Validation

The complete API pipeline was also validated on an Apple M1 Mac with 16 GB
unified memory, Python 3.12.3, Ollama 0.17.5, and the same 4,871-chunk
PostgreSQL 18.4 corpus. A warm `POST /api/query` request for
`How does PostgreSQL MVCC work?` produced:

| Retrieval | Generation | Total | Generated tokens | Tokens/s |
|---:|---:|---:|---:|---:|
| 0.039 s | 12.051 s | 12.090 s | 198 | 16.430 |

The response was grounded in `mvcc-intro.html`, `routine-vacuuming.html`, and
`different-replication-solutions.html`. An unrelated chocolate-cake question
was rejected in 0.128 seconds without invoking the generation model. These Mac
figures validate the HTTP pipeline and guardrail; they are not directly
comparable with the CUDA benchmark below because the hardware and generated
token counts differ.

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

These backend figures were captured before normalized, task-prefixed
embeddings were introduced. They remain useful as a generation-throughput
comparison because both backends used the same retrieved chunks in that run;
rerun ingestion and both benchmarks before publishing new end-to-end numbers.

## Setup

Python 3.12 is recommended. The default Ollama backend works on Apple Silicon;
the optional direct Transformers backend in this project currently requires
NVIDIA CUDA and is not enabled on macOS.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2
```

Download the pinned PostgreSQL 18.4 official HTML documentation archive, then
build the local vector store. The downloader extracts only HTML documentation
members and records the source URL in a manifest:

```bash
python scripts/download_docs.py
ollama pull nomic-embed-text
python scripts/ingest_docs.py --reset
```

Ingestion may take a while because embeddings are generated locally for every
chunk. Progress and the final collection count are printed as it runs.
Deterministic chunk IDs and Chroma `upsert` make a non-reset rerun safe; use
`--reset` when intentionally rebuilding the entire `pg_docs` collection.

## Run the FastAPI Service

The API and CLI share the same retrieval, guardrail, prompt, and inference
implementation. Start it from the repository root after building the index:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger documentation. The
service exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Report backend configuration and indexed chunk count |
| `POST` | `/api/query` | Retrieve sources, apply the guardrail, and generate an answer |

Request example:

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does PostgreSQL MVCC work?","top_k":3}'
```

The response includes `grounded`, retrieved source metadata, retrieval and
generation latency, token counts, and generation tokens/second. Synchronous
Chroma and local-model work runs outside FastAPI's event loop. Generation is
serialized per process to avoid concurrent access to one Transformers model
and excessive pressure on a 6 GB GPU.

This local demo does not include authentication or rate limiting. Bind it to
`127.0.0.1` unless it is placed behind an authenticated reverse proxy.

Run the offline unit and API contract tests without Ollama, Chroma data, or a
GPU:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

## Retrieval Quality Evaluation

The versioned evaluation set contains 15 PostgreSQL questions and five
out-of-domain questions. It measures source hit rate at three results, mean
reciprocal rank (MRR), and guardrail accuracy without spending time generating
answers:

```bash
python scripts/evaluate_retrieval.py \
  --output results/retrieval-eval.json
```

The baseline intentionally includes difficult CTE, `EXPLAIN ANALYZE`, and HOT
questions instead of selecting only known successes. The evaluation fails with
a non-zero exit code if retrieval hit@3 drops below 75% or guardrail accuracy
drops below 90%. GitHub Actions validates the dataset and runs all offline tests
on every push and pull request; the real retrieval evaluation stays local
because it requires Ollama and the 4,871-chunk Chroma index.

Current Mac baseline:

| Retrieval hit@3 | MRR@3 | Guardrail accuracy | Result |
|---:|---:|---:|---|
| 80.0% (12/15) | 0.767 | 95.0% (19/20) | PASS |

### Run the API with Docker

Keep Ollama running on the Mac host and start the API container:

```bash
docker compose up --build
```

The Compose configuration mounts the existing `./chroma` index and connects
the container to host Ollama through `host.docker.internal`. This path uses the
default Ollama generation backend. The direct CUDA Transformers option requires
an NVIDIA environment and is not available on this Apple Silicon host.

## Run with Ollama (default)

No backend setting is required:

```bash
python scripts/chat.py
```

The bilingual demo is also available:

```bash
python scripts/demo_rag.py
```

To choose another model served by Ollama:

```bash
export OLLAMA_LLM_MODEL="llama3.2:latest"
python scripts/chat.py
```

## Run with Transformers + CUDA (optional)

Install the optional dependencies using the PyTorch build appropriate for the
installed NVIDIA driver/CUDA environment. If a CUDA-enabled PyTorch build is
already installed, run:

```bash
pip install -r requirements-transformers.txt
```

The default direct-Transformers model is
`Qwen/Qwen2.5-3B-Instruct`, loaded with bitsandbytes 4-bit NF4 quantization to
fit a 6 GB VRAM target more comfortably. The first run downloads model files
from Hugging Face.

```bash
export LLM_BACKEND="transformers"
export HF_MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
export HF_LOAD_IN_4BIT="true"
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
| `CHROMA_PATH` | `./chroma` | Persistent Chroma directory |
| `CHROMA_COLLECTION` | `pg_docs` | Chroma collection name |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `RAG_DISTANCE_THRESHOLD` | `0.6` | Maximum accepted normalized L2 distance |

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

```bash
# 1. Ollama
export LLM_BACKEND="ollama"
python scripts/benchmark_rag.py \
  --question "How does PostgreSQL MVCC work?" \
  --warmup \
  --output results/ollama.json

# Free the Ollama generation model before loading Transformers.
ollama stop llama3.2

# 2. Transformers
export LLM_BACKEND="transformers"
export HF_LOAD_IN_4BIT="true"
python scripts/benchmark_rag.py \
  --question "How does PostgreSQL MVCC work?" \
  --warmup \
  --output results/transformers.json

# 3. Print a comparison table
python scripts/compare_benchmarks.py \
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
│   ├── documents.py             # PostgreSQL documentation parsing/chunking
│   ├── inference.py             # Ollama/Transformers backend abstraction
│   ├── rag.py                   # Shared retrieval/guardrail/generation service
│   └── main.py                  # FastAPI endpoints and Pydantic contracts
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
├── evals/                       # Versioned retrieval/guardrail question set
├── tests/                       # Offline core and API contract tests
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── requirements-dev.txt
├── requirements-test.txt
└── requirements-transformers.txt
```

## Guardrail and Design Notes

The shared RAG service checks the closest Chroma distance before generating an
answer and refuses unrelated questions. Retrieved chunks are labeled in a
prompt that tells the model to answer only from the supplied PostgreSQL context
and cite `[Source N]`. The API returns each source's title, file, chunk index,
and distance so clients can inspect the evidence.

`nomic-embed-text` receives its required retrieval task prefixes:
`search_document:` during ingestion and `search_query:` during retrieval. The
current Ollama `/api/embed` endpoint returns normalized vectors, making the
distance threshold stable and interpretable. Common PostgreSQL acronyms such as
MVCC, WAL, CTE, PITR, and HOT are expanded before query embedding so terse
technical questions retrieve their full-form documentation. Rebuild an index
created by an older project version with
`python scripts/ingest_docs.py --reset` before using the API.

This remains a local RAG prototype: it demonstrates document cleaning,
chunking, embeddings, vector retrieval, grounded generation, selectable local
inference, an HTTP contract, and the deployment trade-offs between a packaged
runtime and direct model control.
