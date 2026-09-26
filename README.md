# Layarank: Fast Local Relevance Filtering & Reranking for RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/layarank.svg)](https://pypi.org/project/layarank/)

**Layarank** is a high-performance Python library for reranking search results and filtering retrieved documents using **Laya** local neural routers.

It runs **completely locally or on your own hardware**, eliminating cloud API keys, rate limits, and network latency.

---

## 🎯 Why Layarank?

In Retrieval-Augmented Generation (RAG) and semantic search, retrieval algorithms often return passages that match keywords or share general topic overlap without containing the specific factual evidence needed to answer the query.

Passing non-evidential passages to downstream LLMs causes:
- **Token Waste**: Inflated LLM prompt token costs and slower generation.
- **Distraction & Hallucination**: "Lost in the Middle" syndrome where LLMs synthesize irrelevant details.

**Layarank solves this** by evaluating retrieved documents with specialized neural evidence criteria in a **single local forward pass**:
- **Relevance Filtering (`relevance_rerank`)**: Drops passages below an evidence threshold (default `0.2`), retaining only actionable facts.
- **Topical Reranking (`rerank`)**: Orders candidates by relevance without aggressive filtering (default `0.0`).

---

## 📦 Requirements & Prerequisites

- **Python**: 3.10 or higher
- **OS**: Linux, macOS, or Windows
- **Compute**: CPU or CUDA-capable GPU (Apple Silicon MPS also supported)
- **Dependencies**: `laya>=0.3.3` (automatically installed)

---

## 🚀 Installation & Setup

### Option 1: Install from PyPI

```bash
pip install layarank
```

### Option 2: Install from Source (Development Mode)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/layarank.git
   cd layarank
   ```

2. Create and activate a virtual environment:
   ```bash
   # On macOS / Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

---

## 💡 Step-by-Step Usage Guide

### Step 1: Filter Evidence for RAG (`relevance_rerank`)

Use `relevance_rerank` when you want to filter out distractor passages before constructing an LLM prompt.

```python
from layarank import LayaRank

# 1. Initialize ranker (preloads local neural router)
ranker = LayaRank()

# 2. Define query and retrieved candidate documents
query = "How long do I have to return an online order to ACME Shop?"
documents = [
    "ACME Shop accepts online returns within 30 days of delivery.",
    "For ACME Shop online orders, submit your return request within 30 days of receiving the item.",
    "ACME Shop in-store purchases can be returned within 14 days of purchase.",
    "ACME Shop products come with a one-year repair warranty covering manufacturing defects.",
    "FooBar Shop accepts online returns within 60 days of delivery.",
]

# 3. Filter candidates (default threshold is 0.2)
evidence = ranker.relevance_rerank(query, documents, threshold=0.2)

# 4. Use only verified evidence
for item in evidence["results"]:
    print(f"[{item['score']:.2f}] (Doc #{item['document_index']}): {item['text']}")
```

**Output:**
```text
[0.86] (Doc #0): ACME Shop accepts online returns within 30 days of delivery.
[0.86] (Doc #1): For ACME Shop online orders, submit your return request within 30 days of receiving the item.
```

---

### Step 2: Order Search Results (`rerank`)

Use `rerank` when you want to re-order search results by relevance without filtering candidates out.

```python
from layarank import LayaRank

ranker = LayaRank()

query = "Which planet is called the Red Planet?"
documents = [
    "Venus has a thick atmosphere of carbon dioxide.",
    "Mars is commonly known as the Red Planet due to iron oxide.",
    "Jupiter is the largest planet in our solar system."
]

# Rerank and keep top 2
response = ranker.rerank(query, documents, top_k=2)

for rank, item in enumerate(response["results"], 1):
    print(f"{rank}. [Score: {item['score']:.3f}] {item['text']}")
```

---

### Step 3: Async Usage in FastAPI / Async Pipelines

Layarank provides non-blocking asynchronous methods (`a_rerank`, `a_relevance_rerank`):

```python
import asyncio
from layarank import LayaRank

async def search_and_filter(query: str, raw_docs: list[str]):
    ranker = LayaRank()
    
    # Non-blocking async inference
    filtered = await ranker.a_relevance_rerank(query, raw_docs, threshold=0.2)
    return filtered["results"]

asyncio.run(search_and_filter("Photosynthesis", ["Doc 1...", "Doc 2..."]))
```

---

### Step 4: RAG Pipeline Integration (Example with LLM Prompting)

```python
from layarank import LayaRank

def rag_pipeline(user_query: str, retrieved_chunks: list[str]) -> str:
    ranker = LayaRank()
    
    # 1. Filter out irrelevant chunks
    filtered = ranker.relevance_rerank(user_query, retrieved_chunks, threshold=0.25)
    
    # 2. Check if any evidence passed
    if not filtered["results"]:
        return "I could not find sufficient factual evidence to answer your question."
    
    # 3. Assemble clean context for LLM
    context = "\n\n".join([f"- {item['text']}" for item in filtered["results"]])
    prompt = f"Answer the question using only the evidence below:\n\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    return prompt
```

---

### Step 5: Detailed Diagnostics (`detail=True`)

Set `detail=True` to inspect execution timings, chunk splits, and all evaluated scores (including filtered-out candidates):

```python
response = ranker.relevance_rerank(query, documents, detail=True)

detail = response["detail"]
print(f"Elapsed Time: {detail['timing_ms']:.2f} ms")
print(f"Selection Stats: {detail['selection']}")
# Prints: {'input_count': 5, 'above_threshold_count': 2, 'returned_count': 2, 'top_k': None}

for doc_stat in detail["documents"]:
    print(f"Doc #{doc_stat['document_index']} - Score: {doc_stat['score']:.3f}, Passed: {doc_stat['passes_threshold']}")
```

---

## 🛠️ API Reference

### `LayaRank` Constructor

```python
LayaRank(
    router=None,             # Pre-instantiated laya.Router or laya.Agent
    model=None,              # Optional model checkpoint override
    device=None,             # 'cpu', 'cuda', 'mps'
    preload=True,            # Whether to preload router weights into memory
    mode="listwise",         # 'listwise' (default), 'pointwise', or 'pairwise'
    max_state_budget=24000,  # Character/token budget before splitting long candidate sets
    instruction=None         # Custom default prompt dictionary
)
```

### Methods

| Method | Description | Default Threshold |
| :--- | :--- | :---: |
| `relevance_rerank(query, documents, ...)` | Filter and retain verified evidence | `0.2` |
| `rerank(query, documents, ...)` | Order candidates by topical relevance | `0.0` |
| `a_relevance_rerank(query, documents, ...)` | Async non-blocking relevance filter | `0.2` |
| `a_rerank(query, documents, ...)` | Async non-blocking reranking | `0.0` |

---

## 🧪 Running Tests

To run the full test suite:

```bash
pytest -v
```

---

## 📦 Building and Publishing to PyPI

1. Install build tools:
   ```bash
   pip install build twine
   ```

2. Build source and wheel distributions:
   ```bash
   python -m build
   ```

3. Verify packages:
   ```bash
   twine check dist/*
   ```

4. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
