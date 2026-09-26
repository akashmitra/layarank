# Layarank Implementation Plan & Architecture

This document outlines the architectural design, core concepts, and implementation roadmap for **`layarank`**, a fast local relevance filtering and reranking library built with **Laya**.

---

## 1. Executive Summary & Problem Statement

In Retrieval-Augmented Generation (RAG) and search pipelines, keyword or dense vector retrieval often returns documents that are lexically or semantically close to a query without actually providing the factual evidence needed to answer it.

Passing irrelevant or distracting passages to LLMs causes:
- **Token Bloat**: Higher inference costs and latency.
- **Lost in the Middle / Distraction**: Reduced answer accuracy and increased hallucination risk.

`layarank` provides:
1. **Relevance Filtering (`relevance_rerank`)**: Scores documents for factual evidence usefulness and filters candidates below a threshold (default `0.2`), retaining only verified evidence (including partial facts and multi-hop links).
2. **Relevance Ordering (`rerank`)**: Stably sorts candidate documents by relevance score (default `threshold=0.0`).
3. **Local & High-Performance Execution**: Powered by `laya.Router` / `laya.Agent` for zero-network-latency, local CPU/GPU neural evaluation using native `noul` (probability/binary relevance) primitives.

---

## 2. Core Architecture

- **Engine / Backend**: Local `laya.Router` / `laya.Agent`
- **Latency & Network**: In-process, local GPU/CPU forward pass
- **Dependencies**: Zero cloud dependencies; local weights
- **Scoring Primitive**: `noul` question schema supported natively by Laya
- **Modes**: Listwise (default), Pointwise, Pairwise
- **Context Management**: Token budget partitioning based on Laya context
- **API Interface**: Sync & Async (`rerank`, `relevance_rerank`, `a_rerank`, `a_relevance_rerank`)

---

## 3. Architecture & Scoring Modes

### 3.1 Listwise Scoring (Default)
In listwise mode, multiple candidate documents are evaluated in a single forward pass:
- **State Schema**:
  ```python
  state = {
      "query": query,
      "documents": {
          "doc_0": "Document 0 text...",
          "doc_1": "Document 1 text...",
      }
  }
  ```
- **Question Schema**:
  ```python
  questions = {
      "doc_0": {
          "type": "noul",
          "instructions": "Does document doc_0 help answer `query`?",
          "criteria": {
              "true": "Contains specific information that answers or supports answering the query.",
              "false": "Unrelated, merely tangential, or lacks needed facts."
          }
      },
      ...
  }
  ```

### 3.2 Pointwise Scoring
Evaluates each `(query, document)` pair independently:
- Ideal for very long individual documents or streaming retrieval.

### 3.3 Pairwise Scoring
Compares candidate pairs (`{left}` vs `{right}`) to compute win rates and sort candidates when fine-grained relative ordering is preferred.

---

## 4. Prompt Presets & Relevance Calibration

1. **`RELEVANCE_INSTRUCTION`**:
   - Instructs Laya to evaluate whether a document provides concrete evidence (including partial facts, entity disambiguation, and multi-hop links) and actively penalizes topic overlap without usable facts.
   - Pushes non-evidential documents towards $0.0$, making threshold filtering (`threshold >= 0.2`) highly effective.
2. **`RERANK_INSTRUCTION`**:
   - Focuses on general topical relevance and ordering.
3. **`POINTWISE_RELEVANCE_INSTRUCTION` & `PAIRWISE_INSTRUCTION`**:
   - Specialized prompts tailored for 1-on-1 and tournament scoring.

---

## 5. Directory Structure

```text
layarank/
├── docs/
│   └── plan.md                      # Architecture and design document
├── src/
│   └── layarank/
│       ├── __init__.py              # Public exports: LayaRank, LayaRanker, instructions, errors
│       ├── reranker.py              # Main LayaRank class with sync & async methods
│       ├── instructions.py          # Prompt templates & validation logic
│       ├── _ranking.py              # Balanced chunking, token estimation, listwise scoring
│       ├── _runtime.py              # Thread execution and async coordination
│       └── errors.py                # Domain exceptions (ConfigurationError, ContextLimitError)
├── examples/
│   ├── basic_rerank.py              # Simple ordering walkthrough
│   └── relevance_filter.py          # Evidence selection & filtering for RAG
└── tests/
    ├── test_instructions.py         # Prompt validation tests
    ├── test_ranking.py              # Balanced chunking tests
    └── test_reranker.py             # Full end-to-end reranker tests
```
