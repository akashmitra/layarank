"""Listwise partitioning and scoring coordination for LayaReranker."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

from .errors import ContextLimitError

SHUFFLE_SEED = "laya-reranker-listwise-v1"


def balanced_chunks(indices: list[int], lengths: list[int], count: int) -> list[list[int]]:
    """Partition document indices across `count` chunks balancing total lengths."""
    base, remainder = divmod(len(indices), count)
    capacities = [base + (i >= count - remainder) for i in range(count)]
    chunks: list[list[int]] = [[] for _ in range(count)]
    totals = [0] * count

    # Sort descending by length to pack heaviest items first
    for index in sorted(indices, key=lambda i: (-lengths[i], i)):
        # Pick the chunk with capacity remaining that has smallest current total
        target = min(
            (i for i in range(count) if len(chunks[i]) < capacities[i]),
            key=lambda i: (totals[i], len(chunks[i]), i),
        )
        chunks[target].append(index)
        totals[target] += lengths[index]

    return chunks


def score_listwise_sync(
    *,
    query: str,
    documents: list[str],
    lengths: list[int],
    max_state_budget: int,
    predict_fn: Callable[[list[int]], list[float]],
    splits_trace: list[dict[str, Any]],
) -> list[float]:
    """Score a set of documents listwise, recursively splitting if budget is exceeded."""
    scores = [0.0] * len(lengths)
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()

    def fits_budget(indices: list[int], budget: int) -> bool:
        total_len = sum(lengths[i] for i in indices)
        # Query + all documents + overhead
        return (len(query) + total_len + len(indices) * 50) <= budget

    def split_and_score(indices: list[int], depth: int, current_budget: int, reason: str) -> None:
        if len(indices) == 1:
            # Single document cannot be split further; run it
            try:
                values = predict_fn(indices)
                scores[indices[0]] = values[0]
                return
            except Exception as exc:
                raise ContextLimitError(
                    f"Document at index {indices[0]} cannot fit the context budget."
                ) from exc

        count = max(2, math.ceil(sum(lengths[i] for i in indices) / max(1, current_budget)))
        count = min(len(indices), count)

        while True:
            chunks = balanced_chunks(indices, lengths, count)
            if all(fits_budget(c, current_budget) for c in chunks) or count >= len(indices):
                break
            count += 1

        for chunk in chunks:
            # Deterministically shuffle to avoid positional bias
            chunk.sort(
                key=lambda i: hashlib.sha256(f"{SHUFFLE_SEED}:{query_hash}:{i}".encode("utf-8")).digest()
            )

        splits_trace.append(
            {
                "reason": reason,
                "depth": depth,
                "document_indices": indices,
                "budget": current_budget,
                "chunks": chunks,
            }
        )

        for chunk in chunks:
            evaluate_chunk(chunk, depth + 1, current_budget)

    def evaluate_chunk(indices: list[int], depth: int, current_budget: int) -> None:
        if not fits_budget(indices, current_budget):
            split_and_score(indices, depth, current_budget, "budget_exceeded")
            return

        try:
            values = predict_fn(indices)
            for idx, score in zip(indices, values, strict=True):
                scores[idx] = score
        except ContextLimitError:
            # If router context limit was hit at runtime, split with halved budget
            split_and_score(indices, depth, max(1, current_budget // 2), "runtime_context_exceeded")

    evaluate_chunk(list(range(len(documents))), depth=0, current_budget=max_state_budget)
    return scores
