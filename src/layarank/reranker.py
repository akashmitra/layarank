"""LayaRank: High-performance local relevance filtering and reranking powered by Laya."""

from __future__ import annotations

import copy
import time
from typing import Any, Sequence

import laya
from laya import Router

from ._ranking import score_listwise_sync
from ._runtime import run_in_thread
from .errors import ConfigurationError, ModelExecutionError
from .instructions import (
    PAIRWISE_INSTRUCTION,
    POINTWISE_RELEVANCE_INSTRUCTION,
    RELEVANCE_INSTRUCTION,
    RERANK_INSTRUCTION,
    validate_instruction,
)


class LayaRank:
    """Rerank and filter document collections using Laya neural router models."""

    def __init__(
        self,
        router: Any | None = None,
        *,
        model: str | None = None,
        device: str | None = None,
        preload: bool = True,
        mode: str = "listwise",
        max_state_budget: int = 24000,
        instruction: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a LayaRank instance.

        Args:
            router: Pre-instantiated `laya.Router` or `laya.Agent`. If None, a `Router` is created.
            model: Optional model checkpoint to route or load.
            device: Computing device, e.g., 'cpu', 'cuda', 'mps'.
            preload: Whether to preload router models into memory on init.
            mode: Ranking mode ('listwise', 'pointwise', or 'pairwise').
            max_state_budget: Context length/character budget before splitting candidates.
            instruction: Optional default prompt instructions/criteria dictionary.
        """
        if mode not in ("listwise", "pointwise", "pairwise"):
            raise ConfigurationError("mode must be one of 'listwise', 'pointwise', or 'pairwise'.")

        if router is not None:
            self._router = router
        else:
            self._router = Router(preload=preload, device=device)

        self._model = model
        self._mode = mode
        self._max_state_budget = max_state_budget

        self._default_instruction: dict[str, Any] | None = None
        if instruction is not None:
            self._default_instruction = validate_instruction(instruction, self._mode)

    @property
    def router(self) -> Any:
        """Return the underlying Laya Router or Agent."""
        return self._router

    @property
    def mode(self) -> str:
        """Return the current scoring mode."""
        return self._mode

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.0,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Order retrieved documents by topical relevance.

        Args:
            query: The search query string.
            documents: Sequence of candidate document texts.
            instruction: Optional per-call prompt dictionary.
            threshold: Minimum score required to retain a document (default 0.0).
            top_k: Optional maximum number of results to return.
            return_documents: Whether to include original document texts in the output.
            detail: Whether to include comprehensive execution and scoring diagnostics.

        Returns:
            Dictionary containing 'results' list and optional 'detail' dictionary.
        """
        effective_instruction = instruction or self._default_instruction or (
            PAIRWISE_INSTRUCTION if self._mode == "pairwise" else RERANK_INSTRUCTION
        )
        return self._score_and_filter(
            query=query,
            documents=documents,
            instruction=effective_instruction,
            threshold=threshold,
            top_k=top_k,
            return_documents=return_documents,
            detail=detail,
        )

    def relevance_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.2,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Select and filter evidence documents needed to answer the query.

        Args:
            query: The question or query string.
            documents: Sequence of candidate document texts.
            instruction: Optional prompt override.
            threshold: Minimum evidence score cutoff (default 0.2).
            top_k: Optional maximum number of results to return.
            return_documents: Whether to include original document texts in output.
            detail: Whether to include comprehensive diagnostics.

        Returns:
            Dictionary containing 'results' list and optional 'detail' dictionary.
        """
        if self._mode == "pairwise":
            raise ConfigurationError("relevance_rerank is not supported with mode='pairwise'.")

        preset = POINTWISE_RELEVANCE_INSTRUCTION if self._mode == "pointwise" else RELEVANCE_INSTRUCTION
        effective_instruction = instruction or self._default_instruction or preset

        return self._score_and_filter(
            query=query,
            documents=documents,
            instruction=effective_instruction,
            threshold=threshold,
            top_k=top_k,
            return_documents=return_documents,
            detail=detail,
        )

    async def a_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.0,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously order retrieved documents by relevance without blocking the event loop."""
        return await run_in_thread(
            self.rerank,
            query=query,
            documents=documents,
            instruction=instruction,
            threshold=threshold,
            top_k=top_k,
            return_documents=return_documents,
            detail=detail,
        )

    async def a_relevance_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.2,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously select and filter evidence documents without blocking the event loop."""
        return await run_in_thread(
            self.relevance_rerank,
            query=query,
            documents=documents,
            instruction=instruction,
            threshold=threshold,
            top_k=top_k,
            return_documents=return_documents,
            detail=detail,
        )

    def _score_and_filter(
        self,
        *,
        query: str,
        documents: Sequence[str],
        instruction: dict[str, Any],
        threshold: float,
        top_k: int | None,
        return_documents: bool,
        detail: bool,
    ) -> dict[str, Any]:
        # Input validation
        if not isinstance(query, str) or not query.strip():
            raise ConfigurationError("query must be a non-empty string.")
        if not isinstance(documents, (list, tuple)):
            raise ConfigurationError("documents must be a sequence of strings.")
        if any(not isinstance(doc, str) for doc in documents):
            raise ConfigurationError("all items in documents must be strings.")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ConfigurationError("threshold must be a numeric value.")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
            raise ConfigurationError("top_k must be a non-negative integer or None.")

        validated_instruction = validate_instruction(instruction, self._mode)
        doc_list = list(documents)
        start_time = time.perf_counter()

        # Handle trivial cases
        if not doc_list or top_k == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            response: dict[str, Any] = {"results": []}
            if detail:
                response["detail"] = {
                    "configuration": {
                        "mode": self._mode,
                        "threshold": threshold,
                        "top_k": top_k,
                        "instruction": validated_instruction,
                    },
                    "selection": {
                        "input_count": len(doc_list),
                        "above_threshold_count": 0,
                        "returned_count": 0,
                        "top_k": top_k,
                    },
                    "documents": [],
                    "splits": [],
                    "timing_ms": elapsed_ms,
                }
            return response

        splits_trace: list[dict[str, Any]] = []

        # Execute scoring based on mode
        if self._mode == "listwise":
            scores = self._score_listwise(query, doc_list, validated_instruction, splits_trace)
        elif self._mode == "pointwise":
            scores = self._score_pointwise(query, doc_list, validated_instruction)
        elif self._mode == "pairwise":
            scores = self._score_pairwise(query, doc_list, validated_instruction)
        else:
            raise ConfigurationError(f"Unsupported mode: {self._mode}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Build candidate items with zero-based original indices
        candidates = []
        doc_details = []
        above_threshold_count = 0

        for idx, (doc_text, score) in enumerate(zip(doc_list, scores, strict=True)):
            passes = score >= threshold
            if passes:
                above_threshold_count += 1
                item: dict[str, Any] = {
                    "document_index": idx,
                    "score": score,
                }
                if return_documents:
                    item["text"] = doc_text
                candidates.append(item)

            if detail:
                doc_details.append(
                    {
                        "document_index": idx,
                        "score": score,
                        "passes_threshold": passes,
                        "text_length": len(doc_text),
                    }
                )

        # Sort candidates stably by descending score
        candidates.sort(key=lambda c: c["score"], reverse=True)

        # Apply top_k cutoff
        if top_k is not None:
            final_results = candidates[:top_k]
        else:
            final_results = candidates

        response = {"results": final_results}

        if detail:
            response["detail"] = {
                "configuration": {
                    "mode": self._mode,
                    "threshold": threshold,
                    "top_k": top_k,
                    "instruction": validated_instruction,
                },
                "selection": {
                    "input_count": len(doc_list),
                    "above_threshold_count": above_threshold_count,
                    "returned_count": len(final_results),
                    "top_k": top_k,
                },
                "documents": doc_details,
                "splits": splits_trace,
                "timing_ms": elapsed_ms,
            }

        return response

    def _score_listwise(
        self,
        query: str,
        documents: list[str],
        instruction: dict[str, Any],
        splits_trace: list[dict[str, Any]],
    ) -> list[float]:
        lengths = [len(doc) for doc in documents]

        def predict_chunk(indices: list[int]) -> list[float]:
            state = {
                "query": query,
                "documents": {f"doc_{i}": documents[i] for i in indices},
            }
            questions = {
                f"doc_{i}": {
                    "type": "noul",
                    "instructions": instruction["instructions"].format(document=f"documents.doc_{i}"),
                    "criteria": instruction["criteria"],
                }
                for i in indices
            }
            try:
                if self._model:
                    res = self._router.predict(state, questions, model=self._model)
                else:
                    res = self._router.predict(state, questions)

                answers = res.get("answers", {})
                scores = []
                for i in indices:
                    key = f"doc_{i}"
                    ans = answers.get(key, {})
                    val = ans.get("noul", 0.0) if isinstance(ans, dict) else 0.0
                    scores.append(float(val))
                return scores
            except Exception as exc:
                raise ModelExecutionError(f"Laya router prediction failed: {exc}") from exc

        return score_listwise_sync(
            query=query,
            documents=documents,
            lengths=lengths,
            max_state_budget=self._max_state_budget,
            predict_fn=predict_chunk,
            splits_trace=splits_trace,
        )

    def _score_pointwise(
        self,
        query: str,
        documents: list[str],
        instruction: dict[str, Any],
    ) -> list[float]:
        scores = []
        for doc in documents:
            state = {
                "query": query,
                "document": doc,
            }
            questions = {
                "relevance": {
                    "type": "noul",
                    "instructions": instruction["instructions"].format(document="document"),
                    "criteria": instruction["criteria"],
                }
            }
            try:
                if self._model:
                    res = self._router.predict(state, questions, model=self._model)
                else:
                    res = self._router.predict(state, questions)
                ans = res.get("answers", {}).get("relevance", {})
                val = ans.get("noul", 0.0) if isinstance(ans, dict) else 0.0
                scores.append(float(val))
            except Exception as exc:
                raise ModelExecutionError(f"Pointwise scoring failed: {exc}") from exc
        return scores

    def _score_pairwise(
        self,
        query: str,
        documents: list[str],
        instruction: dict[str, Any],
    ) -> list[float]:
        n = len(documents)
        if n <= 1:
            return [1.0] * n

        win_counts = [0.0] * n
        comparisons = 0

        for i in range(n):
            for j in range(i + 1, n):
                state = {
                    "query": query,
                    "left": documents[i],
                    "right": documents[j],
                }
                questions = {
                    "better": {
                        "type": "noul",
                        "instructions": instruction["instructions"].format(left="left", right="right"),
                        "criteria": instruction["criteria"],
                    }
                }
                try:
                    if self._model:
                        res = self._router.predict(state, questions, model=self._model)
                    else:
                        res = self._router.predict(state, questions)
                    prob_i = float(res.get("answers", {}).get("better", {}).get("noul", 0.5))
                    win_counts[i] += prob_i
                    win_counts[j] += (1.0 - prob_i)
                    comparisons += 1
                except Exception as exc:
                    raise ModelExecutionError(f"Pairwise scoring failed: {exc}") from exc

        # Normalize win score to [0, 1]
        max_possible = max(1.0, float(n - 1))
        return [win / max_possible for win in win_counts]


# Alias for backward compatibility or alternate naming preference
LayaRanker = LayaRank
