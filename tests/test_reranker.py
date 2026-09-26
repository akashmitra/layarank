"""Tests for LayaRank API contracts and behavior."""

import pytest
from layarank import LayaRank, LayaRanker
from layarank.errors import ConfigurationError


class MockRouter:
    """Deterministic mock router for testing reranker logic."""

    def predict(self, state: dict, questions: dict, **kwargs) -> dict:
        answers = {}
        for key in questions.keys():
            if "documents" in state:
                doc_text = state["documents"].get(key, "")
                score = 0.95 if "MATCH_TARGET" in doc_text else 0.05
            elif "document" in state:
                doc_text = state.get("document", "")
                score = 0.95 if "MATCH_TARGET" in doc_text else 0.05
            elif "better" in questions:
                left_doc = state.get("left", "")
                score = 0.9 if "MATCH_TARGET" in left_doc else 0.1
            else:
                score = 0.5
            answers[key] = {"type": "noul", "noul": score}
        return {"answers": answers}


@pytest.fixture
def mock_reranker():
    return LayaRank(router=MockRouter(), mode="listwise")


def test_alias_equivalence():
    assert LayaRank is LayaRanker


def test_empty_documents(mock_reranker):
    res = mock_reranker.rerank("query", [], detail=True)
    assert res["results"] == []
    assert res["detail"]["selection"]["input_count"] == 0


def test_top_k_zero(mock_reranker):
    docs = ["MATCH_TARGET doc 1", "distractor doc 2"]
    res = mock_reranker.rerank("query", docs, top_k=0, detail=True)
    assert res["results"] == []
    assert res["detail"]["selection"]["returned_count"] == 0


def test_rerank_ordering(mock_reranker):
    docs = [
        "distractor document alpha",
        "MATCH_TARGET relevant document",
        "distractor document beta",
    ]
    res = mock_reranker.rerank("query", docs)
    results = res["results"]

    assert len(results) == 3
    assert results[0]["document_index"] == 1
    assert results[0]["score"] == 0.95
    assert results[0]["text"] == "MATCH_TARGET relevant document"


def test_relevance_rerank_threshold_filtering(mock_reranker):
    docs = [
        "distractor document alpha",
        "MATCH_TARGET relevant document",
        "distractor document beta",
    ]
    res = mock_reranker.relevance_rerank("query", docs, threshold=0.2, detail=True)
    results = res["results"]

    assert len(results) == 1
    assert results[0]["document_index"] == 1
    assert results[0]["score"] == 0.95

    selection = res["detail"]["selection"]
    assert selection["input_count"] == 3
    assert selection["above_threshold_count"] == 1
    assert selection["returned_count"] == 1


def test_return_documents_flag(mock_reranker):
    docs = ["MATCH_TARGET relevant document"]
    res = mock_reranker.rerank("query", docs, return_documents=False)
    assert "text" not in res["results"][0]
    assert "document_index" in res["results"][0]
    assert "score" in res["results"][0]


def test_invalid_arguments(mock_reranker):
    with pytest.raises(ConfigurationError):
        mock_reranker.rerank("", ["doc"])

    with pytest.raises(ConfigurationError):
        mock_reranker.rerank("query", "not a list")

    with pytest.raises(ConfigurationError):
        mock_reranker.rerank("query", ["doc"], threshold="invalid")

    with pytest.raises(ConfigurationError):
        mock_reranker.rerank("query", ["doc"], top_k=-1)


@pytest.mark.asyncio
async def test_async_rerank(mock_reranker):
    docs = ["MATCH_TARGET relevant doc", "distractor doc"]
    res = await mock_reranker.a_relevance_rerank("query", docs, threshold=0.2)
    assert len(res["results"]) == 1
    assert res["results"][0]["document_index"] == 0
