"""Tests for balanced chunking and listwise partitioning."""

from layarank._ranking import balanced_chunks, score_listwise_sync


def test_balanced_chunks_distribution():
    indices = [0, 1, 2, 3, 4]
    lengths = [100, 20, 80, 40, 60]

    chunks = balanced_chunks(indices, lengths, 2)
    assert len(chunks) == 2
    flattened = [idx for c in chunks for idx in c]
    assert sorted(flattened) == [0, 1, 2, 3, 4]


def test_score_listwise_sync_execution():
    query = "test query"
    documents = ["doc A", "doc B", "doc C"]
    lengths = [len(d) for d in documents]
    splits: list[dict] = []

    def mock_predict(indices: list[int]) -> list[float]:
        return [float(i) * 0.1 for i in indices]

    scores = score_listwise_sync(
        query=query,
        documents=documents,
        lengths=lengths,
        max_state_budget=1000,
        predict_fn=mock_predict,
        splits_trace=splits,
    )

    assert len(scores) == 3
    assert scores[0] == 0.0
    assert scores[1] == 0.1
    assert scores[2] == 0.2
