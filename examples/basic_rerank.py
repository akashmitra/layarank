"""Example: Basic reranking to order search candidates by topical relevance with LayaRank."""

import sys
from pathlib import Path

# Ensure src is in sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layarank import LayaRank


def main():
    print("Initializing LayaRank...")
    reranker = LayaRank()

    query = "Which planet is called the Red Planet?"
    documents = [
        "Venus has a thick atmosphere containing clouds of sulfuric acid.",
        "Mars is often referred to as the Red Planet due to its reddish appearance from iron oxide.",
        "Jupiter is the largest planet in our solar system, famous for the Great Red Spot.",
        "Mercury is the smallest and closest planet to the Sun.",
    ]

    print(f"\nQuery: {query}\n")
    print("Scoring candidates...")
    results = reranker.rerank(query, documents, top_k=3, detail=True)

    print("\n--- Ranked Results ---")
    for rank, item in enumerate(results["results"], 1):
        print(f"{rank}. [Score: {item['score']:.4f}] (Doc #{item['document_index']}): {item['text']}")

    print(f"\nElapsed time: {results['detail']['timing_ms']:.2f} ms")


if __name__ == "__main__":
    main()
