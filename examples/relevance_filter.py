"""Example: Relevance filtering for RAG to drop non-evidential and distractor passages with LayaRank."""

import sys
from pathlib import Path

# Ensure src is in sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layarank import LayaRank


def main():
    print("Initializing LayaRank...")
    reranker = LayaRank()

    query = "How long do I have to return an online order to ACME Shop?"
    documents = [
        "ACME Shop accepts online returns within 30 days of delivery.",
        "For ACME Shop online orders, submit your return request within 30 days of receiving the item.",
        "ACME Shop in-store purchases can be returned within 14 days of purchase.",
        "ACME Shop products come with a one-year repair warranty covering manufacturing defects.",
        "FooBar Shop accepts online returns within 60 days of delivery.",
    ]

    print(f"Query: {query}")
    print(f"Input documents count: {len(documents)}\n")

    # Run relevance filtering with default threshold=0.2
    response = reranker.relevance_rerank(query, documents, threshold=0.2, detail=True)

    print("--- Filtered Evidence (passed threshold >= 0.2) ---")
    for item in response["results"]:
        print(f"[PASS] [{item['score']:.4f}] (Doc #{item['document_index']}): {item['text']}")

    print("\n--- Diagnostic Details ---")
    selection = response["detail"]["selection"]
    print(f"Total Evaluated : {selection['input_count']}")
    print(f"Above Threshold : {selection['above_threshold_count']}")
    print(f"Returned Count  : {selection['returned_count']}")
    print(f"Elapsed Time    : {response['detail']['timing_ms']:.2f} ms")


if __name__ == "__main__":
    main()
