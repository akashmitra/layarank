"""Example: Relevance filtering for RAG to drop non-evidential and distractor passages with LayaRank.

This example demonstrates how to use LayaRank's `relevance_rerank()` to evaluate candidate
passages against a query, retaining true answer evidence while discarding distractor passages
and off-topic noise.

Topic: The Lord of the Rings (LOTR)
Query: "How can the One Ring be destroyed, and where must it be taken?"

Expected Result Overview:
------------------------
1. Direct Evidence (Doc #0): High score (~0.76+ / top rank). Directly explains how the Ring
   is destroyed (fires of Mount Doom) and where it was forged.
2. Supporting Evidence (Doc #1): High score (~0.72+). Provides necessary supporting facts
   about Frodo carrying the Ring to Mordor / Cracks of Doom.
3. Partial / Contextual Evidence (Doc #2): Moderate score (~0.70+). Mentions Sauron forging it in Mount Doom.
4. Distractors / Non-evidential passages (Doc #3, Doc #4, Doc #5):
   - Doc #3 (Elrond lore) & Doc #4 (Gimli's axe failing): Mention Ring/Rivendell but lack answer facts.
   - Doc #5 (D&D Bag of Holding): Completely irrelevant cross-domain distractor.
"""

import sys
from pathlib import Path

# Ensure src is in sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layarank import LayaRank


def main():
    print("Initializing LayaRank...")
    reranker = LayaRank()

    # Query asking a specific factual question about Middle-earth lore
    query = "How can the One Ring be destroyed, and where must it be taken?"

    # A retrieval candidate pool containing direct evidence, partial evidence,
    # topical distractors (LOTR lore without the answer), and completely irrelevant text (D&D).
    documents = [
        # Doc 0: Direct, complete answer evidence
        "The One Ring can only be destroyed by casting it into the fires of Mount Doom (Orodruin) in Mordor, where it was originally forged by Sauron.",
        # Doc 1: Supporting evidence (who carries it and where)
        "Frodo Baggins was appointed Ring-bearer and tasked with carrying the One Ring to Mordor to cast it into the Cracks of Doom.",
        # Doc 2: Contextual background (where it was made)
        "The One Ring was forged by the Dark Lord Sauron in the fires of Mount Doom during the Second Age.",
        # Doc 3: Topical distractor (LOTR lore about Rivendell/Elves, but does not answer how to destroy the Ring)
        "Elrond was the Lord of Rivendell and bearer of Vilya, the Ring of Air, one of the three Elven Rings of Power.",
        # Doc 4: Topical distractor (Mentions trying to destroy the Ring with an axe, but not the actual solution)
        "Gimli offered his battleaxe to destroy the One Ring at the Council of Elrond, but the weapon shattered upon impact.",
        # Doc 5: Irrelevant cross-domain distractor (Dungeons & Dragons item description)
        "In Dungeons & Dragons 5e, a Bag of Holding is a wondrous item that opens into an extradimensional space holding up to 500 pounds.",
    ]

    print(f"\nQuery: {query}")
    print(f"Input documents count: {len(documents)}\n")

    # -------------------------------------------------------------------------
    # Relevance Filtering:
    # `relevance_rerank` scores how well each passage provides factual evidence
    # to answer the query. Passages meeting or exceeding the `threshold` are kept.
    # -------------------------------------------------------------------------
    threshold = 0.2  # Cutoff score for retaining evidence
    response = reranker.relevance_rerank(
        query=query,
        documents=documents,
        threshold=threshold,
        detail=True,
    )

    print(f"--- Filtered Evidence (passed threshold >= {threshold}) ---")
    # Expected output: Passages ordered from highest evidential quality to lowest.
    # Direct answer documents (Doc #0 and Doc #1) will rank at the top.
    for item in response["results"]:
        print(
            f"[PASS] [Score: {item['score']:.4f}] (Doc #{item['document_index']}): {item['text']}"
        )

    # -------------------------------------------------------------------------
    # Diagnostic Details:
    # LayaRank provides execution metadata and threshold selection statistics.
    # -------------------------------------------------------------------------
    print("\n--- Diagnostic Details ---")
    selection = response["detail"]["selection"]
    print(f"Total Evaluated : {selection['input_count']}")
    print(f"Above Threshold : {selection['above_threshold_count']}")
    print(f"Returned Count  : {selection['returned_count']}")
    print(f"Elapsed Time    : {response['detail']['timing_ms']:.2f} ms")


if __name__ == "__main__":
    main()
