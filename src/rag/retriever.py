"""
Week 4, Day 3 — Semantic Search (Retrieval).
Takes a damage finding, embeds it, and searches FAISS for matching rules.
"""

import os
import numpy as np
import faiss
from embedder import get_cloud_embedding, process_policy_document


def load_vector_store(store_path="src/rag/data/faiss.index"):
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"FAISS index not found at {store_path}")
    return faiss.read_index(store_path)


def search_policy(query, index, chunks, top_k=1):
    print(f"\nSearching policy for: '{query}'")

    # 1. Convert the text query into a vector using the exact same Cloud API
    query_vector = get_cloud_embedding(query)
    query_array = np.array([query_vector]).astype("float32")

    # 2. Search FAISS for the closest matching policy vector
    distances, indices = index.search(query_array, top_k)

    # 3. Retrieve the matching text chunk
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])
            print(
                f"Match {i+1} (Distance: {distances[0][i]:.4f}): {chunks[idx][:75]}..."
            )

    return results


if __name__ == "__main__":
    sample_file = "src/rag/data/policy_CLM-TEST-001.txt"

    # Load the text chunks and the FAISS database
    chunks = process_policy_document(sample_file)
    index = load_vector_store()

    # Simulate a finding coming from our CV Damage JSON
    cv_query = "dent on the front bumper"

    # Retrieve the top matching policy rule
    matched_rules = search_policy(cv_query, index, chunks, top_k=1)

    print("\n--- RETRIEVED POLICY RULE ---")
    if matched_rules:
        print(matched_rules[0])
