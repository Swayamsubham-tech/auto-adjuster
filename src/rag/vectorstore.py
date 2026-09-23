"""
Week 4, Day 2 — Vector Database (FAISS).
Takes text chunks, embeds them, and stores them in a searchable local index.
"""

import os
import numpy as np
import faiss
from embedder import process_policy_document, get_cloud_embedding


def create_vector_store(chunks, store_path="src/rag/data/faiss.index"):
    print(f"Creating vector store for {len(chunks)} chunks...")

    # 384 is the dimension size from our Day 1 embedder
    dimension = 384
    index = faiss.IndexFlatL2(dimension)

    vectors = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}...")
        vec = get_cloud_embedding(chunk)
        vectors.append(vec)

    # Convert list of vectors to a NumPy array (FAISS requirement)
    vector_array = np.array(vectors).astype("float32")

    # Add vectors to the FAISS index
    index.add(vector_array)

    # Save the index file to disk
    faiss.write_index(index, store_path)
    print(f"Successfully saved FAISS index with {index.ntotal} vectors to {store_path}")
    return index


if __name__ == "__main__":
    sample_file = "src/rag/data/policy_CLM-TEST-001.txt"
    if os.path.exists(sample_file):
        chunks = process_policy_document(sample_file)
        create_vector_store(chunks)
    else:
        print("Policy file not found. Please run embedder.py first.")
