import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(
    "data/faiss/knowledge.index"
)

with open(
    "data/faiss/documents.json",
    "r",
    encoding="utf-8"
) as f:
    docs = json.load(f)

query = input("Ask: ")

embedding = model.encode(
    [query],
    convert_to_numpy=True
)

distance, indices = index.search(
    embedding.astype(np.float32),
    1
)

print("\nBest Match\n")

print(docs[indices[0][0]]["text"])