#!/usr/bin/env python3
"""Load embedded chunks into Qdrant SERVER (localhost:6333). 
   No local-mode RAM explosion."""
import json, os, gc, sys, time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector, HnswConfigDiff,
)

QDRANT_COLLECTION = "law_ru"
BATCH = 2048
DENSE_DIM = 1024

client = QdrantClient(host="localhost", port=6333, timeout=120)

# Ensure collection
collections = [c.name for c in client.get_collections().collections]
if QDRANT_COLLECTION not in collections:
    print(f"Creating collection {QDRANT_COLLECTION}...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True)),
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10000),
    )
else:
    print(f"Collection {QDRANT_COLLECTION} exists")

existing = client.count(collection_name=QDRANT_COLLECTION, exact=True).count
print(f"Existing points: {existing}")

def load_file(path):
    global existing
    print(f"\nLoading {path}...")
    batch = []
    uploaded = 0
    t0 = time.time()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = data.get("id", "")
            if not chunk_id: continue
            dense = data.get("dense")
            sparse_data = data.get("sparse", {})
            if not dense or not sparse_data: continue
            point_id = abs(hash(chunk_id)) % (2**63 - 1) or 1
            batch.append(PointStruct(
                id=point_id,
                vector={
                    "dense": dense,
                    "sparse": SparseVector(
                        indices=sparse_data.get("indices", []),
                        values=[float(v) for v in sparse_data.get("values", [])],
                    ),
                },
                payload={
                    "id": chunk_id,
                    "parent_id": data.get("parent_id", ""),
                    "text": data.get("text", ""),
                    "chunk_index": data.get("chunk_index", 0),
                    "total_chunks": data.get("total_chunks", 0),
                    "category": data.get("category", ""),
                    "code_name": data.get("code_name", ""),
                    "active": data.get("active", False),
                    "timestamp": data.get("timestamp", ""),
                },
            ))
            if len(batch) >= BATCH:
                client.upsert(collection_name=QDRANT_COLLECTION, points=batch, wait=False)
                uploaded += len(batch)
                elapsed = time.time() - t0
                print(f"  {uploaded} points ({uploaded/elapsed:.0f} pts/s)", flush=True)
                batch = []
                gc.collect()
    if batch:
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch, wait=True)
        uploaded += len(batch)
    total = client.count(collection_name=QDRANT_COLLECTION, exact=True).count
    existing = total
    elapsed = time.time() - t0
    print(f"Done: {uploaded} points, collection total: {total} ({elapsed:.0f}s)")

if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        load_file(f)
