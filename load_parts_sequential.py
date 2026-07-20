#!/usr/bin/env python3
"""Load parts into Qdrant SERVER sequentially. One part at a time."""
import json, os, gc, sys, time
from qdrant_client import QdrantClient, models

COLLECTION = "law_ru"
BATCH = 512
DENSE_DIM = 1024
PARTS_DIR = "/home/vllm/rag/emb_parts"
PARTS = ["part_aa", "part_ab", "part_ac", "part_ad", "part_ae",
         "part_af", "part_ag", "part_ah", "part_ai", "part_aj", "part_ak"]

client = QdrantClient(host="localhost", port=6333, timeout=120)

def ensure_collection():
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        print(f"Creating collection {COLLECTION}...")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True)),
            },
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10000),
        )

def count_collection():
    try:
        return client.count(collection_name=COLLECTION, exact=True).count
    except Exception:
        return 0

def load_part(path):
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
            cid = data.get("id", "")
            if not cid: continue
            dense = data.get("dense")
            sparse_data = data.get("sparse", {})
            if not dense or not sparse_data: continue
            pid = abs(hash(cid)) % (2**63 - 1) or 1
            batch.append(models.PointStruct(
                id=pid,
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(
                        indices=sparse_data.get("indices", []),
                        values=[float(v) for v in sparse_data.get("values", [])],
                    ),
                },
                payload={
                    "id": cid,
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
                client.upsert(collection_name=COLLECTION, points=batch, wait=False)
                uploaded += len(batch)
                elapsed = time.time() - t0
                print(f"  {uploaded} pts ({uploaded/elapsed:.0f} pts/s)", flush=True)
                batch = []
                gc.collect()
    if batch:
        client.upsert(collection_name=COLLECTION, points=batch, wait=True)
        uploaded += len(batch)
    total = count_collection()
    elapsed = time.time() - t0
    print(f"  Done: {uploaded} uploaded, collection: {total} total ({elapsed:.0f}s)")
    return total

def main():
    ensure_collection()
    before = count_collection()
    print(f"Collection before: {before} points\n")

    for part in PARTS:
        path = os.path.join(PARTS_DIR, part)
        if not os.path.exists(path):
            print(f"[SKIP] {part} — not found")
            continue
        print(f"=== {part} ===")
        load_part(path)

    after = count_collection()
    print(f"\nBefore: {before}")
    print(f"After:  {after}")
    print(f"Added:  {after - before}")

if __name__ == "__main__":
    main()
