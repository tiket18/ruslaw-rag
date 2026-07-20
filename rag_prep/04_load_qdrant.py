#!/usr/bin/env python3
"""
Load embedded law chunks into Qdrant.
Input: /home/vllm/rag/raw/embeddings/*.jsonl
"""
import json
import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct,
    SparseVector, PayloadSchemaType
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "law_ru"
EMB_DIR = Path("/home/vllm/rag/raw/embeddings")
BATCH_SIZE = 100

def main():
    if not EMB_DIR.exists():
        print(f"ERROR: {EMB_DIR} does not exist. Run 03_embed_laws.py first.", file=sys.stderr)
        sys.exit(1)

    client = QdrantClient(url=QDRANT_URL, timeout=60)

    # Create collection if not exists
    collections = client.get_collections().collections
    if COLLECTION not in [c.name for c in collections]:
        print(f"Creating collection {COLLECTION}...")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index={"on_disk": True}),
            },
        )
    else:
        print(f"Collection {COLLECTION} exists, upserting...")

    # Payload indexes
    for field in ["source", "code_name", "article"]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    # Load and upsert
    for src_file in sorted(EMB_DIR.glob("*.jsonl")):
        print(f"Loading {src_file.name}...")
        points = []
        count = 0

        with open(src_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                point = PointStruct(
                    id=rec["id"],
                    vector={
                        "dense": rec["dense"],
                        "sparse": SparseVector(
                            indices=rec["sparse"]["indices"],
                            values=rec["sparse"]["values"],
                        ),
                    },
                    payload={
                        "text": rec["text"],
                        "source": rec["source"],
                        "code_name": rec["code_name"],
                        "article": rec["article"],
                        "title": rec["title"],
                        "chunk_index": rec["chunk_index"],
                        "total_chunks": rec["total_chunks"],
                        "parent_id": rec["parent_id"],
                        "meta": rec.get("meta", {}),
                    },
                )
                points.append(point)
                count += 1

                if len(points) >= BATCH_SIZE:
                    client.upsert(collection_name=COLLECTION, points=points, wait=True)
                    points = []

        if points:
            client.upsert(collection_name=COLLECTION, points=points, wait=True)

        print(f"  Upserted {count} points from {src_file.name}")

    # Verify
    total = client.count(collection_name=COLLECTION, exact=True).count
    print(f"\nTotal points in {COLLECTION}: {total}")

if __name__ == "__main__":
    main()