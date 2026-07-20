#!/usr/bin/env python3
"""
Generate bge-m3 embeddings (dense + sparse) for law chunks.
Input: /home/vllm/rag/raw/jsonl/*.jsonl
Output: /home/vllm/rag/raw/embeddings/*.jsonl
"""
import json
import os
import torch
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel

JSONL_DIR = Path("/home/vllm/rag/raw/jsonl")
EMB_DIR = Path("/home/vllm/rag/raw/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
MAX_LENGTH = 8192

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading bge-m3 on {device}...")
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        device=device,
        use_fp16=(device == "cuda"),
    )
    return model

def sparse_to_dict(sparse_vec: dict) -> dict:
    """Convert bge-m3 sparse output to qdrant format."""
    return {
        "indices": list(sparse_vec.keys()),
        "values": list(sparse_vec.values()),
    }

def process_file(model, in_file: Path, out_file: Path):
    records = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"  {in_file.name}: empty")
        return 0

    texts = [r["text"] for r in records]
    print(f"  Encoding {len(texts)} chunks...")

    outputs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vecs = outputs["dense_vecs"]
    sparse_vecs = outputs["lexical_weights"]

    with open(out_file, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            out_rec = {
                "id": rec["id"],
                "dense": dense_vecs[i].tolist(),
                "sparse": sparse_to_dict(sparse_vecs[i]),
                "text": rec["text"],
                "source": rec["source"],
                "code_name": rec["code_name"],
                "article": rec["article"],
                "title": rec["title"],
                "chunk_index": rec["chunk_index"],
                "total_chunks": rec["total_chunks"],
                "parent_id": rec["parent_id"],
                "meta": rec["meta"],
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    return len(records)

def main():
    model = load_model()
    print("Generating embeddings...")

    total = 0
    for in_file in sorted(JSONL_DIR.glob("*.jsonl")):
        out_file = EMB_DIR / in_file.name
        print(f"Processing {in_file.name}...")
        count = process_file(model, in_file, out_file)
        total += count
        print(f"  Done: {count} embeddings")

    print(f"\nTotal embedded: {total}")

if __name__ == "__main__":
    main()