#!/usr/bin/env python3
"""Chunk + Embed + Load Plenum Supreme Court decisions into Qdrant."""
import json, os, gc, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/home/vllm/huggingface_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/vllm/huggingface_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/home/vllm/huggingface_cache")

# ── Config ──
PLENUM_INPUT = "/home/vllm/rag/plenum_output/plenum_clean.jsonl"
CHUNKS_OUT = "/home/vllm/rag/plenum_output/plenum_chunks.jsonl"
EMB_OUT = "/home/vllm/rag/plenum_output/plenum_emb.jsonl"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBED_BATCH = 256
EMBED_MODEL = "Roflmax/bge-m3-legal-ru-cocktail-40-60"
COLLECTION = "law_ru"
QDRANT_BATCH = 512

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Step 1: Chunk ──
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def step_chunk():
    if os.path.exists(CHUNKS_OUT) and os.path.getsize(CHUNKS_OUT) > 1000:
        log(f"[SKIP] chunks exist ({os.path.getsize(CHUNKS_OUT)//1024} KB)")
        return True
    log("Chunking plenum documents...")
    total = 0
    doc_count = 0
    with open(PLENUM_INPUT) as fin, open(CHUNKS_OUT, "w") as fout:
        for line in fin:
            doc = json.loads(line)
            chunks = chunk_text(doc["text"])
            for ci, ctext in enumerate(chunks):
                fout.write(json.dumps({
                    "id": f"plenum_{doc['id']}_{ci}",
                    "parent_id": doc["id"],
                    "text": ctext,
                    "source": "plenum_vs",
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                    "title": doc.get("title", ""),
                    "year": doc.get("year", ""),
                    "doc_date": doc.get("date", ""),
                    "doc_number": doc.get("number", ""),
                }, ensure_ascii=False) + "\n")
                total += 1
            doc_count += 1
    size = os.path.getsize(CHUNKS_OUT) / 1024
    log(f"OK: {total} chunks from {doc_count} docs ({size:.0f} KB)")
    return True

# ── Step 2: Embed ──
def step_embed():
    if os.path.exists(EMB_OUT) and os.path.getsize(EMB_OUT) > 1000:
        log(f"[SKIP] embeddings exist ({os.path.getsize(EMB_OUT)//1024} KB)")
        return True
    import torch
    from FlagEmbedding import BGEM3FlagModel

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)

    log("Loading BGE-M3 model for Plenum...")
    model = BGEM3FlagModel(EMBED_MODEL, device="cuda", use_fp16=True)
    log("Model loaded")

    written = 0
    batch_texts = []
    batch_docs = []
    t0 = time.time()

    with open(CHUNKS_OUT) as fin:
        for line in fin:
            doc = json.loads(line)
            batch_texts.append(doc["text"])
            batch_docs.append(doc)
            if len(batch_texts) >= EMBED_BATCH:
                outputs = model.encode(
                    batch_texts, batch_size=EMBED_BATCH, max_length=8192,
                    return_dense=True, return_sparse=True, return_colbert_vecs=False,
                )
                with open(EMB_OUT, "a") as fout:
                    for i, d in enumerate(batch_docs):
                        sd = outputs["lexical_weights"][i]
                        fout.write(json.dumps({
                            "id": d["id"],
                            "parent_id": d["parent_id"],
                            "text": d["text"],
                            "chunk_index": d.get("chunk_index", 0),
                            "total_chunks": d.get("total_chunks", 0),
                            "category": "plenum_vs",
                            "code_name": "",
                            "active": True,
                            "timestamp": str(d.get("year", "")),
                            "dense": outputs["dense_vecs"][i].tolist(),
                            "sparse": {
                                "indices": list(sd.keys()),
                                "values": [float(v) for v in sd.values()],
                            },
                        }, ensure_ascii=False) + "\n")
                written += len(batch_docs)
                batch_texts.clear()
                batch_docs.clear()
                gc.collect()
                torch.cuda.empty_cache()

    if batch_texts:
        outputs = model.encode(
            batch_texts, batch_size=EMBED_BATCH, max_length=8192,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        with open(EMB_OUT, "a") as fout:
            for i, d in enumerate(batch_docs):
                sd = outputs["lexical_weights"][i]
                fout.write(json.dumps({
                    "id": d["id"],
                    "parent_id": d["parent_id"],
                    "text": d["text"],
                    "chunk_index": d.get("chunk_index", 0),
                    "total_chunks": d.get("total_chunks", 0),
                    "category": "plenum_vs",
                    "code_name": "",
                    "active": True,
                    "timestamp": str(d.get("year", "")),
                    "dense": outputs["dense_vecs"][i].tolist(),
                    "sparse": {
                        "indices": list(sd.keys()),
                        "values": [float(v) for v in sd.values()],
                    },
                }, ensure_ascii=False) + "\n")
        written += len(batch_docs)

    elapsed = time.time() - t0
    log(f"OK: {written} chunks embedded in {elapsed:.0f}s ({written/elapsed:.0f} ch/s)")
    return True

# ── Step 3: Load Qdrant ──
def step_qdrant():
    from qdrant_client import QdrantClient, models

    client = QdrantClient(host="localhost", port=6333, timeout=120)
    before = client.count(collection_name=COLLECTION, exact=True).count
    log(f"Collection before: {before} points")

    batch = []
    uploaded = 0
    t0 = time.time()

    with open(EMB_OUT) as f:
        for line in f:
            data = json.loads(line)
            cid = data["id"]
            dense = data["dense"]
            sd = data["sparse"]
            pid = abs(hash(cid)) % (2**63 - 1) or 1
            batch.append(models.PointStruct(
                id=pid,
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(indices=sd["indices"], values=[float(v) for v in sd["values"]]),
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
            if len(batch) >= QDRANT_BATCH:
                client.upsert(collection_name=COLLECTION, points=batch, wait=False)
                uploaded += len(batch)
                if uploaded % (QDRANT_BATCH * 10) == 0:
                    log(f"Uploaded {uploaded} points")
                batch = []
                gc.collect()

    if batch:
        client.upsert(collection_name=COLLECTION, points=batch, wait=True)
        uploaded += len(batch)

    after = client.count(collection_name=COLLECTION, exact=True).count
    elapsed = time.time() - t0
    log(f"OK: {uploaded} points loaded. Collection: {before} → {after} ({elapsed:.0f}s)")
    return True

# ── Main ──
if __name__ == "__main__":
    steps = [("chunk", step_chunk), ("embed", step_embed), ("qdrant", step_qdrant)]
    for name, fn in steps:
        log(f"[STEP] {name}")
        if not fn():
            log(f"[FAIL] {name}")
            sys.exit(1)
        log(f"[DONE] {name}")
