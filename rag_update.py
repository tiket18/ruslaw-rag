#!/opt/vllm/.venv/bin/python3
"""RAG update pipeline for Russian legal documents (RusLawOD).

Modes:
  --full         Full pipeline: download → filter → chunk → embed → qdrant
  --update       Incremental: compare timestamps, reprocess changed docs only
  --status       Show current state
  --clean        Remove all generated data

Step flags (use with --full or standalone):
  --download     Download raw parquet from HuggingFace
  --filter       Filter documents (codes + government + constitutional court)
  --chunk        Split into chunks (1500 chars, 200 overlap)
  --embed        GPU embedding with BGEM3FlagModel
  --qdrant       Load into Qdrant
"""
import argparse
import json
import os
import sys
import gc
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

HF_REPO = "irlspbru/RusLawOD"
HF_FILES = [f"ruslawod_{i:02d}.parquet" for i in range(1, 12)]

RAW_DIR = "/home/vllm/rag/raw"
FILTERED_FILE = "/home/vllm/rag/filtered/filtered.jsonl"
CHUNKS_DIR = "/home/vllm/rag/chunks"
CHUNKS_FILE = "/home/vllm/rag/chunks/law_chunks.jsonl"
EMB_DIR = "/home/vllm/rag/embeddings"
EMB_FILE = "/home/vllm/rag/embeddings/law_chunks_emb.jsonl"
CHECKPOINT_FILE = "/home/vllm/rag/embeddings/.checkpoint.json"
STATE_FILE = "/home/vllm/rag/state.json"
TIMESTAMP_FILE = "/home/vllm/rag/doc_timestamps.json"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBED_BATCH = 256
CHECKPOINT_INTERVAL = 10
EMBED_MODEL = "Roflmax/bge-m3-legal-ru-cocktail-40-60"
EMBED_MAX_LENGTH = 8192
GPU_MEM_FRACTION = 0.85

# ── Filters ──────────────────────────────────────────────────────────────────

CODE_NAMES = {
    "Гражданский кодекс": "ГК",
    "Налоговый кодекс": "НК",
    "Трудовой кодекс": "ТК",
    "Жилищный кодекс": "ЖК",
    "Семейный кодекс": "СК",
    "Кодекс об административных правонарушениях": "КоАП",
    "Уголовный кодекс": "УК",
    "Уголовно-процессуальный кодекс": "УПК",
    "Арбитражный процессуальный кодекс": "АПК",
    "Гражданский процессуальный кодекс": "ГПК",
    "Кодекс административного судопроизводства": "КАС",
    "Бюджетный кодекс": "БК",
    "Земельный кодекс": "ЗК",
    "Лесной кодекс": "ЛК",
    "Водный кодекс": "ВК",
    "Градостроительный кодекс": "ГрК",
    "Воздушный кодекс": "ВзК",
    "Кодекс торгового мореплавания": "КТМ",
    "Таможенный кодекс": "ТмК",
}

FZ_TYPES = {
    "Федеральный закон",
    "Федеральный конституционный закон",
}

GOV_ISSUERS = {
    "Постановление Правительства Российской Федерации",
    "Распоряжение Правительства Российской Федерации",
}

KS_ISSUERS = {
    "Определение Конституционного Суда Российской Федерации",
    "Постановление Конституционного Суда Российской Федерации",
}

# ── Utils ────────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "version": 1,
        "last_full_update": None,
        "last_update_check": None,
        "source": HF_REPO,
        "docs_total": 0,
        "docs_filtered": 0,
        "chunks_total": 0,
        "embeddings_total": 0,
        "collections": ["law_ru"],
        "filters": {
            "codes": True,
            "federal_laws": True,
            "government": True,
            "constitutional_court": True,
        }
    }

def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)

def load_timestamps():
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE) as f:
            return json.load(f)
    return {}

def save_timestamps(timestamps):
    tmp = TIMESTAMP_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(timestamps, f, ensure_ascii=False)
    os.replace(tmp, TIMESTAMP_FILE)

def step_done(path, min_size=1000):
    return os.path.exists(path) and os.path.getsize(path) > min_size

# ── Step 1: Download ─────────────────────────────────────────────────────────

def step_download():
    from huggingface_hub import hf_hub_download
    ensure_dir(RAW_DIR)
    downloaded = 0
    for fname in HF_FILES:
        dst = os.path.join(RAW_DIR, fname)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            log(f"[SKIP] {fname} exists ({os.path.getsize(dst)//1024//1024} MB)")
            continue
        t0 = time.time()
        log(f"[DOWNLOAD] {fname}...")
        path = hf_hub_download(HF_REPO, fname, repo_type="dataset",
                               local_dir=RAW_DIR, local_dir_use_symlinks=False)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(path) / 1024 / 1024
        log(f"[DONE] {fname} ({size_mb:.0f} MB in {elapsed:.0f}s, {size_mb/elapsed:.1f} MB/s)")
        downloaded += 1
    log(f"[OK] Download complete, {len(HF_FILES)} files in {RAW_DIR}")
    return True

# ── Step 2: Filter ───────────────────────────────────────────────────────────

def filter_doc(doc):
    dtype = str(doc.get("doc_typeIPS") or "")
    issuer = str(doc.get("issuedByIPS") or "")
    heading = str(doc.get("headingIPS") or "")
    text = str(doc.get("textIPS") or "")
    status = str(doc.get("statusIPS") or "")

    if not text or len(text) < 100:
        return None

    h = heading.lower()

    # 1. Codes: doc_type == "Кодекс", text > 5000, not amendments
    if dtype == "Кодекс" and len(text) > 5000 and "внесен" not in h:
        active = status and "Действует" in status
        code_name = None
        for name, short in CODE_NAMES.items():
            if name.lower() in h:
                code_name = short
                break
        return {
            "id": doc.get("idIPS") or str(doc.get("docNumberIPS", "")),
            "doc_type": dtype,
            "issuer": issuer,
            "heading": heading,
            "doc_number": str(doc.get("docNumberIPS") or ""),
            "doc_date": str(doc.get("docdateIPS") or ""),
            "status": status,
            "active": active,
            "text": text,
            "text_len": len(text),
            "category": "code",
            "code_name": code_name,
            "timestamp": str(doc.get("actual_datetimeIPS") or ""),
        }

    # 2. Federal laws (original, not amendments)
    if dtype in FZ_TYPES and "внесен" not in h and len(text) > 3000:
        return {
            "id": doc.get("idIPS") or str(doc.get("docNumberIPS", "")),
            "doc_type": dtype,
            "issuer": issuer,
            "heading": heading,
            "doc_number": str(doc.get("docNumberIPS") or ""),
            "doc_date": str(doc.get("docdateIPS") or ""),
            "status": status,
            "active": status and "Действует" in status,
            "text": text,
            "text_len": len(text),
            "category": "fz",
            "code_name": None,
            "timestamp": str(doc.get("actual_datetimeIPS") or ""),
        }

    # 3. Government regulations (exact issuer match)
    if issuer in GOV_ISSUERS:
        return {
            "id": doc.get("idIPS") or str(doc.get("docNumberIPS", "")),
            "doc_type": dtype,
            "issuer": issuer,
            "heading": heading,
            "doc_number": str(doc.get("docNumberIPS") or ""),
            "doc_date": str(doc.get("docdateIPS") or ""),
            "status": status,
            "active": status and "Действует" in status,
            "text": text,
            "text_len": len(text),
            "category": "gov",
            "code_name": None,
            "timestamp": str(doc.get("actual_datetimeIPS") or ""),
        }

    # 4. Constitutional Court (exact issuer match)
    if issuer in KS_ISSUERS:
        return {
            "id": doc.get("idIPS") or str(doc.get("docNumberIPS", "")),
            "doc_type": dtype,
            "issuer": issuer,
            "heading": heading,
            "doc_number": str(doc.get("docNumberIPS") or ""),
            "doc_date": str(doc.get("docdateIPS") or ""),
            "status": status,
            "active": status and "Действует" in status,
            "text": text,
            "text_len": len(text),
            "category": "ks",
            "code_name": None,
            "timestamp": str(doc.get("actual_datetimeIPS") or ""),
        }

    return None

def step_filter():
    if step_done(FILTERED_FILE):
        lines = sum(1 for _ in open(FILTERED_FILE))
        log(f"[SKIP] filter already done ({lines} docs)")
        return True

    import pyarrow.parquet as pq
    ensure_dir(os.path.dirname(FILTERED_FILE))

    files = sorted(Path(RAW_DIR).glob("ruslawod_*.parquet"))
    if not files:
        log("[ERROR] No parquet files found. Run --download first.")
        return False

    total = 0
    filtered = 0
    t0 = time.time()

    with open(FILTERED_FILE, "w") as fout:
        for pf in files:
            tbl = pq.read_table(str(pf))
            for i in range(tbl.num_rows):
                total += 1
                doc = {col: tbl.column(col)[i].as_py() for col in tbl.column_names}
                result = filter_doc(doc)
                if result:
                    filtered += 1
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            log(f"[PROGRESS] {pf.name}: {total} total, {filtered} filtered")

    elapsed = time.time() - t0
    log(f"[OK] Filtered {filtered}/{total} docs in {elapsed:.0f}s")

    state = load_state()
    state["docs_total"] = total
    state["docs_filtered"] = filtered
    save_state(state)
    return True

# ── Step 3: Chunk ────────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

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
    if step_done(CHUNKS_FILE):
        lines = sum(1 for _ in open(CHUNKS_FILE))
        log(f"[SKIP] chunk already done ({lines} chunks)")
        return True

    ensure_dir(CHUNKS_DIR)

    if not os.path.exists(FILTERED_FILE):
        log("[ERROR] No filtered data. Run --filter first.")
        return False

    total_chunks = 0
    doc_id = 0
    t0 = time.time()

    with open(FILTERED_FILE) as fin, open(CHUNKS_FILE, "w") as fout:
        for line in fin:
            doc = json.loads(line)
            chunks = chunk_text(doc["text"])
            for ci, ctext in enumerate(chunks):
                fout.write(json.dumps({
                    "id": f"chunk_{doc_id}_{ci}",
                    "parent_id": doc.get("id", f"doc_{doc_id}"),
                    "doc_type": doc.get("doc_type", ""),
                    "issuer": doc.get("issuer", ""),
                    "heading": doc.get("heading", ""),
                    "doc_number": doc.get("doc_number", ""),
                    "doc_date": doc.get("doc_date", ""),
                    "status": doc.get("status", ""),
                    "active": doc.get("active", False),
                    "category": doc.get("category", ""),
                    "code_name": doc.get("code_name", ""),
                    "timestamp": doc.get("timestamp", ""),
                    "text": ctext,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                }, ensure_ascii=False) + "\n")
                total_chunks += 1
            doc_id += 1

    elapsed = time.time() - t0
    size_mb = os.path.getsize(CHUNKS_FILE) / 1024 / 1024 if os.path.exists(CHUNKS_FILE) else 0
    log(f"[OK] {total_chunks} chunks from {doc_id} docs ({size_mb:.0f} MB) in {elapsed:.0f}s")

    state = load_state()
    state["chunks_total"] = total_chunks
    save_state(state)
    return True

# ── Step 4: Embed ────────────────────────────────────────────────────────────

shutdown_requested = False
def signal_handler(signum, frame):
    global shutdown_requested
    log("Signal received, finishing batch...")
    shutdown_requested = True

def to_json_serializable(obj):
    import numpy as np
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    return obj

def get_embed_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                cp = json.load(f)
                return cp.get("last_id"), cp.get("written", 0)
        except Exception:
            pass
    if os.path.exists(EMB_FILE):
        try:
            with open(EMB_FILE) as f:
                last = None
                count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        last = d.get("id")
                        count += 1
                    except json.JSONDecodeError:
                        continue
                if last:
                    return last, count
        except Exception:
            pass
    return None, 0

def save_embed_checkpoint(last_id, written):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"last_id": last_id, "written": written}, f)
    os.replace(tmp, CHECKPOINT_FILE)

def step_embed():
    resume_id, already_written = get_embed_checkpoint()
    if already_written > 0:
        log(f"[SKIP] embed already in progress ({already_written} done), resuming")
    elif step_done(EMB_FILE):
        lines = sum(1 for _ in open(EMB_FILE))
        log(f"[SKIP] embed already done ({lines} chunks)")
        return True

    import torch
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    global shutdown_requested
    shutdown_requested = False

    ensure_dir(EMB_DIR)

    if not os.path.exists(CHUNKS_FILE):
        log("[ERROR] No chunks. Run --chunk first.")
        return False

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HOME"] = "/home/vllm/huggingface_cache"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)

    log(f"Loading model {EMBED_MODEL}...")
    model = BGEM3FlagModel(EMBED_MODEL, device="cuda", use_fp16=True)
    log("Model loaded")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    resume_id, already_written = get_embed_checkpoint()
    log(f"Resume from: {resume_id}, already written: {already_written}")

    written = already_written
    batch_texts = []
    batch_docs = []
    batch_count = 0
    skipping = resume_id is not None

    with open(CHUNKS_FILE) as fin:
        for line in fin:
            if shutdown_requested:
                log("Shutdown requested, saving checkpoint...")
                break

            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk_id = doc.get("id", "")

            if skipping:
                if chunk_id == resume_id:
                    log(f"Resuming at {chunk_id}")
                    skipping = False
                else:
                    continue

            batch_texts.append(doc["text"])
            batch_docs.append(doc)

            if len(batch_texts) >= EMBED_BATCH:
                outputs = model.encode(
                    batch_texts,
                    batch_size=EMBED_BATCH,
                    max_length=EMBED_MAX_LENGTH,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                with open(EMB_FILE, "a") as fout:
                    for i, d in enumerate(batch_docs):
                        sparse = outputs["lexical_weights"][i]
                        fout.write(json.dumps({
                            "id": d["id"],
                            "parent_id": d["parent_id"],
                            "text": d["text"],
                            "chunk_index": d.get("chunk_index", 0),
                            "total_chunks": d.get("total_chunks", 0),
                            "category": d.get("category", ""),
                            "code_name": d.get("code_name", ""),
                            "active": d.get("active", False),
                            "timestamp": d.get("timestamp", ""),
                            "dense": outputs["dense_vecs"][i].tolist(),
                            "sparse": {
                                "indices": list(sparse.keys()),
                                "values": [float(v) for v in sparse.values()],
                            },
                        }, ensure_ascii=False) + "\n")

                written += len(batch_docs)
                batch_count += 1
                batch_texts.clear()
                batch_docs.clear()

                if batch_count % CHECKPOINT_INTERVAL == 0:
                    save_embed_checkpoint(batch_docs[-1]["id"] if batch_docs else None, written)
                    log(f"Checkpoint: {written} chunks embedded")
                    gc.collect()
                    torch.cuda.empty_cache()

    if batch_texts:
        outputs = model.encode(
            batch_texts,
            batch_size=EMBED_BATCH,
            max_length=EMBED_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        with open(EMB_FILE, "a") as fout:
            for i, d in enumerate(batch_docs):
                sparse = outputs["lexical_weights"][i]
                fout.write(json.dumps({
                    "id": d["id"],
                    "parent_id": d["parent_id"],
                    "text": d["text"],
                    "chunk_index": d.get("chunk_index", 0),
                    "total_chunks": d.get("total_chunks", 0),
                    "category": d.get("category", ""),
                    "code_name": d.get("code_name", ""),
                    "active": d.get("active", False),
                    "timestamp": d.get("timestamp", ""),
                    "dense": outputs["dense_vecs"][i].tolist(),
                    "sparse": {
                        "indices": list(sparse.keys()),
                        "values": [float(v) for v in sparse.values()],
                    },
                }, ensure_ascii=False) + "\n")
        written += len(batch_docs)

    save_embed_checkpoint(None, written)
    log(f"[OK] Embedding complete. Total: {written} chunks")

    state = load_state()
    state["embeddings_total"] = written
    save_state(state)
    return True

# ── Step 5: Load Qdrant ──────────────────────────────────────────────────────

QDRANT_PATH = "/home/vllm/rag/qdrant_db"
QDRANT_COLLECTION = "law_ru"
QDRANT_BATCH = 512
DENSE_DIM = 1024


def step_qdrant(input_file=None):
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, SparseVectorParams, SparseIndexParams,
        PointStruct, SparseVector, HnswConfigDiff,
    )

    emb_file = input_file or EMB_FILE
    log(f"Input file: {emb_file}")

    if not os.path.exists(emb_file):
        log(f"[ERROR] No embeddings file at {emb_file}")
        return False

    # — Connect to Qdrant (local persistent)
    ensure_dir(QDRANT_PATH)
    client = QdrantClient(path=QDRANT_PATH)
    log(f"Connected to Qdrant (local: {QDRANT_PATH})")

    # — Create collection
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if QDRANT_COLLECTION not in existing:
        log(f"Creating collection '{QDRANT_COLLECTION}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True),
                ),
            },
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            ),
        )
        log("Collection created")
    else:
        log(f"Collection '{QDRANT_COLLECTION}' already exists")

    # — Count existing points for resume
    collection_info = client.get_collection(QDRANT_COLLECTION)
    existing_count = collection_info.points_count
    log(f"Existing points: {existing_count}")

    # — Read embeddings and upload
    total_processed = 0
    total_uploaded = 0
    batch = []
    t0 = time.time()

    with open(emb_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk_id = data.get("id", "")
            if not chunk_id:
                continue

            dense = data.get("dense")
            sparse_data = data.get("sparse", {})
            if not dense or not sparse_data:
                continue

            total_processed += 1

            payload = {
                "id": chunk_id,
                "parent_id": data.get("parent_id", ""),
                "text": data.get("text", ""),
                "chunk_index": data.get("chunk_index", 0),
                "total_chunks": data.get("total_chunks", 0),
                "category": data.get("category", ""),
                "code_name": data.get("code_name", ""),
                "active": data.get("active", False),
                "timestamp": data.get("timestamp", ""),
            }

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
                payload=payload,
            ))

            if len(batch) >= QDRANT_BATCH:
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=batch,
                    wait=False,
                )
                total_uploaded += len(batch)
                elapsed = time.time() - t0
                rate = total_uploaded / elapsed if elapsed > 0 else 0
                log(f"Uploaded {total_uploaded} points ({total_uploaded/elapsed:.0f} pts/s)")
                batch.clear()
                gc.collect()

    # Final batch
    if batch:
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
            wait=True,
        )
        total_uploaded += len(batch)

    elapsed = time.time() - t0
    log(f"[OK] Qdrant load complete: {total_uploaded} points in {elapsed:.0f}s "
        f"({total_uploaded/elapsed:.0f} pts/s)")

    # Final collection info
    info = client.get_collection(QDRANT_COLLECTION)
    vectors = getattr(info, 'vectors', None)
    vector_count = vectors.get('dense', {}).get('count', 0) if vectors else '?'
    log(f"Collection '{QDRANT_COLLECTION}': {info.points_count} points, "
        f"{vector_count} vectors")

    return True

# ── Update Mode ──────────────────────────────────────────────────────────────

def step_update():
    log("[TODO] Update mode not yet implemented")
    return True

# ── Status ────────────────────────────────────────────────────────────────────

def step_status():
    state = load_state()
    print()
    print("=" * 60)
    print(f"  Source: {state['source']}")
    print(f"  Last full update: {state['last_full_update'] or 'never'}")
    print(f"  Last update check: {state['last_update_check'] or 'never'}")
    print(f"  Total docs: {state['docs_total']}")
    print(f"  Filtered docs: {state['docs_filtered']}")
    print(f"  Chunks: {state['chunks_total']}")
    print(f"  Embeddings: {state['embeddings_total']}")
    print(f"  Collections: {', '.join(state['collections'])}")
    flt = state.get("filters", {})
    print(f"  Filters: codes={flt.get('codes')}, fz={flt.get('federal_laws')}, gov={flt.get('government')}, ks={flt.get('constitutional_court')}")
    print()
    for path, label in [
        (RAW_DIR, "Raw parquet"),
        (FILTERED_FILE, "Filtered docs"),
        (CHUNKS_FILE, "Chunks"),
        (EMB_FILE, "Embeddings"),
    ]:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024 if os.path.isfile(path) else 0
            if os.path.isdir(path):
                size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024 / 1024
            print(f"  {label}: {path} ({size:.0f} MB)")
        else:
            print(f"  {label}: {path} (not found)")
    print("=" * 60)

# ── Clean ────────────────────────────────────────────────────────────────────

def step_clean():
    log("[CLEAN] Removing all generated data...")
    for path in [FILTERED_FILE, CHUNKS_FILE, EMB_FILE, CHECKPOINT_FILE, STATE_FILE, TIMESTAMP_FILE]:
        if os.path.exists(path):
            os.remove(path)
            log(f"  Removed {path}")
    for d in [os.path.dirname(FILTERED_FILE), CHUNKS_DIR, EMB_DIR]:
        if os.path.exists(d):
            remaining = list(Path(d).rglob("*"))
            if not remaining:
                os.rmdir(d)
                log(f"  Removed empty dir {d}")
    log("[CLEAN] Done")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG pipeline for Russian legal documents (RusLawOD)"
    )
    parser.add_argument("--full", action="store_true", help="Full pipeline")
    parser.add_argument("--update", action="store_true", help="Incremental update")
    parser.add_argument("--status", action="store_true", help="Show state")
    parser.add_argument("--clean", action="store_true", help="Remove all generated data")
    parser.add_argument("--download", action="store_true", help="Download only")
    parser.add_argument("--filter", action="store_true", help="Filter only")
    parser.add_argument("--chunk", action="store_true", help="Chunk only")
    parser.add_argument("--embed", action="store_true", help="Embed only")
    parser.add_argument("--qdrant", action="store_true", help="Load Qdrant only")
    parser.add_argument("--qdrant-file", help="Input file for --qdrant (default: law_chunks_emb.jsonl)")
    args = parser.parse_args()

    if args.status:
        step_status()
        return

    if args.clean:
        step_clean()
        return

    if args.update:
        step_update()
        return

    qdrant_file = getattr(args, "qdrant_file", None)

    if args.full or not any([args.download, args.filter, args.chunk, args.embed, args.qdrant]):
        steps = [
            ("download", step_download),
            ("filter", step_filter),
            ("chunk", step_chunk),
            ("embed", step_embed),
            ("qdrant", lambda: step_qdrant(qdrant_file)),
        ]
        for name, step_fn in steps:
            log(f"[STEP] {name}")
            if not step_fn():
                log(f"[FAILED] {name}")
                sys.exit(1)
            log(f"[DONE] {name}")
        state = load_state()
        state["last_full_update"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        return

    step_map = {
        "download": step_download,
        "filter": step_filter,
        "chunk": step_chunk,
        "embed": step_embed,
        "qdrant": lambda: step_qdrant(qdrant_file),
    }
    for name, step_fn in step_map.items():
        if getattr(args, name):
            log(f"[STEP] {name}")
            if not step_fn():
                log(f"[FAILED] {name}")
                sys.exit(1)

if __name__ == "__main__":
    main()
