#!/usr/bin/env python3
"""RAG helper: embed(BGE-M3) → search(Qdrant) → output context JSON.
Does NOT kill llama-server — shares GPU with running LLM."""
import json, sys, os
from datetime import datetime

os.environ.setdefault("HF_HOME", "/home/vllm/huggingface_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

QUERY = sys.argv[1] if len(sys.argv) > 1 else ""
TOP_K = 5

from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("Roflmax/bge-m3-legal-ru-cocktail-40-60", device="cuda", use_fp16=True)

output = model.encode([QUERY], return_dense=True, return_sparse=True, max_length=8192)
dense = output["dense_vecs"][0]

from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333, timeout=30)
result = client.query_points(
    collection_name="law_ru",
    query=dense.tolist(),
    using="dense",
    with_payload=True,
    limit=TOP_K,
)

CATEGORY_LABELS = {
    "ks": "Конституционный Суд РФ",
    "code": "Кодекс РФ",
    "fz": "Федеральный закон",
    "gov": "Постановление Правительства РФ",
    "plenum": "Пленум ВС РФ",
}

docs = []
for r in result.points:
    p = r.payload or {}
    cat = p.get("category", "")
    code_name = p.get("code_name", "")
    parent_id = p.get("parent_id", "")
    ts = p.get("timestamp")
    if code_name == "None":
        code_name = ""

    date_str = ""
    if ts:
        try:
            date_str = datetime.fromtimestamp(float(ts)).strftime("%d.%m.%Y")
        except (ValueError, OSError):
            date_str = ""

    label = CATEGORY_LABELS.get(cat, cat)
    source_parts = [label]
    if parent_id and parent_id != "None":
        source_parts.append(f"№ {parent_id}")
    if code_name:
        source_parts.append(code_name)
    if date_str:
        source_parts.append(f"от {date_str}")

    doc_number = parent_id if parent_id and parent_id != "None" else ""
    search_q = f"{label} {doc_number}".strip()
    if search_q:
        consultant_url = f"https://www.consultant.ru/search/?q={search_q.replace(' ', '+')}"
    else:
        consultant_url = ""

    docs.append({
        "score": round(r.score, 4),
        "text": p.get("text", "")[:1500],
        "source": ", ".join(source_parts),
        "date": date_str,
        "category": cat,
        "code_name": code_name,
        "parent_id": doc_number,
        "active": p.get("active", False),
        "consultant_url": consultant_url,
    })
print(json.dumps({"query": QUERY, "docs": docs, "count": len(docs)}, ensure_ascii=False))
