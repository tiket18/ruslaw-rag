#!/usr/bin/env python3
"""RAG test: query → embed(BGE-M3) → search(Qdrant) → LLM"""
import json, sys, os, time

os.environ["HF_HOME"] = "/home/vllm/huggingface_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient

query = sys.argv[1] if len(sys.argv) > 1 else "Какая ответственность за неуплату налогов?"
TOP_K = 5

client = QdrantClient(host="localhost", port=6333)

print(f"Query: {query}\n", flush=True)

t0 = time.time()
print("Loading BGE-M3...", end=" ", flush=True)
model = BGEM3FlagModel("Roflmax/bge-m3-legal-ru-cocktail-40-60", device="cuda", use_fp16=True)
print(f"done ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
print("Embedding query...", end=" ", flush=True)
output = model.encode([query], return_dense=True, return_sparse=True)
dense = output["dense_vecs"][0]
sparse = output["lexical_weights"][0]
print(f"done ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
print("Searching Qdrant...", end=" ", flush=True)
results = client.search(
    collection_name="law_ru",
    query_vector=("dense", dense.tolist()),
    with_payload=True,
    limit=TOP_K,
)
print(f"done ({time.time()-t0:.1f}s)", flush=True)

print(f"\n=== Top {TOP_K} results ===\n")
for i, r in enumerate(results):
    payload = r.payload
    print(f"[{i+1}] Score: {r.score:.4f}")
    print(f"    Category: {payload.get('category','')} | {payload.get('code_name','')} | Active: {payload.get('active','')}")
    print(f"    Text: {payload.get('text','')[:200]}...")
    print()

context = "\n\n".join([
    f"Документ {i+1}:\n{r.payload.get('text','')}"
    for i, r in enumerate(results)
])

prompt = f"""Ты — юрист. Ответь на вопрос, используя контекст.

Контекст:
{context}

Вопрос: {query}

Ответ:"""

print(f"=== Prompt ({len(prompt)} chars) ===\n{prompt[:500]}...\n")
