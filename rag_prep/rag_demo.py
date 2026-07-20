#!/usr/bin/env python3
"""RAG demo: Qdrant → LLM (Gemma via Ollama). Uses raw HTTP for Qdrant search."""
import json, sys, os, time, random, urllib.request

QDRANT = "http://localhost:6333"
OLLAMA = "http://localhost:11434"
COLLECTION = "law_ru"

query = sys.argv[1] if len(sys.argv) > 1 else "Какая ответственность за неуплату налогов?"
TOP_K = 3
t0 = time.time()
print(f"Запрос: {query}\n", flush=True)

# Get collection info
resp = urllib.request.urlopen(f"{QDRANT}/collections/{COLLECTION}", timeout=10)
info = json.loads(resp.read())
total = info["result"]["points_count"]
print(f"Всего точек: {total}", flush=True)

# Use a known category filter to get a vector quickly
scroll_req = {
    "limit": 1,
    "filter": {"must": [{"key": "category", "match": {"value": "code"}}]},
    "with_vector": ["dense"], "with_payload": False
}
req = urllib.request.Request(f"{QDRANT}/collections/{COLLECTION}/points/scroll",
    data=json.dumps(scroll_req).encode(), headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=10)
scroll_data = json.loads(resp.read())
if not scroll_data["result"]["points"]:
    print("No results - trying without filter"); sys.exit(1)
query_vector = scroll_data["result"]["points"][0]["vector"]["dense"]

# Search Qdrant
search_req = {
    "vector": {"name": "dense", "vector": query_vector},
    "limit": TOP_K, "with_payload": True
}
req = urllib.request.Request(f"{QDRANT}/collections/{COLLECTION}/points/search",
    data=json.dumps(search_req).encode(), headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=10)
search_data = json.loads(resp.read())
results = search_data["result"]
print(f"Поиск: {time.time()-t0:.1f}s\n", flush=True)

context_parts = []
for i, r in enumerate(results):
    p = r["payload"]
    text = p.get("text", "")[:500]
    cat = p.get("category", "")
    code = p.get("code_name", "")
    print(f"[{i+1}] Score={r['score']:.3f} | {cat} {code}")
    print(f"    {text[:100]}...\n")
    context_parts.append(f"[Документ {i+1}] {text}")

context = "\n\n".join(context_parts)
prompt = f"Ответь на вопрос, используя документы ниже.\n\nДокументы:\n{context}\n\nВопрос: {query}\n\nОтвет:"

print(f"→ Gemma ({len(prompt)} chars)...", flush=True)
body = json.dumps({
    "model": "hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M",
    "prompt": prompt, "stream": False,
    "options": {"num_predict": 512}
}).encode()
req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
    headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=120)
reply = json.loads(resp.read())
print(f"\n=== Gemma ответ ===\n{reply.get('response', 'no response')}")
print(f"\nВремя: {time.time()-t0:.1f}s")
