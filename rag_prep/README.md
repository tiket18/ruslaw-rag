# RAG Pipeline for Russian Laws (GK, ZhK, TK, KAS, GPK, APK, UK, KoAP, NK)

## Data Sources

### Option 1: Official XML Export (Recommended)
The Russian government provides bulk XML exports at:
- `https://pravo.gov.ru/proxy/ips/?download=1` — all laws in XML
- Individual laws: `https://pravo.gov.ru/proxy/ips/?docbody=&nd=<ID>`

You'll need to parse the XML to extract articles. Each law has a unique `nd` identifier.

### Option 2: Consultant.ru / Garant.ru API
If you have a subscription, their APIs provide structured JSON.

### Option 3: Manual CSV/JSON from Open Data Portal
`data.gov.ru` has some datasets (search "гражданский кодекс").

## Quick Start (with existing data)

If you have JSONL files with law articles, place them in:
```
/home/vllm/rag/raw/json/
```

Each file named by code: `gk.jsonl`, `zhk.jsonl`, `tk.jsonl`, etc.

Format per line:
```json
{"source":"gk","code_name":"ГК РФ","article":"290","title":"...","text":"...","meta":{}}
```

Then run:
```bash
cd /home/vllm/rag_prep
./run_all.sh
```

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `01_fetch_laws.py` | Normalize local JSONL files (no network) |
| `02_normalize_chunk.py` | Chunk long articles (max 512 tokens, overlap 50) |
| `03_embed_laws.py` | Generate bge-m3 embeddings (dense + sparse + colbert) |
| `04_load_qdrant.py` | Upsert to Qdrant collection `law_ru` |
| `05_eval_rag.py` | Test retrieval quality |
| `run_all.sh` | Run full pipeline |

## Requirements

```bash
pip install FlagEmbedding qdrant-client torch tqdm
```

## Qdrant Collection Schema

```python
vectors_config={
    "dense": VectorParams(size=1024, distance=Distance.COSINE),
}
sparse_vectors_config={
    "sparse": SparseVectorParams(index={"on_disk": True}),
}
```

Payload indexes on: `source`, `code_name`, `article`

## Retrieval Example

```python
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", device="cuda")
client = QdrantClient(url="http://localhost:6333")

def search(query, top_k=10):
    out = model.encode([query], return_dense=True, return_sparse=True)
    results = client.query_points(
        collection_name="law_ru",
        query=out["dense_vecs"][0].tolist(),
        sparse=SparseVector(
            indices=list(out["lexical_weights"][0].keys()),
            values=list(out["lexical_weights"][0].values()),
        ),
        limit=top_k,
        with_payload=True,
    ).points
    return results
```

## Expected Performance

- ~50k articles across 9 codes
- ~80k chunks after chunking
- Embedding: ~10 min on RTX 3060 (12GB)
- Qdrant index: ~2 GB disk

## Getting Full Law Texts

Since public GitHub repos don't have full texts, you have these options:

1. **pravo.gov.ru bulk XML** — register, download, parse with `lxml`
2. **Consultant.ru/Garant.ru** — paid API, best quality
3. **data.gov.ru** — search for "кодекс" datasets
4. **Manual** — copy from consultant.ru HTML (small scale)

The pipeline is ready — just add the source JSONL files.