# RusLaw RAG — Семантический поиск по законодательству РФ

Векторный поиск по российскому праву: кодексы, постановления правительства, определения КС РФ, пленум ВС РФ.

**Датасет:** [irlspbru/RusLawOD](https://huggingface.co/datasets/irlspbru/RusLawOD) (304K док., 1991–2025) + Пленум ВС (926 док.)

**Модель:** `Roflmax/bge-m3-legal-ru-cocktail-40-60` — dense (1024d) + sparse вектора

**Хранилище:** Qdrant, коллекция `law_ru`, 1.1M чанков

## Состав

| Компонент | Описание |
|-----------|----------|
| [`rag_update.py`](rag_update.py) | Единый пайплайн: фильтрация → чанкинг → эмбеддинг → Qdrant |
| [`rag_prep/`](rag_prep/) | Скрипты этапов пайплайна (fetch, chunk, embed, load, eval) |
| [`fetch_plenum.py`](fetch_plenum.py) | Парсер Пленума ВС с plenum.rf |
| [`merge_plenum.py`](merge_plenum.py) | Слияние plenum.rf + legalacts.ru |
| [`fetch_vsrf.py`](fetch_vsrf.py) | Альтернативный парсер с vsrf.ru |
| [`load_parts_sequential.py`](load_parts_sequential.py) | Загрузка эмбеддингов в Qdrant по частям |
| [`embed_resumable.py`](embed_resumable.py) | Эмбеддинг с чекпоинтами (resume после сбоя) |

## Pipeline

```
RusLawOD (11 parquet, 304K docs)
  → фильтр: codes + government + constitutional_court (~52K docs)
  → chunking (1500 chars, 200 overlap)
  → BGE-M3 эмбеддинг (batch=256, GPU)
  → загрузка в Qdrant
```

## Требования

```bash
pip install FlagEmbedding qdrant-client torch tqdm transformers
```

Qdrant server на `localhost:6333`, коллекция `law_ru`:

```python
vectors_config = {
    "dense": VectorParams(size=1024, distance=Distance.COSINE),
}
sparse_vectors_config = {
    "sparse": SparseVectorParams(index={"on_disk": True}),
}
```

## Пример поиска

```python
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("Roflmax/bge-m3-legal-ru-cocktail-40-60", device="cuda")
client = QdrantClient("http://localhost:6333")

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

## Статус

- ✅ RusLawOD: 1.1M чанков в Qdrant
- ✅ Пленум ВС: 9K чанков (plenum.rf + legalacts.ru)
- ✅ BGE-M3: dense + sparse, 1024d
- `rag_unload` — тяжёлые LLM выгружаются перед эмбеддингом
