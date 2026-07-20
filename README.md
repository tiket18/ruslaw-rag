# RusLaw RAG — Семантический поиск по законодательству РФ

Векторный поиск по российскому праву: кодексы, постановления правительства, определения КС РФ, пленум ВС РФ.

## Демоны

| Сервис | Порт | Роль |
|--------|------|------|
| `ollama-proxy.service` | 11401 | FastAPI-прокси: Ollama ↔ OpenAI API, MTP/Eagle3 |
| `ollama.service` | 11434 | Ollama (магистраль) |
| `qdrant.service` | 6333 | Векторное хранилище |

**Qdrant:** коллекция `law_ru`, **1 106 682 точки** (RusLawOD + Пленум ВС).  
Данные на диске: `/home/vllm/qdrant_storage/`.

## Источники

- **[irlspbru/RusLawOD](https://huggingface.co/datasets/irlspbru/RusLawOD)** — 304 864 док., 1991–2025, 5.9 GB parquet
- **Пленум ВС** — 926 док. (plenum.rf + legalacts.ru), 9 050 чанков
- Фильтрация: кодексы + ФЗ + ПП РФ + КС РФ → 112 508 док. → 1 097 632 чанка

## Модель эмбеддингов

`Roflmax/bge-m3-legal-ru-cocktail-40-60` (2.2 GB, GPU):
- dense: 1024d, COSINE
- sparse: lexical_weights
- batch: 256, max_length: 8192, fp16
- ~256 chunks/sec на RTX 3060 (12 GB)

## Скрипты

| Файл | Назначение |
|------|------------|
| `rag_update.py` | Единый пайплайн: download → filter → chunk → embed → qdrant |
| `rag_prep/` | Поэтапные скрипты (fetch, chunk, embed, load, eval) |
| `fetch_plenum.py` | Парсер Пленума ВС с plenum.rf |
| `merge_plenum.py` | Слияние plenum.rf + legalacts.ru |
| `fetch_vsrf.py` | Парсер Пленума ВС с vsrf.ru |
| `load_parts_sequential.py` | Загрузка эмбеддингов в Qdrant по частям |
| `load_part_server.py` | Серверная загрузка эмбеддингов |
| `embed_resumable.py` | Эмбеддинг с чекпоинтами |

## Pipeline

```
RusLawOD (11 parquet, 304K docs)
  → фильтр: codes + fz + gov + ks (112K docs)
  → chunking (1500 chars, 200 overlap) → 1.1M chunks
  → BGE-M3 эмбеддинг (batch=256, GPU)
  → загрузка в Qdrant collection law_ru
```

## Быстрый старт

```bash
# Установка
pip install FlagEmbedding qdrant-client torch tqdm transformers

# Qdrant
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar xzf qdrant-*.tar.gz && ./qdrant &

# Полный цикл
python3 rag_update.py --full

# Статус
python3 rag_update.py --status
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

| Компонент | Объём |
|-----------|-------|
| 🔹 RusLawOD parquet | 304 864 док., 5.9 GB |
| 🔹 Отфильтровано | 112 508 док., 2.2 GB |
| 🔹 Чанков | 1 097 632, 3.3 GB |
| 🔹 Эмбеддингов | 1 097 633, 24.3 GB |
| 🔹 Пленум ВС | 9 050 чанков |
| 🔹 **Всего в Qdrant** | **1 106 682 точки** |
| 🔹 GPU (RTX 3060) | 8.7 GB / 12 GB |

