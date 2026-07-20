#!/usr/bin/env python3
"""
Evaluate RAG retrieval quality.
Uses test queries with expected articles.
"""
import json
import os
import sys
from pathlib import Path

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "law_ru"

# Test queries: (query, expected_article_or_code)
TEST_QUERIES = [
    ("Что такое кредитная организация согласно закону о банках?", "395-I", "article"),
    ("Порядок применения патентной системы налогообложения", "346.43", "article"),
    ("Какие виды общественных объединений запрещены?", "КоАП", "code"),
    ("Срок давности по общим искам", "196", "article"),
    ("Право общей долевой собственности", "244", "article"),
    ("Основания для возникновения обязательств", "307", "article"),
    ("Сроки исполнения обязательств", "314", "article"),
    ("Ответственность за нарушение трудового договора", "ТК РФ", "code"),
    ("Порядок разрешения трудовых спорев", "ТК РФ", "code"),
    ("Административные правонарушения в сфере налогов", "КоАП", "code"),
]

def search(query: str, top_k: int = 10):
    from qdrant_client import QdrantClient
    from FlagEmbedding import BGEM3FlagModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel("BAAI/bge-m3", device=device, use_fp16=(device=="cuda"))

    outputs = model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense_vec = outputs["dense_vecs"][0].tolist()
    sparse_vec = outputs["lexical_weights"][0]

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    results = client.query_points(
        collection_name=COLLECTION,
        query=dense_vec,
        sparse=sparse_vec,
        limit=top_k,
        with_payload=True,
    ).points
    return results

def evaluate():
    print(f"Evaluating {len(TEST_QUERIES)} queries against {COLLECTION}...")
    for i, (query, expected, exp_type) in enumerate(TEST_QUERIES, 1):
        print(f"\nQ{i}: {query}")
        print(f"  Expected: {expected} ({exp_type})")
        results = search(query, top_k=5)
        for rank, r in enumerate(results, 1):
            art = r.payload.get("article", "")
            code = r.payload.get("code_name", "")
            src = r.payload.get("source", "")
            match = "✓" if (exp_type == "article" and expected in art) or (exp_type == "code" and expected in code) else ""
            print(f"  {rank}. {code} ст.{art} | score={r.score:.3f} {match}")

if __name__ == "__main__":
    evaluate()