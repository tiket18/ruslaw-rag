#!/usr/bin/env python3
"""
Normalize raw zakon-rf JSONL files and chunk long articles.
Input: /home/vllm/rag/raw/json/*.jsonl
Output: /home/vllm/rag/raw/jsonl/*.jsonl (unified format)
"""
import json
import re
from pathlib import Path

RAW_JSON_DIR = Path("/home/vllm/rag/raw/json")
OUT_DIR = Path("/home/vllm/rag/raw/jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODE_MAP = {
    "gk": ("gk_rf", "ГК РФ"),
    "zhk": ("zhk_rf", "ЖК РФ"),
    "tk": ("tk_rf", "ТК РФ"),
    "kas": ("kas_rf", "КАС РФ"),
    "gpk": ("gpk_rf", "ГПК РФ"),
    "apk": ("apk_rf", "АПК РФ"),
    "uk": ("uk_rf", "УК РФ"),
    "koap": ("koap_rf", "КоАП РФ"),
    "nk": ("nk_rf", "НК РФ"),
}

MAX_TOKENS = 1200  # ~1200 tokens per chunk
OVERLAP = 150

def estimate_tokens(text: str) -> int:
    return len(text) // 3  # rough for Russian

def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> list[str]:
    tokens_est = estimate_tokens(text)
    if tokens_est <= max_tokens:
        return [text]

    # Split by paragraphs first
    paras = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""
    current_tokens = 0

    for para in paras:
        para_tokens = estimate_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append(current.strip())
            # Start new chunk with overlap
            tail = current[-overlap*3:] if len(current) > overlap*3 else current
            current = tail + "\n\n" + para
            current_tokens = estimate_tokens(current)
        else:
            current += ("\n\n" if current else "") + para
            current_tokens += para_tokens

    if current:
        chunks.append(current.strip())
    return chunks

def normalize_record(rec: dict, code_key: str) -> dict:
    code_id, code_name = CODE_MAP.get(code_key, (code_key, code_key.upper()))

    # Extract fields from various possible formats
    article = rec.get("number") or rec.get("article") or rec.get("num") or ""
    title = rec.get("title") or rec.get("name") or ""
    content = rec.get("content") or rec.get("text") or rec.get("body") or ""

    # Clean content
    content = re.sub(r'\s+', ' ', content).strip()

    return {
        "source": code_key,
        "code_name": code_name,
        "article": str(article),
        "title": title,
        "text": content,
        "meta": {k: v for k, v in rec.items() if k not in ["number", "article", "num", "title", "name", "content", "text", "body"]},
    }

def process_code(code_key: str):
    in_file = RAW_JSON_DIR / f"{code_key}.jsonl"
    if not in_file.exists():
        print(f"  {code_key}: not found, skipping")
        return 0

    out_file = OUT_DIR / f"{code_key}.jsonl"
    count = 0
    chunk_id = 0

    with open(in_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            norm = normalize_record(raw, code_key)
            parent_id = f"{code_key}_{norm['article']}"

            # Chunk if needed
            chunks = chunk_text(norm["text"])
            for i, chunk in enumerate(chunks):
                rec = {
                    "id": f"{code_key}_{norm['article']}_{i}" if len(chunks) > 1 else f"{code_key}_{norm['article']}",
                    "parent_id": parent_id,
                    "source": norm["source"],
                    "code_name": norm["code_name"],
                    "article": norm["article"],
                    "title": norm["title"],
                    "text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "meta": norm["meta"],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chunk_id += 1
            count += 1

    print(f"  {code_key}: {count} articles -> {chunk_id} chunks")
    return chunk_id

def main():
    print("Normalizing and chunking laws...")
    total = 0
    for code_key in CODE_MAP.keys():
        total += process_code(code_key)
    print(f"\nTotal chunks: {total}")

if __name__ == "__main__":
    main()