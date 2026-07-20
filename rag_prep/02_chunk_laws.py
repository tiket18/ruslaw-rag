#!/usr/bin/env python3
"""
Chunk normalized law articles for embedding.
Input: /home/vllm/rag/raw/normalized/*.jsonl
Output: /home/vllm/rag/chunks/*.jsonl
"""
import json
import os
import sys
from pathlib import Path

NORM_DIR = Path("/home/vllm/rag/raw/normalized")
CHUNK_DIR = Path("/home/vllm/rag/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Chunking config
MAX_TOKENS = 512
OVERLAP_TOKENS = 50

def count_tokens(text: str) -> int:
    """Rough token count: ~2 chars per token for Russian."""
    return max(1, len(text) // 2)

def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Split text into overlapping chunks by sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_text = " ".join(current)
            overlap_tokens = count_tokens(overlap_text)
            while overlap_tokens > overlap and len(current) > 1:
                current.pop(0)
                overlap_text = " ".join(current)
                overlap_tokens = count_tokens(overlap_text)
            current_tokens = overlap_tokens
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks

def process_file(src_file: Path, dst_file: Path):
    count_in = 0
    count_out = 0
    with open(src_file, "r", encoding="utf-8") as f_in, \
         open(dst_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            count_in += 1
            rec = json.loads(line)
            text = rec.get("text", "").strip()
            if not text:
                continue
            title = rec.get("title", "").strip()
            article = rec.get("article", "")
            code = rec.get("code_name", rec.get("source", ""))

            # Build chunk prefix with citation
            prefix = f"{code}, ст. {article}"
            if title:
                prefix += f" — {title}"
            prefix += "\n\n"

            full_text = prefix + text
            chunks = chunk_text(full_text)

            for i, chunk in enumerate(chunks):
                chunk_rec = {
                    "id": f"{rec['id']}_chunk_{i}",
                    "parent_id": rec["id"],
                    "source": rec["source"],
                    "code_name": code,
                    "article": article,
                    "title": title,
                    "text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "meta": rec.get("meta", {}),
                }
                f_out.write(json.dumps(chunk_rec, ensure_ascii=False) + "\n")
                count_out += 1
    print(f"  {src_file.name}: {count_in} articles -> {count_out} chunks")
    return count_out

def main():
    if not NORM_DIR.exists():
        print(f"ERROR: {NORM_DIR} does not exist. Run 01_fetch_laws.py first.", file=sys.stderr)
        sys.exit(1)

    total_chunks = 0
    for src_file in sorted(NORM_DIR.glob("*.jsonl")):
        dst_file = CHUNK_DIR / src_file.name
        total_chunks += process_file(src_file, dst_file)

    print(f"\nTotal chunks: {total_chunks}")
    print(f"Output dir: {CHUNK_DIR}")

if __name__ == "__main__":
    main()