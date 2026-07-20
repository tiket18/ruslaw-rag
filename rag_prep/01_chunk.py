#!/usr/bin/env python3
"""
Chunk law documents for embedding.
Input: /home/vllm/rag/parsed/law_documents.jsonl
Output: /home/vllm/rag/chunks/law_chunks.jsonl
"""
import json
from pathlib import Path

INPUT_FILE = Path("/home/vllm/rag/parsed/law_documents.jsonl")
OUTPUT_DIR = Path("/home/vllm/rag/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "law_chunks.jsonl"

MAX_CHARS = 1500  # ~500 tokens for Russian
OVERLAP = 200

def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 chars
            search_start = max(start, end - 200)
            for i in range(end, search_start, -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= len(text):
            break
    return chunks

def main():
    output_path = OUTPUT_FILE
    total_chunks = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            
            doc_id = doc.get('id', '')
            text = doc.get('text', '')
            meta = {k: v for k, v in doc.items() if k not in ('text',)}
            
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "id": f"{doc_id}_chunk_{i}",
                    "parent_id": doc_id,
                    "text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **meta
                }
                fout.write(json.dumps(chunk_doc, ensure_ascii=False) + '\n')
                total_chunks += 1
    
    print(f"Created {total_chunks} chunks in {output_path}")

if __name__ == "__main__":
    main()