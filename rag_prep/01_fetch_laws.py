#!/usr/bin/env python3
"""
Fetch Russian legislation from local directory.
Expected input: JSONL files with law articles in /home/vllm/rag/raw/json/
Each line: {"source": "gk", "code_name": "ГК РФ", "article": "290", "title": "...", "text": "...", "meta": {...}}
"""
import json
import os
import sys
from pathlib import Path

RAW_DIR = Path("/home/vllm/rag/raw/json")
OUT_DIR = Path("/home/vllm/rag/raw/normalized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_record(rec: dict, src_file: str) -> dict:
    """Normalize to standard format."""
    return {
        "id": f"{rec.get('source','')}_art_{rec.get('article','')}",
        "source": rec.get("source", "unknown"),
        "code_name": rec.get("code_name", ""),
        "article": str(rec.get("article", "")),
        "title": rec.get("title", "").strip(),
        "text": rec.get("text", "").strip(),
        "meta": rec.get("meta", {}),
        "_src_file": src_file,
    }

def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist", file=sys.stderr)
        print("Place law JSONL files in /home/vllm/rag/raw/json/", file=sys.stderr)
        print("Expected format per line:", file=sys.stderr)
        print('  {"source":"gk","code_name":"ГК РФ","article":"290","title":"...","text":"...","meta":{}}', file=sys.stderr)
        sys.exit(1)

    total = 0
    for src_file in sorted(RAW_DIR.glob("*.jsonl")):
        out_file = OUT_DIR / f"{src_file.stem}.jsonl"
        count = 0
        with open(src_file, "r", encoding="utf-8") as f_in, \
             open(out_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    norm = normalize_record(rec, src_file.name)
                    f_out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"WARN: {src_file}:{count+1}: JSON decode error: {e}", file=sys.stderr)
        print(f"  {src_file.name}: {count} articles -> {out_file}")
        total += count

    print(f"\nTotal normalized: {total} articles")
    print(f"Output dir: {OUT_DIR}")

if __name__ == "__main__":
    main()