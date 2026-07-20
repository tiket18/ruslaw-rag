#!/usr/bin/env python3
"""
Simple fetcher - just get what we can quickly.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("/home/vllm/rag/raw")
PARSED_DIR = Path("/home/vllm/rag/parsed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd, timeout=60):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode == 0

def main():
    all_articles = []
    
    # 1. Clone GitHub repos (quick, shallow)
    print("=== Cloning GitHub repos ===")
    repos = [
        ("stkmv", "https://github.com/stkmv/Zakony-RF.git"),
        ("opendatarus", "https://github.com/opendatarus/russian-laws.git"),
    ]
    
    for name, url in repos:
        dest = RAW_DIR / f"github_{name}"
        if not dest.exists():
            if run(f"git clone --depth 1 {url} {dest}", timeout=120):
                print(f"  {name}: cloned")
            else:
                print(f"  {name}: failed")
                continue
        else:
            print(f"  {name}: exists")
    
    # 2. Parse JSON/JSONL from repos
    for name in ["stkmv", "opendatarus"]:
        repo_path = RAW_DIR / f"github_{name}"
        if not repo_path.exists():
            continue
        print(f"=== Parsing {name} ===")
        count = 0
        for jsonl_file in repo_path.rglob("*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            all_articles.append({
                                "source": name,
                                "code": data.get("code", ""),
                                "code_name": data.get("code_name", ""),
                                "article": str(data.get("article", data.get("number", ""))),
                                "title": data.get("title", data.get("name", "")),
                                "text": data.get("text", data.get("content", data.get("body", ""))),
                                "meta": {}
                            })
                            count += 1
            except Exception as e:
                print(f"  Error reading {jsonl_file}: {e}")
        for json_file in repo_path.rglob("*.json"):
            if json_file.name == "package.json":
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            all_articles.append({
                                "source": name,
                                "code": item.get("code", ""),
                                "code_name": item.get("code_name", ""),
                                "article": str(item.get("article", item.get("number", ""))),
                                "title": item.get("title", item.get("name", "")),
                                "text": item.get("text", item.get("content", item.get("body", ""))),
                                "meta": {}
                            })
                            count += 1
                    elif isinstance(data, dict):
                        all_articles.append({
                            "source": name,
                            "code": data.get("code", ""),
                            "code_name": data.get("code_name", ""),
                            "article": str(data.get("article", data.get("number", ""))),
                            "title": data.get("title", data.get("name", "")),
                            "text": data.get("text", data.get("content", data.get("body", ""))),
                            "meta": {}
                        })
                        count += 1
            except Exception as e:
                print(f"  Error reading {json_file}: {e}")
        print(f"  {name}: {count} articles")
    
    # 3. Save combined
    if all_articles:
        out_path = PARSED_DIR / "all_laws_combined.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for art in all_articles:
                f.write(json.dumps(art, ensure_ascii=False) + '\n')
        print(f"\nSaved {len(all_articles)} articles to {out_path}")
    else:
        print("No articles found!")
    
    print("Done!")

if __name__ == "__main__":
    main()