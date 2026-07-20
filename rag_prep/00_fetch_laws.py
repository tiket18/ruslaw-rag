#!/usr/bin/env python3
"""
Fetch Russian law texts from multiple open sources.
Sources:
1. pravo.gov.ru - bulk XML download
2. data.gov.ru - open datasets
3. GitHub repos (zakon-rf, opendatarus, etc.)
4. Fallback: local QA datasets for evaluation
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

RAW_DIR = Path("/home/vllm/rag/raw")
PARSED_DIR = Path("/home/vllm/rag/parsed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# Target codes we want
TARGET_CODES = {
    "gk": "ГК РФ",
    "gk2": "ГК РФ (часть 2)",
    "gk3": "ГК РФ (часть 3)",
    "gk4": "ГК РФ (часть 4)",
    "zhk": "ЖК РФ",
    "tk": "ТК РФ",
    "kas": "КАС РФ",
    "gpk": "ГПК РФ",
    "apk": "АПК РФ",
    "uk": "УК РФ",
    "koap": "КоАП РФ",
    "nk": "НК РФ",
    "sk": "СК РФ",
    "kasz": "КАС РФ (арбитражный)",
    "gpkz": "ГПК РФ (закон о несостоятельности)",
}

def run(cmd, cwd=None, timeout=300):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def download_file(url, dest, chunk_size=8192):
    """Download with progress bar."""
    print(f"Downloading {url} -> {dest}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def try_pravo_gov_bulk():
    """Try to download bulk XML from pravo.gov.ru."""
    print("\n=== Trying pravo.gov.ru bulk XML ===")
    urls = [
        "https://pravo.gov.ru/proxy/ips/?download=1&format=xml",
        "https://pravo.gov.ru/proxy/ips/?download=1",
    ]
    for url in urls:
        try:
            zip_path = RAW_DIR / "pravo_bulk.zip"
            download_file(url, zip_path)
            # Extract
            extract_dir = RAW_DIR / "pravo_bulk"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir()
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            print(f"Extracted to {extract_dir}")
            return extract_dir
        except Exception as e:
            print(f"Failed {url}: {e}")
    return None

def try_data_gov_ru():
    """Try to find law datasets on data.gov.ru."""
    print("\n=== Searching data.gov.ru ===")
    search_terms = [
        "гражданский кодекс",
        "налоговый кодекс",
        "трудовой кодекс",
        "жилищный кодекс",
        "кодекс административных",
        "уголовный кодекс",
        "гражданский процессуальный кодекс",
        "арбитражный процессуальный кодекс",
    ]
    datasets = []
    for term in search_terms:
        try:
            url = f"https://data.gov.ru/api/json/dataset/?search={requests.utils.quote(term)}"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data:
                    for item in data['data']:
                        datasets.append(item)
        except Exception as e:
            print(f"Search failed for {term}: {e}")
    return datasets

def try_github_repos():
    """Clone known GitHub repos with Russian laws."""
    print("\n=== Trying GitHub repos ===")
    repos = [
        ("zakon-rf", "https://github.com/zakon-rf/zakon-rf.git"),
        ("stkmv", "https://github.com/stkmv/Zakony-RF.git"),
        ("opendatarus", "https://github.com/opendatarus/russian-laws.git"),
        ("oszonline", "https://github.com/oszonline/oszonline.github.io.git"),
    ]
    cloned = []
    for name, url in repos:
        try:
            dest = RAW_DIR / f"github_{name}"
            if dest.exists():
                print(f"  {name}: already exists, pulling")
                run(f"git -C {dest} pull --ff-only", timeout=60)
            else:
                print(f"  {name}: cloning")
                run(f"git clone --depth 1 {url} {dest}", timeout=120)
            cloned.append(dest)
        except Exception as e:
            print(f"  {name}: failed - {e}")
    return cloned

def parse_github_jsonl(repo_path):
    """Parse JSONL files from GitHub repos."""
    articles = []
    for jsonl_file in repo_path.rglob("*.jsonl"):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    articles.append(data)
        except Exception as e:
            print(f"Error reading {jsonl_file}: {e}")
    for json_file in repo_path.rglob("*.json"):
        if json_file.name == "package.json":
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    articles.extend(data)
                elif isinstance(data, dict):
                    articles.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    return articles

def normalize_article(art, source):
    """Normalize article to standard format."""
    return {
        "source": source,
        "code": art.get("code") or art.get("source") or art.get("law_code") or "",
        "code_name": art.get("code_name") or art.get("law_name") or "",
        "article": str(art.get("article") or art.get("number") or art.get("num") or ""),
        "title": art.get("title") or art.get("name") or "",
        "text": art.get("text") or art.get("content") or art.get("body") or "",
        "meta": {
            "chapter": art.get("chapter") or art.get("section") or "",
            "part": art.get("part") or "",
            "date": art.get("date") or art.get("updated") or "",
        }
    }

def save_parsed(articles, filename):
    """Save parsed articles to JSONL."""
    out_path = PARSED_DIR / filename
    with open(out_path, 'w', encoding='utf-8') as f:
        for art in articles:
            f.write(json.dumps(art, ensure_ascii=False) + '\n')
    print(f"Saved {len(articles)} articles to {out_path}")
    return out_path

def main():
    print("=" * 60)
    print("FETCHING RUSSIAN LAW TEXTS")
    print("=" * 60)
    
    all_articles = []
    
    # 1. Try pravo.gov.ru bulk
    bulk_dir = try_pravo_gov_bulk()
    if bulk_dir:
        print(f"Parsing pravo.gov.ru XML from {bulk_dir}...")
        # XML parsing would go here - complex, skipping for now
        pass
    
    # 2. Try data.gov.ru
    datasets = try_data_gov_ru()
    print(f"Found {len(datasets)} datasets on data.gov.ru")
    
    # 3. Try GitHub repos
    repos = try_github_repos()
    
    # 4. Parse GitHub repos
    for repo in repos:
        articles = parse_github_jsonl(repo)
        if articles:
            source = repo.name
            normalized = [normalize_article(a, source) for a in articles]
            all_articles.extend(normalized)
            print(f"  {source}: {len(normalized)} articles")
    
    # 5. If nothing found, use HF QA datasets as fallback for eval
    if not all_articles:
        print("\n=== No full texts found, using HF QA datasets as eval data ===")
        from datasets import load_dataset
        for name in ['shokhjakhon/zakon_data', 'Roflmax/Ru-Legal-QA-v1', 'parlorsky/legal-rag-benchmark-ru']:
            try:
                ds = load_dataset(name, split='train')
                for item in ds:
                    all_articles.append({
                        "source": name,
                        "code": "",
                        "code_name": "",
                        "article": "",
                        "title": item.get('question') or item.get('Вопрос') or item.get('input') or "",
                        "text": item.get('text') or item.get('Ответ') or item.get('expected_answer') or item.get('output') or "",
                        "meta": {}
                    })
                print(f"  {name}: {len(ds)} QA pairs")
            except Exception as e:
                print(f"  {name}: failed - {e}")
    
    # Save combined
    if all_articles:
        save_parsed(all_articles, "all_laws_combined.jsonl")
    
    print(f"\nTotal articles collected: {len(all_articles)}")
    print("Done!")

if __name__ == "__main__":
    main()