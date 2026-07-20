#!/usr/bin/env python3
"""
Fetch Russian legislation from zakon-rf/zakon-rf via zip download.
Extract JSON per code, save to /home/vllm/rag/raw/json/
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ZIP_URL = "https://github.com/zakon-rf/zakon-rf/archive/refs/heads/main.zip"
CLONE_DIR = Path("/home/vllm/rag/raw/zakon-rf")
OUT_DIR = Path("/home/vllm/rag/raw/json")

CODES = [
    "gk", "zhk", "tk", "kas", "gpk", "apk", "uk", "koap", "nk",
]

def run(cmd, cwd=None):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download zip if not present
    zip_path = CLONE_DIR.parent / "zakon-rf-main.zip"
    if not CLONE_DIR.exists():
        print("Downloading zakon-rf...")
        run(f"curl -L --max-time 300 -o {zip_path} {ZIP_URL}")
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(CLONE_DIR.parent)
        # Rename extracted folder
        extracted = CLONE_DIR.parent / "zakon-rf-main"
        if extracted.exists():
            if CLONE_DIR.exists():
                shutil.rmtree(CLONE_DIR)
            extracted.rename(CLONE_DIR)
        zip_path.unlink(missing_ok=True)
    else:
        print("zakon-rf already exists, skipping download")

    # Process each code
    for code in CODES:
        code_dir = CLONE_DIR / code
        if not code_dir.exists():
            print(f"  WARNING: {code} not found in repo")
            continue

        out_file = OUT_DIR / f"{code}.jsonl"
        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for json_file in sorted(code_dir.rglob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    data["_source_code"] = code
                    data["_source_file"] = str(json_file.relative_to(CLONE_DIR))
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    print(f"  ERROR reading {json_file}: {e}", file=sys.stderr)

        print(f"  {code}: {count} articles -> {out_file}")

    print("\nDone. Files in:", OUT_DIR)

if __name__ == "__main__":
    main()
