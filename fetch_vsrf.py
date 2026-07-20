"""
Scraper for Plenum decisions from vsrf.ru (2016-2022).
Uses init=y JSON endpoint to get all document IDs, then fetches each doc.
"""
import asyncio
import json
import os
import re
import time
import requests
from playwright.async_api import async_playwright

BASE = "https://www.vsrf.ru"
OUTPUT = "/home/vllm/rag/plenum_vsrf"
YEARS = list(range(2016, 2023))


async def get_all_ids(page):
    """Capture init=y JSON response for each year and extract all document IDs."""
    year_ids = {}

    async def capture_init(response):
        if "init=y" in response.url and response.status == 200:
            ct = response.headers.get("content-type", "")
            if "json" in ct:
                try:
                    data = await response.json()
                    if isinstance(data, dict) and "id" in data:
                        id_list = data["id"].get("list", [])
                        if id_list:
                            year_match = re.search(r"year=(\d+)", response.url)
                            if year_match:
                                year = int(year_match.group(1))
                                year_ids[year] = id_list
                except:
                    pass

    page.on("response", capture_init)

    for year in YEARS:
        url = f"{BASE}/documents/own/?category=resolutions_plenum_supreme_court_russian&year={year}"
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(5000)

    page.remove_listener("response", capture_init)
    return year_ids


import subprocess
import tempfile


def fetch_document_text(doc_id, year):
    """Fetch PDF document and extract text via pdftotext."""
    url = f"{BASE}/documents/own/{doc_id}/"

    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return {"id": doc_id, "url": url, "error": str(e), "year": year}

    # Check content type
    ct = r.headers.get("content-type", "")
    if "pdf" not in ct:
        return {"id": doc_id, "url": url, "error": f"Not PDF: {ct}", "year": year}

    # Extract title from Content-Disposition
    title = ""
    cd = r.headers.get("content-disposition", "")
    filename_match = re.search(r'filename\*?=(?:UTF-8\'\')?([^;]+)', cd)
    if filename_match:
        from urllib.parse import unquote
        title = unquote(filename_match.group(1)).replace(".pdf", "").strip()

    # Save PDF to temp file and extract text
    pdf_data = r.content
    text = ""

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf.write(pdf_data)
            pdf_path = tmp_pdf.name

        tmp_txt = pdf_path + ".txt"
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, tmp_txt],
            capture_output=True, timeout=30,
        )

        if result.returncode == 0 and os.path.exists(tmp_txt):
            with open(tmp_txt, "r", encoding="utf-8") as f:
                text = f.read().strip()
            os.unlink(tmp_txt)

        os.unlink(pdf_path)
    except Exception as e:
        return {"id": doc_id, "url": url, "title": title, "error": f"PDF extract: {e}", "year": year}

    return {
        "id": doc_id,
        "url": url,
        "title": title,
        "text": text,
        "year": year,
    }


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    # Step 1: Get all document IDs via Playwright
    print("=== Getting document IDs for all years ===")

    async def get_ids():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--disable-gpu"])
            page = await browser.new_page()
            ids = await get_all_ids(page)
            await browser.close()
            return ids

    year_ids = asyncio.run(get_ids())

    total_ids = sum(len(v) for v in year_ids.values())
    print(f"Total document IDs found: {total_ids}")

    for year, ids in sorted(year_ids.items()):
        print(f"  {year}: {len(ids)} docs (IDs {ids[-1]}-{ids[0]})")

    # Step 2: Fetch each document via requests
    all_docs = {}
    processed = 0

    for year in sorted(year_ids.keys()):
        ids = year_ids[year]
        print(f"\n=== Year {year} ({len(ids)} docs) ===")

        for i, doc_id in enumerate(ids):
            if str(doc_id) in all_docs:
                continue

            print(f"  [{processed+1}/{total_ids}] ID {doc_id}...", end="", flush=True)
            doc = fetch_document_text(doc_id, year)
            all_docs[str(doc_id)] = doc
            processed += 1

            if doc.get("text"):
                print(f" {len(doc['text'])} chars")
            else:
                print(f" NO TEXT")
                if "error" in doc:
                    print(f"    Error: {doc['error']}")

            time.sleep(0.5)

            # Checkpoint every 50 docs
            if processed % 50 == 0:
                save_checkpoint(all_docs)

    # Step 3: Save final results
    save_checkpoint(all_docs)
    save_merged(all_docs)


def save_checkpoint(all_docs):
    path = os.path.join(OUTPUT, "plenum_vsrf_checkpoint.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    docs_with_text = sum(1 for d in all_docs.values() if d.get("text"))
    print(f"\n  Checkpoint: {len(all_docs)} docs ({docs_with_text} with text)")


def save_merged(all_docs):
    # Full JSONL
    jsonl_path = os.path.join(OUTPUT, "plenum_vsrf_merged.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc_id in sorted(all_docs.keys()):
            f.write(json.dumps(all_docs[doc_id], ensure_ascii=False) + "\n")

    # Clean JSONL (only essential fields, non-empty)
    clean_path = os.path.join(OUTPUT, "plenum_vsrf_clean.jsonl")
    count = 0
    with open(clean_path, "w", encoding="utf-8") as f:
        for doc_id in sorted(all_docs.keys()):
            doc = all_docs[doc_id]
            if doc.get("text"):
                clean = {
                    "title": doc.get("title", ""),
                    "text": doc["text"],
                    "year": doc.get("year", 0),
                    "url": doc.get("url", ""),
                }
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
                count += 1

    print(f"\nFull: {jsonl_path} ({len(all_docs)} docs)")
    print(f"Clean: {clean_path} ({count} docs with text)")


if __name__ == "__main__":
    main()
