#!/opt/vllm/.venv/bin/python3
"""Download all Plenum decisions from пленум.рф and extract text to JSONL."""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path

BASE_URL = "http://xn--b1azaj.xn--p1ai"
RAW_DIR = "/home/vllm/rag/plenum_raw"
OUTPUT_FILE = "/home/vllm/rag/plenum_output/plenum.jsonl"

# Year pages on the site: individual pages 2000-2015, compound pages for earlier
YEAR_PAGES = list(range(2000, 2016))  # 2000..2015 have individual year pages
COMPOUND_PAGES = {
    "1980-1972": list(range(1972, 1981)),
    "1971-1961": list(range(1961, 1972)),
}
# 1981-1999 have individual pages too
for y in range(1981, 2000):
    YEAR_PAGES.append(y)

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

log_file = open(OUTPUT_FILE + ".log", "w", encoding="utf-8")
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.text_parts = []
        self.skip_tags = {"script", "style", "nav", "header", "footer", "noindex"}
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "tr"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        is_skip = any(t in self.skip_tags for t in self.tag_stack)
        if not is_skip:
            data = data.strip()
            if data:
                self.text_parts.append(data + " ")

    def get_text(self):
        text = "".join(self.text_parts)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)"
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                try:
                    return raw.decode("utf-8")
                except:
                    try:
                        return raw.decode("cp1251")
                    except:
                        return raw.decode("latin-1")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log(f"[FAIL] {url}: {e}")
                return None

def get_year_urls(year):
    if isinstance(year, int):
        slug = f"{year}-postanovlenie-plenuma-vs-rf.html"
    else:
        slug = f"{year}-postanovlenie-plenuma-vs-rf.html"
    return f"{BASE_URL}/{slug}"

def extract_resolution_links(html):
    links = set()
    for m in re.finditer(r'href="([^"]+)"', html):
        url = m.group(1)
        if url.endswith(".html") and "/N" in url.upper():
            if url.startswith("http"):
                links.add(url)
            else:
                links.add(BASE_URL + "/" + url.lstrip("/"))
    return sorted(links)

def extract_metadata_from_html(html, url):
    meta = {"source": url, "year": None, "number": None, "date": None, "title": None}

    m = re.search(r'/N(\d+)-ot-(\d{2})\.(\d{2})\.(\d{4})', url)
    if m:
        meta["number"] = m.group(1)
        meta["date"] = f"{m.group(4)}-{m.group(3)}-{m.group(2)}"
        meta["year"] = int(m.group(4))

    m = re.search(r'<title>\s*(.*?)\s*</title>', html, re.DOTALL | re.IGNORECASE)
    if m:
        meta["title"] = clean_title(m.group(1))

    if not meta["title"]:
        h1 = re.search(r'<h1[^>]*>\s*(.*?)\s*</h1>', html, re.DOTALL | re.IGNORECASE)
        if h1:
            meta["title"] = clean_title(h1.group(1))

    if not meta["year"]:
        m = re.search(r'(\d{4})\s*г[\.о]', html)
        if m:
            meta["year"] = int(m.group(1))

    return meta

def clean_title(t):
    t = re.sub(r'<[^>]+>', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_text(html):
    extractor = TextExtractor()
    extractor.feed(html)
    return extractor.get_text()

def process_year(year):
    year_key = str(year) if isinstance(year, int) else year
    url = get_year_urls(year)
    log(f"[FETCH] {year_key} page: {url}")

    html = fetch(url)
    if not html:
        log(f"[SKIP] Cannot fetch year page {year_key}")
        return []

    raw_path = os.path.join(RAW_DIR, f"{year_key}.html")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(html)

    links = extract_resolution_links(html, year_key)
    log(f"[LINKS] {year_key}: {len(links)} resolutions")
    return links

def download_resolutions(links):
    results = []
    for i, url in enumerate(links):
        log(f"[RES] {i+1}/{len(links)}: {url.split('/')[-1]}")
        html = fetch(url)
        if not html:
            continue

        meta = extract_metadata_from_html(html, url)
        text = extract_text(html)

        if not text or len(text) < 200:
            log(f"[SKIP] {url.split('/')[-1]}: too short ({len(text)} chars)")
            continue

        doc = {
            "id": f"plenum_{meta['year']}_{meta['number']}" if meta['number'] else f"plenum_{i}",
            "source": url,
            "year": meta["year"],
            "number": meta["number"],
            "date": meta["date"],
            "title": meta["title"],
            "text": text,
            "text_len": len(text),
        }
        results.append(doc)
    return results

def main():
    all_links = set()

    # 1. Parse main page for 2015 (resolutions listed directly on main page)
    log("[MAIN] Parsing main page for 2015 resolutions...")
    main_html = fetch(BASE_URL + "/")
    if main_html:
        links_2015 = extract_resolution_links(main_html)
        log(f"[MAIN] Found {len(links_2015)} resolution links on main page")
        all_links.update(links_2015)
    else:
        log("[WARN] Could not fetch main page")

    # 2. Year pages 2000-2014 (individual year pages with resolution lists)
    for year in sorted(YEAR_PAGES, reverse=True):
        if year >= 2016:
            continue
        url = f"{BASE_URL}/{year}-postanovlenie-plenuma-vs-rf.html"
        log(f"[YEAR] {year}: {url}")
        html = fetch(url)
        if not html:
            log(f"[SKIP] {year} page not found")
            continue
        raw_path = os.path.join(RAW_DIR, f"{year}.html")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(html)
        links = extract_resolution_links(html)
        log(f"[YEAR] {year}: {len(links)} resolution links")
        all_links.update(links)
        time.sleep(0.3)

    # 3. Compound pages (pre-2000 grouped years)
    for slug, years in COMPOUND_PAGES.items():
        url = f"{BASE_URL}/{slug}-postanovlenie-plenuma-vs-rf.html"
        log(f"[YEAR] {slug}: {url}")
        html = fetch(url)
        if not html:
            log(f"[SKIP] {slug} page not found")
            continue
        raw_path = os.path.join(RAW_DIR, f"{slug}.html")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(html)
        links = extract_resolution_links(html)
        log(f"[YEAR] {slug}: {len(links)} resolution links")
        all_links.update(links)
        time.sleep(0.3)

    all_links = sorted(all_links)
    log(f"\n[TOTAL] Found {len(all_links)} unique resolution links")

    all_docs = download_resolutions(all_links)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    log(f"\n[DONE] Written {len(all_docs)} plenum decisions to {OUTPUT_FILE}")

    total_text = sum(d["text_len"] for d in all_docs)
    log(f"Total text: ~{total_text // 1024 // 1024} MB")

    by_year = {}
    for d in all_docs:
        y = d["year"] or 0
        by_year[y] = by_year.get(y, 0) + 1
    log("By year:")
    for y in sorted(by_year, reverse=True):
        log(f"  {y}: {by_year[y]}")

if __name__ == "__main__":
    main()
