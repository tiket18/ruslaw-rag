#!/opt/vllm/.venv/bin/python3
"""Filter RSFSR from plenum.jsonl, download 2016-2026 from legalacts.ru, merge."""
import json
import os
import re
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path

INPUT = "/home/vllm/rag/plenum_output/plenum.jsonl"
OUTPUT = "/home/vllm/rag/plenum_output/plenum_clean.jsonl"
LOG_FILE = "/home/vllm/rag/plenum_output/merge.log"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")

def clean_text(text):
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&mdash;', '—', text)
    text = re.sub(r'&laquo;', '«', text)
    text = re.sub(r'&raquo;', '»', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
                    return raw.decode("cp1251")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log(f"[FAIL] {url}: {e}")
                return None

# ── Step 1: Filter RSFSR and clean existing data ──

def filter_existing():
    if not os.path.exists(INPUT):
        log(f"[SKIP] {INPUT} not found")
        return [], 0

    kept = 0
    removed = 0
    results = []
    with open(INPUT) as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get("title", "") or ""
            if "РСФСР" in title:
                removed += 1
                continue
            doc["title"] = clean_text(doc.get("title", ""))
            doc["text"] = clean_text(doc.get("text", ""))
            doc["text_len"] = len(doc["text"])
            results.append(doc)
            kept += 1

    log(f"[FILTER] kept={kept}, removed_rsfsr={removed}")
    return results, kept

# ── Step 2: Download from legalacts.ru ──

BASE_LA = "https://legalacts.ru"
PAGES = 9

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip = {"script", "style", "nav", "header", "footer", "noindex"}
        self.stack = []
        self.in_content = False
        self.content_depth = 0

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "")
        if "main-center-block-article-text" in cls:
            self.in_content = True
            self.content_depth = 1
        if self.in_content and tag in ("p", "br", "div"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        if self.in_content and tag == "div":
            self.content_depth -= 1
            if self.content_depth <= 0:
                self.in_content = False

    def handle_data(self, data):
        if self.in_content:
            data = data.strip()
            if data:
                self.text_parts.append(data + " ")

    def get_text(self):
        text = "".join(self.text_parts)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

def extract_links_from_page(html):
    links = set()
    for m in re.finditer(r'href="(/sud/postanovlenie-plenuma[^"]+)"', html):
        links.add(BASE_LA + m.group(1))
    return sorted(links)

def extract_text_from_page(html):
    t = TextExtractor()
    t.feed(html)
    return t.get_text()

def extract_meta_from_url(url):
    meta = {"number": None, "date": None, "year": None}
    m = re.search(r'ot-(\d{2})(\d{2})(\d{4})-n-(\d+)', url)
    if m:
        meta["date"] = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        meta["number"] = m.group(4)
        meta["year"] = int(m.group(3))
    return meta

def download_legalacts():
    all_links = set()
    for page in range(1, PAGES + 1):
        url = f"{BASE_LA}/sud/6/?page={page}"
        log(f"[PAGE] {page}")
        html = fetch(url)
        if not html:
            continue
        links = extract_links_from_page(html)
        all_links.update(links)
        time.sleep(0.5)

    log(f"[LINKS] {len(all_links)} unique links from legalacts.ru")

    docs = []
    for i, url in enumerate(sorted(all_links)):
        log(f"[DOC] {i+1}/{len(all_links)}: {url.split('/')[-2]}")
        html = fetch(url)
        if not html:
            continue
        meta = extract_meta_from_url(url)
        text = extract_text_from_page(html)

        m_title = re.search(r'<title>\s*(.*?)\s*</title>', html, re.DOTALL)
        title = clean_text(m_title.group(1)) if m_title else ""

        if not text or len(text) < 200:
            log(f"[SKIP] too short ({len(text)} chars)")
            continue

        doc = {
            "id": f"plenum_{meta['year']}_{meta['number']}",
            "source": url,
            "year": meta["year"],
            "number": meta["number"],
            "date": meta["date"],
            "title": title,
            "text": text,
            "text_len": len(text),
        }
        docs.append(doc)
        time.sleep(0.3)

    return docs

# ── Main ──

def main():
    existing, kept = filter_existing()
    log(f"\nExisting after RSFSR filter: {kept} docs")

    la_docs = download_legalacts()
    log(f"Downloaded from legalacts.ru: {len(la_docs)} docs")

    # Merge: deduplicate by id (keep legalacts.ru version for overlapping)
    merged = {}
    for doc in existing:
        merged[doc["id"]] = doc
    for doc in la_docs:
        merged[doc["id"]] = doc

    merged_list = sorted(merged.values(), key=lambda d: (d.get("year") or 0, d.get("number") or ""))
    log(f"\nTotal after merge: {len(merged_list)} docs")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for doc in merged_list:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total_text = sum(d["text_len"] for d in merged_list)
    log(f"Total text: ~{total_text // 1024 // 1024} MB")

    by_year = {}
    for d in merged_list:
        y = d.get("year") or 0
        by_year[y] = by_year.get(y, 0) + 1
    log("By year:")
    for y in sorted(by_year, reverse=True):
        log(f"  {y}: {by_year[y]}")

if __name__ == "__main__":
    main()
