import asyncio
import re
from playwright.async_api import async_playwright

BASE = "https://www.vsrf.ru"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        url = f"{BASE}/documents/own/?category=resolutions_plenum_supreme_court_russian&year=2016"
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(5000)

        # Get raw HTML of items container
        items_html = await page.eval_on_selector(
            "#vs-search-items-list",
            "el => el.outerHTML"
        )
        if items_html:
            print(f"Items HTML length: {len(items_html)}")
            # Print the first 2000 chars to understand the structure
            print("=== HTML start ===")
            print(items_html[:2000])
            print("=== HTML end ===")
            print(items_html[-1000:])

        # Try to get all links
        links = await page.eval_on_selector_all(
            "#vs-search-items-list a[href]",
            "els => els.map(el => ({href: el.href, text: el.textContent.trim()}))"
        )
        print(f"\nTotal links in items: {len(links)}")
        for l in links[:20]:
            print(f"  {l['href'][:100]} -> {l['text'][:60]}")

        # Check for document links (not category links)
        doc_links = [l for l in links if "/documents/own/" in l['href'] and "category" not in l['href']]
        print(f"\nDocument links: {len(doc_links)}")
        for l in doc_links[:5]:
            print(f"  {l['href'][:100]} -> {l['text'][:60]}")

        # Try different approach - use the JSON endpoint directly
        import requests
        sess = requests.Session()
        sess.headers.update({"User-Agent": "Mozilla/5.0"})
        cookies = await page.context.cookies()
        for c in cookies:
            sess.cookies.set(c['name'], c['value'])

        # Try with init=y via the browser's cookies
        r = sess.get(
            f"{BASE}/documents/own/?category=resolutions_plenum_supreme_court_russian&year=2016&init=y",
            timeout=30
        )
        print(f"\nDirect init=y request: status={r.status}, type={r.headers.get('content-type')}, len={len(r.text)}")
        if 'json' in r.headers.get('content-type', ''):
            data = r.json()
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:10]}")
                for k, v in data.items():
                    if isinstance(v, str) and len(v) > 50:
                        print(f"  Key '{k}': {len(v)} chars, preview: {v[:200]}")
                    elif isinstance(v, list):
                        print(f"  Key '{k}': list, {len(v)} items, first: {str(v[0])[:100]}")
                    elif isinstance(v, dict):
                        print(f"  Key '{k}': dict, keys: {list(v.keys())[:5]}")

        await browser.close()

asyncio.run(main())
