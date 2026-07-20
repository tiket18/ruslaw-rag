"""Get ALL document IDs from init=y JSON response for each year"""
import asyncio
from playwright.async_api import async_playwright

BASE = "https://www.vsrf.ru"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        ajax_data = {}

        async def capture_json(response):
            if "init=y" in response.url and response.status == 200:
                ct = response.headers.get("content-type", "")
                if "json" in ct:
                    try:
                        body = await response.json()
                        ajax_data[response.url] = body
                        print(f"  Captured JSON for {response.url[:100]}: {type(body).__name__} size={len(str(body))}")
                    except Exception as e:
                        print(f"  JSON parse error: {e}")

        page.on("response", capture_json)

        for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022]:
            print(f"\n=== Year {year} ===")
            url = f"{BASE}/documents/own/?category=resolutions_plenum_supreme_court_russian&year={year}"
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(8000)

            # Find the init=y response for this year
            init_key = None
            for k in ajax_data:
                if f"year={year}" in k:
                    init_key = k
                    break

            if init_key:
                data = ajax_data[init_key]
                if isinstance(data, dict):
                    print(f"  JSON keys: {list(data.keys())[:10]}")
                    for k, v in data.items():
                        if isinstance(v, list):
                            print(f"  Key '{k}': list, {len(v)} items")
                            if len(v) > 0:
                                print(f"    First 5: {v[:5]}")
                                if isinstance(v[0], dict) and "id" in v[0]:
                                    ids = [item["id"] for item in v]
                                    print(f"    IDs count: {len(ids)}")
                                elif isinstance(v[0], (int, str)):
                                    print(f"    IDs: {v[:10]}...{v[-5:]}")
                        elif isinstance(v, dict):
                            print(f"  Key '{k}': dict, keys={list(v.keys())[:5]}")
                        elif isinstance(v, str) and len(v) > 50:
                            print(f"  Key '{k}': str, {len(v)} chars")
                            # Check if it contains IDs
                            import re
                            ids = re.findall(r'data-id="(\d+)"', v)
                            if ids:
                                print(f"    data-id values: {len(ids)} items, first 5: {ids[:5]}")

        await browser.close()

asyncio.run(main())
