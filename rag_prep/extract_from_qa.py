#!/usr/bin/env python3
"""
Extract law texts from HF datasets we already know work.
Datasets: shokhjakhon/zakon_data, Roflmax/Ru-Legal-QA-v1, parlorsky/legal-rag-benchmark-ru, LordBluebell/Zakon
"""
import json
import re
from pathlib import Path
from datasets import load_dataset

PARSED_DIR = Path("/home/vllm/rag/parsed")
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets that have full text citations
def extract_from_zakon_data():
    """shokhjakhon/zakon_data - has full text with citations"""
    print("Loading zakon_data...")
    ds = load_dataset("shokhjakhon/zakon_data", split="train")
    articles = {}
    for item in ds:
        text = item['text']
        # Extract article references
        matches = re.findall(r'(?:стать[ья]\.?\s*)?(\d+(?:\.\d+)?)\s*(?:НК|ГК|ТК|ЖК|КоАП|УК|АПК|ГПК|КАС)\s*РФ', text)
        for match in matches:
            # We have the full text, store it
            articles[match] = text[:5000]  # Store snippet
    return articles

def extract_from_ru_legal_qa():
    """Roflmax/Ru-Legal-QA-v1 - has full_text with citations"""
    print("Loading Ru-Legal-QA-v1...")
    ds = load_dataset("Roflmax/Ru-Legal-QA-v1", split="train")
    articles = {}
    for item in ds:
        full_text = item.get('full_text', '')
        if full_text:
            # Parse the JSON string
            try:
                ft = json.loads(full_text)
                for law_name, law_content in ft.items():
                    if isinstance(law_content, list):
                        for snippet in law_content:
                            # Extract article number from law_name
                            matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:НК|ГК|ТК|ЖК|КоАП|УК|АПК|ГПК|КАС)', law_name)
                            for m in matches:
                                articles[m] = snippet[:3000]
            except:
                pass
    return articles

def extract_from_lordbluebell():
    """LordBluebell/Zakon - has reference with full law text"""
    print("Loading LordBluebell/Zakon...")
    ds = load_dataset("LordBluebell/Zakon", split="train")
    articles = {}
    for item in ds:
        ref = item.get('reference', '')
        if ref:
            matches = re.findall(r'(?:Член|Статья)\s+(\d+)', ref)
            for m in matches:
                articles[m] = ref[:5000]
    return articles

def main():
    all_articles = {}
    
    # Load from all sources
    for func in [extract_from_zakon_data, extract_from_ru_legal_qa, extract_from_lordbluebell]:
        try:
            arts = func()
            print(f"  Extracted {len(arts)} article references")
            for k, v in arts.items():
                if k not in all_articles or len(v) > len(all_articles[k]):
                    all_articles[k] = v
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
    
    print(f"\nTotal unique articles: {len(all_articles)}")
    
    # Save as JSONL
    out_file = PARSED_DIR / "articles_from_qa.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for article_num, text in all_articles.items():
            f.write(json.dumps({
                "article": article_num,
                "text": text,
                "source": "qa_datasets"
            }, ensure_ascii=False) + "\n")
    
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()