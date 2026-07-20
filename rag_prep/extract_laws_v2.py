#!/usr/bin/env python3
"""
Extract law texts from HF QA datasets.
"""
import json
import re
from pathlib import Path
from datasets import load_dataset

PARSED_DIR = Path("/home/vllm/rag/parsed")
PARSED_DIR.mkdir(parents=True, exist_ok=True)

def extract_ru_legal_qa():
    """Roflmax/Ru-Legal-QA-v1 - has full_text with law documents"""
    print("Loading Ru-Legal-QA-v1...")
    ds = load_dataset("Roflmax/Ru-Legal-QA-v1", split="train")
    articles = {}
    
    for item in ds:
        # Parse full_text JSON
        ft_str = item.get('full_text', '')
        if not ft_str:
            continue
        try:
            ft = json.loads(ft_str)
        except:
            continue
        
        for law_name, content in ft.items():
            # law_name like "08.02.2018 № 127"
            # content is the full law text
            if isinstance(content, str) and len(content) > 100:
                # Use law_name as key, store full text
                key = law_name.replace('№', '').replace('.', '').replace(' ', '_')
                articles[key] = content
    
    print(f"  Extracted {len(articles)} law documents from Ru-Legal-QA-v1")
    return articles

def extract_zakon_data():
    """shokhjakhon/zakon_data - QA with citations"""
    print("Loading zakon_data...")
    ds = load_dataset("shokhjakhon/zakon_data", split="train")
    articles = {}
    
    for item in ds:
        text = item.get('text', '')
        # This has Q&A format with citations in the text
        # Extract article references and surrounding text
        # Pattern: "статья 346.43 НК РФ" etc.
        matches = list(re.finditer(
            r'(?:стать[ья]\.?\s*)?(\d+(?:\.\d+)?)\s*(НК|ГК|ТК|ЖК|КоАП|УК|АПК|ГПК|КАС|НК|СК)\s*РФ',
            text, re.IGNORECASE
        ))
        for m in matches:
            art = f"{m.group(1)}_{m.group(2)}"
            # Store context around match
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 1000)
            ctx = text[start:end]
            if art not in articles or len(ctx) > len(articles[art]):
                articles[art] = ctx
    
    print(f"  Extracted {len(articles)} article references from zakon_data")
    return articles

def extract_lordbluebell():
    """LordBluebell/Zakon - Serbian laws but has full text in reference"""
    print("Loading LordBluebell/Zakon...")
    ds = load_dataset("LordBluebell/Zakon", split="train")
    articles = {}
    
    for item in ds:
        ref = item.get('reference', '')
        if ref and len(ref) > 200:
            # This is Serbian law, skip for now
            pass
    
    print(f"  Skipped (Serbian laws)")
    return {}

def main():
    all_articles = {}
    
    for func in [extract_ru_legal_qa, extract_zakon_data, extract_lordbluebell]:
        try:
            arts = func()
            for k, v in arts.items():
                if k not in all_articles or len(v) > len(all_articles.get(k, '')):
                    all_articles[k] = v
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
    
    print(f"\nTotal unique documents: {len(all_articles)}")
    
    # Save
    out_file = PARSED_DIR / "law_documents.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for doc_id, text in all_articles.items():
            f.write(json.dumps({
                "id": doc_id,
                "text": text,
                "source": "hf_qa_datasets"
            }, ensure_ascii=False) + "\n")
    
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()