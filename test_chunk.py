import json, sys

def chunk(t, mx=1500, ov=200):
    if len(t) <= mx: return [t]
    res, s = [], 0
    while s < len(t):
        e = min(s + mx, len(t))
        if e < len(t):
            for i in range(e, max(s, e-200), -1):
                if t[i] in ".!?":
                    e = i + 1
                    break
        res.append(t[s:e].strip())
        s = e - ov
        if s >= len(t): break
    return res

with open("/home/vllm/rag/parsed/law_documents.jsonl") as f:
    line = f.readline().strip()
    d = json.loads(line)
    text = d.get("text", "")
    print(f"Text length: {len(text)}")
    chunks = chunk(text)
    print(f"Number of chunks: {len(chunks)}")
    for i, ch in enumerate(chunks[:3]):
        print(f"Chunk {i}: len={len(ch)}, first 80: {ch[:80]}...")