import json, sys

def chunk(t, mx=1500, ov=200):
    if len(t) <= mx: return [t]
    res, s = [], 0
    while s < len(t):
        e = min(s + mx, len(t))
        if e < len(t):
            for i in range(e, max(s, e-200), -1):
                if t[i] in '.!?':
                    e = i + 1
                    break
        res.append(t[s:e].strip())
        s = e - ov
        if s >= len(t): break
    return res

cnt = 0
with open('/home/vllm/rag/parsed/law_documents.jsonl') as fin, \
     open('/home/vllm/rag/chunks/law_chunks.jsonl', 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        doc_id = d.get('id', '')
        text = d.get('text', '')
        src = d.get('source', '')
        chunks = chunk(text)
        for i, ch in enumerate(chunks):
            fout.write(json.dumps({
                'id': f'{doc_id}_chunk_{i}',
                'parent_id': doc_id,
                'text': ch,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source': src
            }, ensure_ascii=False) + '\n')
        cnt += 1
        if cnt % 10 == 0:
            sys.stderr.write(f'Processed {cnt} docs\n')
    print(f'Done: {cnt} docs')