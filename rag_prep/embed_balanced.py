import json
import sys
import os
import gc
import resource
import subprocess
import signal
from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Hard memory limit: 8GB total
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024))

INPUT = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
os.makedirs('/home/vllm/rag/embeddings', exist_ok=True)

def to_json_serializable(obj):
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    return obj

# Limit memory
resource.setrlimit(resource.RLIMIT_AS, (7 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024))

def log_mem():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return f'RSS: {line.split()[1]} kB'
    return 'unknown'

def main():
    print('Loading model...', flush=True)
    model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cuda', use_fp16=True)
    print(f'Model loaded, {log_mem()}', flush=True)
    
    batch_size = 48
    max_lines = 500  # flush every 500 lines
    batch_texts = []
    batch_docs = []
    written = 0
    line_count = 0
    
    with open(INPUT, 'r') as fin, open(OUTPUT, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = json.loads(line)
                batch_texts.append(doc['text'])
                batch_docs.append(doc)
                
                if len(batch_texts) >= 48:
                    outputs = model.encode(batch_texts, batch_size=48, max_length=8192,
                                           return_dense=True, return_sparse=True, return_colbert_vecs=False)
                    for i, doc in enumerate(batch_docs):
                        sparse_dict = outputs['lexical_weights'][i]
                        fout.write(json.dumps({
                            'id': doc['id'],
                            'parent_id': doc['parent_id'],
                            'text': doc['text'],
                            'chunk_index': doc.get('chunk_index', 0),
                            'total_chunks': doc.get('total_chunks', 0),
                            'source': doc['source'],
                            'dense': outputs['dense_vecs'][i].tolist(),
                            'sparse': {
                                'indices': list(sparse_dict.keys()),
                                'values': [float(v) for v in sparse_dict.values()]
                            }
                        }, ensure_ascii=False) + '\n')
                    written += len(batch_docs)
                    batch_texts.clear()
                    batch_docs.clear()
                    line_count += len(batch_docs)
                    
                    if line_count % max_lines == 0:
                        gc.collect()
                        print(f'Written: {written}, {log_mem()}', flush=True)
                    
            except MemoryError:
                print(f'MEMORY ERROR at line {line_count}, flushing...', flush=True)
                gc.collect()
                if batch_texts:
                    outputs = model.encode(batch_texts, batch_size=16, max_length=8192,
                                           return_dense=True, return_sparse=True, return_colbert_vecs=False)
                    for i, doc in enumerate(batch_docs):
                        sparse_dict = outputs['lexical_weights'][i]
                        fout.write(json.dumps({
                            'id': doc['id'],
                            'parent_id': doc['parent_id'],
                            'text': doc['text'],
                            'chunk_index': doc.get('chunk_index', 0),
                            'total_chunks': doc.get('total_chunks', 0),
                            'source': doc['source'],
                            'dense': outputs['dense_vecs'][i].tolist(),
                            'sparse': {
                                'indices': list(sparse_dict.keys()),
                                'values': [float(v) for v in sparse_dict.values()]
                            }
                        }, ensure_ascii=False) + '\n')
                    written += len(batch_docs)
                    batch_texts.clear()
                    batch_docs.clear()
                    gc.collect()
    
    # Flush remaining
    if batch_texts:
        outputs = model.encode(batch_texts, batch_size=32, max_length=8192,
                               return_dense=True, return_sparse=True, return_colbert_vecs=False)
        for i, doc in enumerate(batch_docs):
            sparse_dict = outputs['lexical_weights'][i]
            fout.write(json.dumps({
                'id': doc['id'],
                'parent_id': doc['parent_id'],
                'text': doc['text'],
                'chunk_index': doc.get('chunk_index', 0),
                'total_chunks': doc.get('total_chunks', 0),
                'source': doc['source'],
                'dense': outputs['dense_vecs'][i].tolist(),
                'sparse': {
                    'indices': list(sparse_dict.keys()),
                    'values': [float(v) for v in sparse_dict.values()]
                }
            }, ensure_ascii=False) + '\n')
            written += len(batch_docs)
    
    print(f'Done! Written: {written} embeddings', flush=True)

INPUT = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'

if __name__ == '__main__':
    main()