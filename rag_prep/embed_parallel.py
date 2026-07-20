import json
import sys
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from FlagEmbedding import BGEM3FlagModel

INPUT = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT_DIR = '/home/vllm/rag/embeddings'
N_WORKERS = min(8, cpu_count())
CHUNKS_PER_FILE = 500000  # ~20M / 8 = ~2.6M per worker

os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_json_serializable(obj):
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    return obj

def init_worker():
    global _model
    _model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cuda', use_fp16=True)

def to_json_serializable(obj):
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    return obj

def embed_chunk_file(args):
    chunk_file, out_file = args
    global _model
    if '_model' not in globals():
        _model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cuda', use_fp16=True)
    
    batch_texts = []
    batch_docs = []
    batch_size = 256
    count = 0
    
    with open(chunk_file) as fin, open(out_file, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            batch_texts.append(doc['text'])
            batch_docs.append(doc)
            
            if len(batch_texts) >= 256:
                outputs = _model.encode(batch_texts, batch_size=256, max_length=8192,
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
                            'values': list(sparse_dict.values())
                        }
                    }, ensure_ascii=False) + '\n')
                batch_texts.clear()
                batch_docs.clear()
        
        if batch_texts:
            outputs = _model.encode(batch_texts, batch_size=256, max_length=8192,
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
                        'values': list(sparse_dict.values())
                    }
                }, ensure_ascii=False) + '\n')
    return f'Done {out_file}'

def split_input_file(input_file, n_parts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    chunk_files = []
    
    with open(INPUT) as fin:
        lines = fin.readlines()
    
    per_part = len(lines) // n_parts + 1
    for i in range(n_parts):
        start = i * per_part
        end = min(start + per_part, len(lines))
        if start >= len(lines):
            break
        chunk_file = os.path.join(out_dir, f'chunk_part_{i}.jsonl')
        with open(chunk_file, 'w') as fout:
            fout.writelines(lines[start:end])
        chunk_files.append(chunk_file)
    
    return chunk_files

INPUT = '/home/vllm/rag/chunks/law_chunks.jsonl'
SPLIT_DIR = '/home/vllm/rag/chunks_split'
OUTPUT_DIR = '/home/vllm/rag/embeddings_parallel'
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':
    print(f'Using {N_WORKERS} workers')
    
    # Split input
    print('Splitting input file...')
    chunk_files = split_input_file(INPUT, N_WORKERS, SPLIT_DIR)
    print(f'Created {len(chunk_files)} chunk files')
    
    # Process in parallel
    out_files = [os.path.join(OUTPUT_DIR, f'emb_part_{i}.jsonl') for i in range(len(chunk_files))]
    args = list(zip(chunk_files, out_files))
    
    print('Starting parallel embedding...')
    with Pool(N_WORKERS, initializer=init_worker) as pool:
        results = pool.map(embed_chunk_file, args)
        for r in results:
            print(r)
    
    # Merge outputs
    print('Merging outputs...')
    final_output = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
    with open(final_output, 'w') as fout:
        for out_file in out_files:
            if os.path.exists(out_file):
                with open(out_file) as fin:
                    for line in fin:
                        fout.write(line)
    
    print('Done!')