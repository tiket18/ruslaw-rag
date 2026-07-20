import json
import sys
import os
import gc
import resource
from FlagEmbedding import BGEM3FlagModel
import numpy as np

INPUT = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
os.makedirs('/home/vllm/rag/embeddings', exist_ok=True)

# Hard memory limit: 6GB max
resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024))

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

def main():
    print('Loading model...')
    model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cuda', use_fp16=True)
    
    batch_size = 64  # small batch
    batch_texts = []
    batch_docs = []
    count = 0
    written = 0
    
    print('Embedding...')
    with open(INPUT, 'r') as fin, open(OUTPUT, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = json.loads(line)
                batch_texts.append(doc['text'])
                batch_docs.append(doc)
                
                if len(batch_texts) >= 64:
                    outputs = model.encode(batch_texts, batch_size=64, max_length=8192,
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
                    
                    if written % 10000 == 0:
                        print(f'Written: {written}')
                        
            except MemoryError:
                print('MEMORY ERROR - flushing and gc')
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
    
    print(f'Done! Written: {written} embeddings')

INPUT = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'

if __name__ == '__main__':
    main()