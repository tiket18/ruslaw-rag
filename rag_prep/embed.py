import json, sys, os
from FlagEmbedding import BGEM3FlagModel
import numpy as np

INPUT = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
os.makedirs('/home/vllm/rag/embeddings', exist_ok=True)

print('Loading model...')
model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cuda', use_fp16=True)

print('Embedding...')
batch_texts = []
batch_docs = []
batch_size = 32

def to_json_serializable(obj):
    """Convert numpy types to Python native types"""
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

with open(INPUT) as fin, open(OUTPUT, 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        batch_texts.append(doc['text'])
        batch_docs.append(doc)
        
        if len(batch_texts) >= 32:
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
                    'dense': to_json_serializable(outputs['dense_vecs'][i]),
                    'sparse': {
                        'indices': list(sparse_dict.keys()),
                        'values': to_json_serializable(list(sparse_dict.values()))
                    }
                }, ensure_ascii=False) + '\n')
            batch_texts.clear()
            batch_docs.clear()
    
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
                'dense': to_json_serializable(outputs['dense_vecs'][i]),
                'sparse': {
                    'indices': list(outputs['lexical_weights'][i].keys()),
                    'values': to_json_serializable(list(outputs['lexical_weights'][i].values()))
                }
            }, ensure_ascii=False) + '\n')

print('Done!')