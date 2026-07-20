import json
import sys
import os
import gc
import resource
import numpy as np
from FlagEmbedding import BGEM3FlagModel

INPUT = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
os.makedirs('/home/vllm/rag/embeddings', exist_ok=True)

# Soft memory limit: 12GB
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024 * 1024 * 1024, 14 * 1024 * 1024 * 1024))

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
    
    batch_size = 128
    batch_texts = []
    batch_docs = []
    written = 0
    
    print('Embedding...')
    with open(INPUT, 'r') as fin, open(OUTPUT, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            doc = json.loads(line)
            batch_texts.append(doc['text'])
            batch_docs.append(doc)
            
            if len(batch_texts) >= batch_size:
                outputs = model.encode(batch_texts, batch_size=batch_size, max_length=8192,
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
                            'indices': list(outputs['lexical_weights'][i].keys()),
                            'values': [float(v) for v in outputs['lexical_weights'][i].values()]
                        }
                    }, ensure_ascii=False) + '\n')
                batch_texts.clear()
                batch_docs.clear()
                gc.collect()
            
            # Progress
            if len(batch_docs) == 0 and written % 50000 == 0:
                print(f'Written: {written}')
        
        # Flush remaining
        if batch_texts:
            outputs = model.encode(batch_texts, batch_size=len(batch_texts), max_length=8192,
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
                        'indices': list(outputs['lexical_weights'][i].keys()),
                        'values': [float(v) for v in outputs['lexical_weights'][i].values()]
                    }
                }, ensure_ascii=False) + '\n')
                written += 1
    
    print(f'Done! Total written: {written}')

if __name__ == '__main__':
    main()