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

# No hard memory limit - use available RAM
# resource.setrlimit(resource.RLIMIT_AS, (14 * 1024 * 1024 * 1024, 14 * 1024 * 1024 * 1024))

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

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HOME'] = '/home/vllm/huggingface_cache'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    print('Loading model on CPU...')
    model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cpu', use_fp16=False)
    print('Model loaded on CPU')
    
    batch_size = 96  # Larger batch for CPU
    batch_texts = []
    batch_docs = []
    written = 0
    
    with open(INPUT, 'r') as fin, open(OUTPUT, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            doc = json.loads(line)
            batch_texts.append(doc['text'])
            batch_docs.append(doc)
            
            if len(batch_texts) >= 96:
                outputs = model.encode(batch_texts, batch_size=96, max_length=8192,
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
                written += len(batch_docs)
                batch_texts.clear()
                batch_docs.clear()
                gc.collect()
                
                if written % 1000 == 0:
                    print(f'Written: {written}', flush=True)
        
        # Flush remaining
        if batch_texts:
            outputs = model.encode(batch_texts, batch_size=len(batch_texts), max_length=8192,
                                   return_dense=True, return_sparse=True, return_colbert_vecs=False)
            for i, doc in enumerate(batch_docs):
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