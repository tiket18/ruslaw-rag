#!/usr/bin/env python3
"""
Resumable embedding script with checkpointing.
Processes chunks in batches, saves progress, resumes from last checkpoint.
"""
import json
import sys
import os
import gc
import signal
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# Configuration
INPUT_FILE = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT_FILE = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
CHECKPOINT_FILE = '/home/vllm/rag/embeddings/.checkpoint.json'
BATCH_SIZE = 32

os.makedirs('/home/vllm/rag/embeddings', exist_ok=True)

# Environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/home/vllm/huggingface_cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Signal handling
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\nReceived signal, finishing current batch and saving checkpoint...", flush=True)
    globals()['shutdown_requested'] = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_last_processed_id():
    """Get the last processed chunk ID from checkpoint or output file."""
    CHECKPOINT_FILE = '/home/vllm/rag/embeddings/.checkpoint.json'
    OUTPUT_FILE = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
    
    # Check checkpoint file first
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                cp = json.load(f)
                return cp.get('last_id'), cp.get('written', 0)
        except:
            pass
    
    # Fallback: scan output file
    if os.path.exists(OUTPUT_FILE):
        last_id = None
        count = 0
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    last_id = d.get('id')
                    count += 1
                except:
                    pass
        if last_id:
            return last_id, count
    return None, 0

def save_checkpoint(last_id, written):
    """Save checkpoint atomically."""
    CHECKPOINT_FILE = '/home/vllm/rag/embeddings/.checkpoint.json'
    tmp = CHECKPOINT_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'last_id': last_id, 'written': written}, f)
    os.rename(tmp, '/home/vllm/rag/embeddings/.checkpoint.json')

def load_model():
    """Load model with memory optimizations."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/home/vllm/huggingface_cache'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    import torch
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)
    
    from FlagEmbedding import BGEM3FlagModel
    print("Loading model on GPU...", flush=True)
    model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cuda', use_fp16=True)
    return model

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

def process_batch(model, texts, docs, output_file):
    """Process a batch of texts and write embeddings."""
    outputs = model.encode(texts, batch_size=32, max_length=8192,
                           return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    with open(OUTPUT_FILE, 'a') as fout:
        for i, doc in enumerate(docs):
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

def main():
    global shutdown_requested
    shutdown_requested = False
    
    INPUT_FILE = '/home/vllm/rag/chunks/law_chunks.jsonl'
    OUTPUT_FILE = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
    CHECKPOINT_FILE = '/home/vllm/rag/embeddings/.checkpoint.json'
    BATCH_SIZE = 32
    CHECKPOINT_INTERVAL = 50
    
    # Load checkpoint
    resume_id, already_written = get_last_processed_id()
    print(f"Resuming from: {resume_id}, already written: {already_written}", flush=True)
    
    # Environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/home/vllm/huggingface_cache'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # Load model
    import torch
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)
    
    from FlagEmbedding import BGEM3FlagModel
    print("Loading model on GPU...", flush=True)
    model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cuda', use_fp16=True)
    print("Model loaded", flush=True)
    
    # Signal handling
    def signal_handler(signum, frame):
        globals()['shutdown_requested'] = True
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Processing
    INPUT_FILE = '/home/vllm/rag/chunks/law_chunks.jsonl'
    OUTPUT_FILE = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
    CHECKPOINT_FILE = '/home/vllm/rag/embeddings/.checkpoint.json'
    BATCH_SIZE = 32
    CHECKPOINT_INTERVAL = 50
    
    written = 0
    batch_texts = []
    batch_docs = []
    batch_count = 0
    shutdown_requested = False
    
    # Skip already processed
    resume_id, already_written = get_last_processed_id()
    skipping = resume_id is not None
    print(f"Resuming from: {resume_id}, already written: {already_written}", flush=True)
    
    def process_batch(model, texts, docs, output_file):
        outputs = model.encode(texts, batch_size=32, max_length=8192,
                               return_dense=True, return_sparse=True, return_colbert_vecs=False)
        with open(OUTPUT_FILE, 'a') as fout:
            for i, doc in enumerate(docs):
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
    
    written = 0
    batch_texts = []
    batch_docs = []
    shutdown_requested = False
    skipping = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'a') as fout:
        for line in fin:
            if shutdown_requested:
                print("\nShutdown requested, saving checkpoint...", flush=True)
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            chunk_id = doc.get('id', '')
            
            # Skip already processed
            if skipping:
                if chunk_id == resume_id:
                    print(f"Resuming at {chunk_id}", flush=True)
                    skipping = False
                else:
                    continue
            
            batch_texts.append(doc['text'])
            batch_docs.append(doc)
            
            if len(batch_texts) >= 32:
                process_batch(model, batch_texts, batch_docs, OUTPUT_FILE)
                written += len(batch_docs)
                batch_texts.clear()
                batch_docs.clear()
                
                if written % 10000 == 0:
                    print(f"Written: {written}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Flush remaining
        if batch_texts:
            process_batch(model, batch_texts, batch_docs, OUTPUT_FILE)
            written += len(batch_docs)
        
        print(f"Done! Total written: {written}", flush=True)
        save_checkpoint(None, written)

if __name__ == '__main__':
    main()