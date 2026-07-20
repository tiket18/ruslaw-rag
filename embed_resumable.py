#!/usr/bin/env python3
"""Resumable embedding with checkpointing. Saves checkpoint every N batches."""
import json, os, gc, signal, sys, time
from datetime import datetime

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_HOME', '/home/vllm/huggingface_cache')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

INPUT  = '/home/vllm/rag/chunks/law_chunks.jsonl'
OUTPUT = '/home/vllm/rag/embeddings/law_chunks_emb.jsonl'
CHECK  = '/home/vllm/rag/embeddings/.checkpoint.json'
BATCH  = 256
CKPT_EVERY = 10

shutdown_requested = False

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def handler(signum, frame):
    global shutdown_requested
    log('SIGNAL received — finishing current batch')
    shutdown_requested = True

signal.signal(signal.SIGINT,  handler)
signal.signal(signal.SIGTERM, handler)

def get_last_processed():
    if os.path.exists(CHECK):
        try:
            with open(CHECK) as f:
                cp = json.load(f)
                return cp['last_id'], cp['written']
        except: pass
    if os.path.exists(OUTPUT):
        try:
            with open(OUTPUT, 'rb') as f:
                f.seek(-262144, 2)
                tail = f.read().decode('utf-8', errors='replace')
            for line in reversed(tail.split('\n')):
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    return d.get('id'), sum(1 for _ in open(OUTPUT))
                except: pass
        except: pass
    return None, 0

def save_checkpoint(last_id, written):
    tmp = CHECK + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'last_id': last_id, 'written': written}, f)
    os.rename(tmp, CHECK)

def main():
    global shutdown_requested

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    resume_id, already_written = get_last_processed()
    log(f'Resume ID: {resume_id}  Already written: {already_written}')

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.90)

    log('Loading model on GPU...')
    model = BGEM3FlagModel('Roflmax/bge-m3-legal-ru-cocktail-40-60', device='cuda', use_fp16=True)
    log('Model loaded')

    written = already_written
    batch_texts = []
    batch_docs = []
    batches_done = 0
    skipping = resume_id is not None

    def process_batch():
        nonlocal written, batches_done
        outputs = model.encode(
            batch_texts, batch_size=BATCH, max_length=8192,
            return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        with open(OUTPUT, 'a') as fout:
            for i, doc in enumerate(batch_docs):
                sd = outputs['lexical_weights'][i]
                fout.write(json.dumps({
                    'id': doc['id'],
                    'parent_id': doc['parent_id'],
                    'text': doc['text'],
                    'chunk_index': doc.get('chunk_index', 0),
                    'total_chunks': doc.get('total_chunks', 0),
                    'source': doc['source'],
                    'dense': outputs['dense_vecs'][i].tolist(),
                    'sparse': {
                        'indices': list(sd.keys()),
                        'values': [float(v) for v in sd.values()]
                    }
                }, ensure_ascii=False) + '\n')
        written += len(batch_docs)
        batches_done += 1
        if batches_done % CKPT_EVERY == 0:
            save_checkpoint(batch_docs[-1]['id'], written)
            log(f'Checkpoint: {written}')
        if written % 5000 == 0:
            log(f'Processed: {written} chunks')

    with open(INPUT) as fin:
        for line in fin:
            if shutdown_requested:
                log('Shutdown requested — saving checkpoint')
                break
            line = line.strip()
            if not line: continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = doc.get('id', '')
            if skipping:
                if cid == resume_id:
                    log(f'Found resume point: {cid}')
                    skipping = False
                else:
                    continue
            batch_texts.append(doc['text'])
            batch_docs.append(doc)
            if len(batch_texts) >= BATCH:
                process_batch()
                batch_texts.clear()
                batch_docs.clear()
                gc.collect()
                torch.cuda.empty_cache()

    if batch_texts:
        process_batch()

    save_checkpoint(None, written)
    log(f'Done. Total chunks written: {written}')

if __name__ == '__main__':
    main()
