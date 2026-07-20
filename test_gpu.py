import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/home/vllm/huggingface_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.85)

from FlagEmbedding import BGEM3FlagModel

print("Loading model on GPU...")
model = BGEM3FlagModel("BAAI/bge-m3", device="cuda", use_fp16=True)
print("Model loaded on GPU")

out = model.encode(["test"], batch_size=1, return_dense=True, return_sparse=True)
print(f"Dense shape: {out['dense_vecs'].shape}")
print("Success!")