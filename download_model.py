import os
import sys
sys.path.insert(0, '/opt/vllm/.venv/lib/python3.14/site-packages')

from huggingface_hub import hf_hub_download

model_id = "Roflmax/bge-m3-legal-ru-cocktail-40-60"
local_dir = "/root/.cache/huggingface/hub/models--Roflmax--bge-m3-legal-ru-cocktail-40-60"

files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors"]

for fname in files:
    path = hf_hub_download(
        repo_id="Roflmax/bge-m3-legal-ru-cocktail-40-60",
        filename=fname,
        local_dir="/root/.cache/huggingface/hub/models--Roflmax--bge-m3-legal-ru-cocktail-40-60",
        local_dir_use_symlinks=False,
        token=None
    )
    print(f"Downloaded {fname} -> {path}")

print("Done!")