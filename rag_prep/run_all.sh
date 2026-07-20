#!/usr/bin/env bash
# run_all.sh - Full RAG pipeline runner

set -euo pipefail

RAG_DIR="/home/vllm/rag_prep"
RAW_DIR="/home/vllm/rag/raw"
JSON_DIR="/home/vllm/rag/raw/json"
EMB_DIR="/home/vllm/rag/raw/embeddings"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

export QDRANT_URL

cd "$RAG_DIR"

echo "=========================================="
echo "RAG Pipeline for Russian Laws"
echo "=========================================="

# Step 1: Fetch
echo ""
echo "Step 1: Fetching laws from zakon-rf..."
python3 01_fetch_zakon_rf.py

# Step 2: Normalize & Chunk
echo ""
echo "Step 2: Normalizing and chunking..."
python3 02_normalize_chunk.py

# Step 3: Embed
echo ""
echo "Step 3: Generating embeddings (bge-m3)..."
python3 03_embed_laws.py

# Step 4: Load to Qdrant
echo ""
echo "Step 4: Loading to Qdrant..."
python3 04_load_qdrant.py

# Step 5: Evaluate
echo ""
echo "Step 5: Evaluating retrieval..."
python3 05_eval_rag.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="