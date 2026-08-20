#!/usr/bin/env bash
# setup.sh — One-command setup after cloning VoiceRAG
# Usage: bash setup.sh
set -e

echo "🎙️ VoiceRAG Setup"
echo "=================="

# 1. Install Python dependencies
echo ""
echo "📦 Step 1/4: Installing Python dependencies..."
pip install -r voicerag/requirements.txt

# 2. Download Hindi MSMARCO-XI dataset
echo ""
echo "📥 Step 2/4: Downloading Hindi dataset (MSMARCO-XI)..."
cd voicerag
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('data/raw', exist_ok=True)
print('  Downloading MSMARCO-XI Hindi train split...')
hf_hub_download(
    repo_id='ai4bharat/MSMARCO-XI',
    filename='train/hintrain.parquet',
    repo_type='dataset',
    local_dir='data/raw',
    local_dir_use_symlinks=False,
)
print('  Downloading MSMARCO-XI Hindi validation split...')
hf_hub_download(
    repo_id='ai4bharat/MSMARCO-XI',
    filename='validation/hinval.parquet',
    repo_type='dataset',
    local_dir='data/raw',
    local_dir_use_symlinks=False,
)
print('  ✅ Dataset downloaded!')
"

# 3. Build the FAISS + BM25 index
echo ""
echo "🔨 Step 3/4: Building FAISS + BM25 index (first run takes ~2 min)..."
python -c "
import sys; sys.path.insert(0, '.')
from ingest.build_index import build_index
build_index()
print('  ✅ Index built!')
"

# 4. Copy .env.example → .env if not exists
echo ""
echo "⚙️  Step 4/4: Setting up environment..."
cd ..
if [ ! -f voicerag/.env ]; then
    cp voicerag/.env.example voicerag/.env
    echo "  Created voicerag/.env from .env.example"
    echo "  ⚠️  Edit voicerag/.env to add your API keys"
else
    echo "  voicerag/.env already exists — skipping"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "  cd HHgoaVoice"
echo "  python run.py"
echo ""
echo "Then open http://localhost:8000 in your browser."
echo ""
echo "Optional: Install Ollama for local LLM:"
echo "  1. Download from https://ollama.com/download"
echo "  2. Run: ollama pull qwen2.5:1.5b"
echo "  3. The server will auto-detect it."
