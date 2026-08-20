#!/usr/bin/env bash
# setup.sh — One-command setup after cloning VoiceRAG
# Works on: Linux, macOS, Windows (Git Bash), Render.com, Railway, etc.
set -e

echo "🎙️ VoiceRAG Setup"
echo "=================="

# 1. Install Python dependencies
echo ""
echo "📦 Step 1/4: Installing Python dependencies..."
pip install -r voicerag/requirements.txt 2>&1 | tail -3

# 2. Download Hindi MSMARCO-XI dataset
echo ""
echo "📥 Step 2/4: Downloading Hindi dataset (MSMARCO-XI)..."
cd voicerag
mkdir -p data/raw data/index data/hf_cache

python -c "
import sys, os
sys.path.insert(0, '.')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'data', 'hf_cache')
from huggingface_hub import hf_hub_download
os.makedirs('data/raw', exist_ok=True)
print('  Downloading MSMARCO-XI Hindi train split...')
hf_hub_download(
    repo_id='ai4bharat/MSMARCO-XI',
    filename='train/hintrain.parquet',
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
    echo ""
    echo "  ⚠️  IMPORTANT: Edit voicerag/.env and add your API keys:"
    echo "     - SARVAM_API_KEY (free at https://dashboard.sarvam.ai)"
    echo "     - LLM_API_KEY (Groq free at https://console.groq.com)"
else
    echo "  voicerag/.env already exists — skipping"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "  python run.py"
echo ""
echo "Then open http://localhost:8000 in your browser."
echo ""
echo "For cloud deployment (Render/Railway):"
echo "  1. Push to GitHub"
echo "  2. Connect repo to Render/Railway"
echo "  3. Set env vars (SARVAM_API_KEY, LLM_API_KEY) in dashboard"
echo "  4. Deploy — index auto-builds on first start!"
