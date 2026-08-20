FROM python:3.11-slim

WORKDIR /app

# Install system deps for faiss-cpu and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY voicerag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model at build time
# so it doesn't need network at runtime
RUN python -c "\
import os; os.environ['HF_HOME']='/app/voicerag/data/hf_cache'; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('Model downloaded successfully')"

# Copy the application code and data
COPY voicerag/ voicerag/

# Set offline mode so sentence_transformers doesn't try to reach network
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/app/voicerag/data/hf_cache
ENV VECTOR_DB_PATH=/app/voicerag/data/index
ENV PYTHONUNBUFFERED=1

WORKDIR /app/voicerag

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
