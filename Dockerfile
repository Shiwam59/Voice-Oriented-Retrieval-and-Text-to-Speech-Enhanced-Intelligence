FROM python:3.11-slim

WORKDIR /app

# Install system deps for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY voicerag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model at build time
RUN python -c "\
import os; os.environ['HF_HOME='/app/voicerag/data/hf_cache']; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('Model downloaded successfully')"

# Copy the application code
COPY voicerag/ voicerag/
COPY run.py .

# Create directories for data (will be built on first start)
RUN mkdir -p voicerag/data/index voicerag/data/raw voicerag/data/hf_cache

# Set environment
ENV HF_HOME=/app/voicerag/data/hf_cache
ENV VECTOR_DB_PATH=/app/voicerag/data/index
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Auto-build index on first start, then run server
CMD ["sh", "-c", "python run.py"]
