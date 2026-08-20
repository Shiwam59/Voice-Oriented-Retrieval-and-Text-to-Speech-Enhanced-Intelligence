FROM python:3.11-slim

WORKDIR /app

# Install system deps for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY voicerag/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY voicerag/ ./voicerag/
COPY run.py .

# Create data directories
RUN mkdir -p voicerag/data/index voicerag/data/raw voicerag/data/hf_cache

# Environment
ENV HF_HOME=/app/voicerag/data/hf_cache
ENV VECTOR_DB_PATH=/app/voicerag/data/index
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "run.py"]
