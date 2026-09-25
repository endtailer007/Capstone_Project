# ─────────────────────────────────────────────────────────────
# Zepto Support Assistant — FastAPI container
# ─────────────────────────────────────────────────────────────
# Build:  docker build -t zepto-support .
# Run:    docker run -p 7860:7860 zepto-support
# Test:   curl -X POST http://localhost:7860/ask \
#              -H "Content-Type: application/json" \
#              -d "{\"query\": \"What is the delivery fee?\"}"
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Keeps Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    # Disable HF progress bars inside the build layer
    TRANSFORMERS_VERBOSITY=error \
    # Store the downloaded model inside the image
    SENTENCE_TRANSFORMERS_HOME=/app/.model_cache

WORKDIR /app

# ── 1. Install system build tools needed by some packages ──
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# ── 2. Install Python dependencies ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 3. Pre-download the embedding model into the image ───────
#    This avoids a live download on every `docker run`.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── 4. Copy application source and policy documents ──────────
COPY app.py             .
COPY zepto_assistant.py .
COPY doc*.txt           ./

# ── 5. Expose the service port ───────────────────────────────
EXPOSE 7860

# ── 6. Start the API server ──────────────────────────────────
#    ChromaDB is built from the .txt files at startup (lifespan hook).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
