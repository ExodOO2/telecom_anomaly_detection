# Lightweight CPU-optimized container for Telecom QoS Anomaly Detection
# Owned by: Person 3 (DevOps & MLOps Engineer)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including OpenMP runtime for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CPU-only PyTorch with PyPI fallback for dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Install API requirements
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application files
COPY ml/ ./ml/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

# Expose FastAPI port
EXPOSE 8000

# Health check to ensure microservice and ML weights are responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
