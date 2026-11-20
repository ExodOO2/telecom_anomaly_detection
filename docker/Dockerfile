# Lightweight CPU-optimized container for Telecom QoS Anomaly Detection
# Owned by: Person 3 (DevOps & MLOps Engineer)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep image lightweight (< 800MB vs > 3GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install API requirements
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application files
COPY ml/ ./ml/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn server
CMD ["uvicorn", "api.app.app", "--host", "0.0.0.0", "--port", "8000"]
