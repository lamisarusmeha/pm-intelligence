FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for packages that compile from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cargo \
    rustc \
    && rm -rf /var/lib/apt/lists/*

# Pre-install pydantic-core with pre-built wheels before other deps
RUN pip install --no-cache-dir "pydantic-core>=2.27.0" "pydantic>=2.10.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD cd backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
