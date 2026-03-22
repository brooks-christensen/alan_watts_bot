FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Minimal runtime packages. build-essential is intentionally omitted.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for the service.
RUN groupadd --system app && useradd --system --gid app --create-home app

# Copy only runtime dependency manifest first for better layer caching.
COPY requirements-api.txt ./requirements-api.txt
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Copy the application code and runtime assets.
COPY config ./config
COPY data/indexes ./data/indexes
COPY src ./src
COPY app.py ./app.py

RUN chown -R app:app /app
USER app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
