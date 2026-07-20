# Lumen API Server — Production Docker Image
# Build:  docker build -t lumen:latest .
# Run:    docker run -p 8848:8848 -v ~/.lumen:/root/.lumen lumen:latest
# Or use docker-compose: docker compose up

FROM python:3.12-slim

LABEL org.opencontainers.image.title="Lumen Memory API"
LABEL org.opencontainers.image.description="Twin-force memory and context framework for sovereign AI agents"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LUMEN_LOG_LEVEL=info
ENV LUMEN_API_HOST=0.0.0.0
ENV LUMEN_API_PORT=8848

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY lumen/ lumen/

RUN pip install --no-cache-dir -e .

VOLUME ["/root/.lumen"]

EXPOSE 8848

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8848/health')" || exit 1

CMD ["uvicorn", "lumen.api.server:app", "--host", "0.0.0.0", "--port", "8848", "--log-level", "info"]