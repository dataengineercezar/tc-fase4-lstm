FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir .

RUN mkdir -p /app/artifacts

EXPOSE 8000

CMD ["uvicorn", "stock_lstm.api:app", "--host", "0.0.0.0", "--port", "8000"]

