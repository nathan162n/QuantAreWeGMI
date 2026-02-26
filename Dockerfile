FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data logs reports/output

RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.database.engine import create_engine_with_fallback; create_engine_with_fallback()" || exit 1

CMD ["python", "main.py", "--paper"]
