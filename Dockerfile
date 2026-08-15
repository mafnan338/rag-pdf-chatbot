# Combined image: FastAPI backend + static frontend, served from one process.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend
COPY frontend/ ./frontend

ENV USE_TF=0
ENV USE_TORCH=1
ENV DATA_DIR=/app/backend/data

WORKDIR /app/backend

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]