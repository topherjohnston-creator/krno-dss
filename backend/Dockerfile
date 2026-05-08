FROM python:3.11-slim

# Install ecCodes C library — package name is libeccodes3 on Debian Trixie
RUN apt-get update && apt-get install -y \
    libeccodes3 \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
