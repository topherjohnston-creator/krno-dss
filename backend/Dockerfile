FROM python:3.11-slim

# Install ecCodes C library — required by cfgrib at runtime
RUN apt-get update && apt-get install -y \
    libeccodes2 \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Expose port and run
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
