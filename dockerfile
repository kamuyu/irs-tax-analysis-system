<!-- filepath: /root/IRS/Dockerfile -->
FROM python:3.10-slim

WORKDIR /root/IRS

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /root/IRS/data/docs /root/IRS/data/chroma_db /root/IRS/logs

# Make scripts executable
RUN chmod +x irs.sh

# Environment variables
ENV PYTHONPATH=/root/IRS
ENV OLLAMA_HOST=localhost
ENV OLLAMA_PORT=11434

# Default command
CMD ["./irs.sh", "help"]