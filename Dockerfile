FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install llama.cpp for quantization (optional)
RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    make && \
    pip3 install -e .

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/checkpoints models/quantized results/evaluation

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python3", "scripts/deploy_api.py", "--host", "0.0.0.0", "--port", "8000"]
