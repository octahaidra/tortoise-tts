FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA=1 \
    TORTOISE_MODELS_DIR="/models"

# Copy the entire project
COPY . /workspace/

# Install CUDA-enabled PyTorch first
RUN pip3 install --no-cache-dir \
    torch==2.2.2+cu122 \
    torchaudio==2.2.2+cu122 \
    --extra-index-url https://download.pytorch.org/whl/cu122

# Install other dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional required packages for RunPod
RUN pip3 install --no-cache-dir \
    deepspeed==0.13.2 \
    runpod==1.6.0 \
    numba \
    inflect

# Install the package itself
RUN pip3 install -e .

# Create model directory
RUN mkdir -p /models

# RunPod specific environment variables
ENV RUNPOD_DEBUG_LEVEL=DEBUG \
    RUNPOD_ENABLE_GPU_METRICS=1

# Default command (using the handler from the copied project)
CMD [ "python3", "-u", "src/rp_handler.py" ]
