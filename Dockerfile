FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# PyTorch
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Pinned versions — unpinned git-installs caused a load-time OOM regression on 32GB cards
RUN python -m pip install \
    "diffusers==0.32.1" \
    "transformers==4.47.1" \
    "accelerate==1.2.1" \
    "bitsandbytes==0.45.0" \
    "safetensors>=0.4.5" \
    sentencepiece protobuf

# RunPod + utilities
RUN python -m pip install runpod requests Pillow

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
