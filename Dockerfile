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

# Diffusers from git main — required for Flux2Pipeline (no released version has it yet).
# Pin to a known-working commit rather than HEAD to keep builds reproducible.
RUN python -m pip install \
    "git+https://github.com/huggingface/diffusers.git@main"

# transformers >= 4.50 required for Mistral3ForConditionalGeneration (FLUX.2 text encoder)
# bitsandbytes >= 0.46 has newer sm_120 (Blackwell) CUDA kernels and bug fixes.
# Note: bnb uses NF4 (software dequant → bf16 matmul), not NVIDIA's native NVFP4.
# Blackwell's FP4 tensor cores are not accessed through bitsandbytes.
RUN python -m pip install \
    "transformers>=4.50,<4.60" \
    "accelerate>=1.2" \
    "bitsandbytes>=0.46" \
    "safetensors>=0.4.5" \
    sentencepiece protobuf

# RunPod + utilities
RUN python -m pip install runpod requests Pillow

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
