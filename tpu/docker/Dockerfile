# Образ с LinQ tpu_framework 2026.9.45 для офлайн-компиляции ONNX -> программа TPU.
# SDK поставляется только колёсами cp312 linux_x86_64 (tpu_framework.pdf, стр. 5: Ubuntu 18.04+, Python 3.12.11),
# поэтому на маке arm64 собираем amd64-образ (podman/docker + Rosetta).
#
# Сборка (контекст — каталог packets из распакованного SDK):
#   docker build --platform linux/amd64 -t linq-sdk:2026.9.45 \
#     -f tpu/docker/Dockerfile /Users/nagimov-timur/Desktop/mosprom/vendor/linq_sdk/tpu_framework/packets
FROM --platform=linux/amd64 docker.io/library/python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/linq/packets/

# torch ставим CPU-сборкой (2.9.1+cpu удовлетворяет пину ==2.9.1), иначе тянутся гигабайты CUDA.
RUN pip install -U pip uv \
    && cd /opt/linq/packets \
    && uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r requirements.txt pillow \
    && rm -rf /root/.cache

WORKDIR /work
