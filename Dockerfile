# Dockerfile pour Qwen3-VL FastAPI avec vLLM
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Métadonnées
LABEL maintainer="Qwen3-VL API Server"
LABEL description="Serveur FastAPI avec Qwen3-VL via vLLM, OpenAI compatible avec function calling"
LABEL version="2.0.0"

# Variables d'environnement pour éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Configuration NVIDIA et CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Variables d'environnement pour l'API
ENV API_PREFIX=""
ENV ROOT_PATH=""
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV WORKERS="1"

# Configuration du modèle Qwen3-VL (2B pour 6GB VRAM)
ENV MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
ENV MAX_SEQ_LENGTH="8192"
ENV GPU_MEMORY_UTILIZATION="0.9"
ENV MAX_MODEL_LEN="8192"

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Création d'un environnement virtuel Python
RUN python3 -m venv /opt/venv

# Activation de l'environnement virtuel
ENV PATH="/opt/venv/bin:$PATH"

# Mise à jour de pip dans l'environnement virtuel
RUN pip install --upgrade pip setuptools wheel

# Répertoire de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation de PyTorch avec support CUDA comme avant
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Installation de vLLM et autres dépendances
RUN pip install -r requirements.txt

# Copie du code source
COPY app/ ./app/

# Création des répertoires nécessaires
RUN mkdir -p /app/logs /app/cache /root/.cache/huggingface

# Exposition du port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000${API_PREFIX}/health || exit 1

# Script de démarrage qui lance d'abord vLLM server puis FastAPI
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Commande par défaut
CMD ["/start.sh"]

# Configuration des volumes recommandés
VOLUME ["/app/logs", "/app/cache", "/root/.cache/huggingface"]

# Labels vLLM
LABEL cuda.version="12.1"
LABEL python.version="3.11"
LABEL vllm="true"
LABEL openai.compatible="true"
LABEL function.calling="true"
LABEL multimodal="true"
LABEL compute.capability="sm75,sm120"