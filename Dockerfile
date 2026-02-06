# Dockerfile pour Qwen3-VL avec llama.cpp
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Variables d'environnement pour éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Configuration NVIDIA et CUDA
ENV NVIDIA_VISIBLE_DEVICES=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système exactement comme demandé
RUN apt-get update
RUN apt-get install git pciutils build-essential cmake curl libcurl4-openssl-dev libssl-dev -y

# Clone et compilation de llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp
RUN cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON
RUN cmake --build llama.cpp/build --config Release -j --clean-first
RUN cp llama.cpp/build/bin/llama-* llama.cpp


# Copie du script de démarrage
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Commande par défaut
CMD ["/start.sh"]
#CMD ["/bin/bash", "-c", "while true; do sleep 30; done"]
