#!/bin/bash
set -e

echo "🚀 Démarrage du serveur Qwen3-VL FastAPI avec vLLM"

# Configuration pour 6GB VRAM - RTX 2060
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2-VL-2B-Instruct"}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.85"}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-"8192"}
export PORT=${PORT:-"8000"}

echo "📋 Configuration:"
echo "  - Modèle: $MODEL_NAME"
echo "  - GPU Memory: $GPU_MEMORY_UTILIZATION"
echo "  - Max Length: $MAX_MODEL_LEN"
echo "  - Port: $PORT"

# Vérification CUDA
echo "🔍 Vérification CUDA:"
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Téléchargement du modèle si nécessaire
echo "📥 Vérification du modèle..."
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME')"

echo "✅ Modèle prêt, démarrage de l'API FastAPI avec vLLM intégré"

# Démarrage de l'API FastAPI (vLLM sera initialisé en interne)
exec python -m uvicorn app.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS