# TheBrain - Qwen3-VL llama.cpp Server

**TheBrain** - Named after the French film "Le Cerveau" (The Brain), this deployment provides an OpenAI-compatible API server powered by llama.cpp and the Qwen3-VL vision-language model.

## 🚀 Features

- ✅ **Native llama.cpp** - Optimized performance with CUDA acceleration
- ✅ **OpenAI Compatible API** - Standard `/v1/chat/completions`, `/v1/models` endpoints
- ✅ **Qwen3-VL GGUF** - Quantized vision-language model Q4_K_XL (~5GB)
- ✅ **Multi-Modal** - Text + image support with 49K context window
- ✅ **Streaming** - Real-time response streaming
- ✅ **GPU Accelerated** - Full CUDA support
- ✅ **Docker Ready** - Simplified deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docker        │───▶│  llama.cpp       │───▶│  OpenAI API     │
│   (CUDA)        │    │  llama-server    │    │  Compatible     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NVIDIA        │    │  Qwen3-VL-8B     │    │  /health        │
│   Runtime       │    │  GGUF Q4_K_XL    │    │  /v1/models     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites

- **NVIDIA GPU** with 6+ GB VRAM
- **Docker + NVIDIA Container Runtime**
- **CUDA 12.8** or compatible

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd thebrain

# Configuration (optional)
cp .env.example .env
# Edit .env if needed

# Launch
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

### GPU Configuration

```bash
# Check NVIDIA support
docker run --rm --runtime=nvidia nvidia/cuda:12.8-base nvidia-smi

# Change GPU (in .env)
echo "CUDA_VISIBLE_DEVICES=1" >> .env
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST_PORT` | `8000` | Host port mapping |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU to use |

### llama.cpp Parameters

The server is configured in [`start.sh`](start.sh) with:

```bash
./llama.cpp/llama-server \
    -hf unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL \
    --n-gpu-layers 99 \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx-size 49152 \
    --parallel 2 \
    --flash-attn on
```

## 📚 Usage

### Basic Chat Completion

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Not used
)

response = client.chat.completions.create(
    model="unsloth/Qwen3-VL-8B-Instruct-GGUF",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ],
    max_tokens=1000,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Chat with Images

```python
response = client.chat.completions.create(
    model="unsloth/Qwen3-VL-8B-Instruct-GGUF",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="unsloth/Qwen3-VL-8B-Instruct-GGUF",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Direct cURL

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'

# List models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health
```

## 🔍 API Endpoints

### OpenAI Compatible
- `POST /v1/chat/completions` - Chat completions with vision support
- `GET /v1/models` - List available models

### llama.cpp Native
- `POST /completion` - Simple text completion
- `GET /health` - Service health status
- `POST /tokenize` - Text tokenization
- `POST /detokenize` - Token detokenization

## ⚡ Performance

### Tested Specifications

| GPU | VRAM | Model | Quantization | Performance |
|-----|------|-------|--------------|-------------|
| RTX 4090 | 24GB | Qwen3-VL-8B | Q4_K_XL | ~25 tokens/s |
| RTX 4080 | 16GB | Qwen3-VL-8B | Q4_K_XL | ~20 tokens/s |
| RTX 4070 | 12GB | Qwen3-VL-8B | Q4_K_XL | ~15 tokens/s |
| RTX 3080 | 10GB | Qwen3-VL-8B | Q4_K_XL | ~12 tokens/s |

### GGUF Optimizations

- **Memory**: ~5GB VRAM (vs ~15GB FP16)
- **Speed**: Native C++ performance
- **Context**: Support up to 49K tokens

## 🛠️ Development

### Project Structure

```
thebrain/
├── Dockerfile               # Docker image with llama.cpp + CUDA
├── start.sh                 # llama-server startup script
├── docker-compose.yml       # Container orchestration
├── .env.example            # Environment variables
└── README.md               # This documentation
```

### Local Build

```bash
# Build image
docker build -t llama-qwen:latest .

# Local test
docker run --rm --runtime=nvidia \
  -p 8000:8000 \
  llama-qwen:latest
```

### Monitoring

```bash
# Real-time logs
docker logs -f llama-qwen-server

# GPU metrics
watch -n 1 nvidia-smi

# Health status
curl http://localhost:8000/health
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory Error**
```bash
# Use GPU with more VRAM
export CUDA_VISIBLE_DEVICES=1

# Or reduce context in start.sh
--ctx-size 32768
```

**2. Model Won't Download**
```bash
# Check logs
docker logs llama-qwen-server

# Check disk space
df -h
```

**3. No GPU Detected**
```bash
# Check NVIDIA runtime
docker run --rm --runtime=nvidia nvidia/cuda:12.8-base nvidia-smi

# Install nvidia-container-toolkit
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

## 📖 API Documentation

llama.cpp does **not include an automatic documentation frontend** like FastAPI's `/docs`.

### Available Resources
- **Endpoints**: Test directly with curl/Postman
- **Official Documentation**: [llama.cpp server README](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
- **OpenAI API Reference**: Compatible with [OpenAI Chat API](https://platform.openai.com/docs/api-reference/chat)

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the inference engine
- [Qwen Team](https://github.com/QwenLM/Qwen2-VL) for the model
- [Unsloth](https://unsloth.ai/) for GGUF quantization