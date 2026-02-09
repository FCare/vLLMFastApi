#!/bin/bash
./llama.cpp/llama-server \
    -hf unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL \
    --n-gpu-layers 99 \
    --jinja \
    --top-p 0.8 \
    --top-k 20 \
    --host 0.0.0.0 \
    --port 8000 \
    --temp 0.7 \
    --parallel 2\
    --min-p 0.0 \
    --flash-attn on \
    --presence-penalty 1.5 \
    --ctx-size 49152
