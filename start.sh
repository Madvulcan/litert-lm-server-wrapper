#!/bin/bash
# Quick-start script for LiteRT-LM Server
# Usage: ./start.sh [2b|4b] [port]

set -e

MODEL_SIZE="${1:-4b}"
PORT="${2:-11454}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== LiteRT-LM Server Quick Start ==="
echo "Model: gemma-4-e${MODEL_SIZE}"
echo "Port: ${PORT}"
echo ""

# Check dependencies
echo "[1/4] Checking dependencies..."
pip install -q -r "${SCRIPT_DIR}/requirements.txt"
echo "  ✓ Dependencies installed"

# Check for model
echo "[2/4] Checking for model..."
MODEL_FILE="gemma-4-E${MODEL_SIZE^^}-it.litertlm"
MODEL_PATH="${HOME}/llama-cpp-server/models/${MODEL_FILE}"

if [ -L "${MODEL_PATH}" ] || [ -f "${MODEL_PATH}" ]; then
    echo "  ✓ Model found: ${MODEL_PATH}"
else
    echo "  ✗ Model not found at ${MODEL_PATH}"
    echo ""
    echo "  Download with:"
    echo "    litert-lm download litert-community/gemma-4-E${MODEL_SIZE^^}-it-litert-lm"
    echo ""
    echo "  Or symlink manually:"
    echo "    mkdir -p ~/llama-cpp-server/models"
    echo "    ln -s /path/to/${MODEL_FILE} ~/llama-cpp-server/models/"
    exit 1
fi

# Check GPU
echo "[3/4] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  ✓ GPU detected: ${GPU_MEM} MB VRAM"
else
    echo "  ⚠ nvidia-smi not found — GPU may not be available"
fi

# Start server
echo "[4/4] Starting server on port ${PORT}..."
echo ""
echo "  API URL: http://127.0.0.1:${PORT}"
echo "  Health:  http://127.0.0.1:${PORT}/health"
echo "  Models:  http://127.0.0.1:${PORT}/v1/models"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

cd "${SCRIPT_DIR}"
exec python3 litert-lm-server.py --mode server --port "${PORT}" --model-size "${MODEL_SIZE}"
