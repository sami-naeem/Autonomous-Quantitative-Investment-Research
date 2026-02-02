#!/bin/bash
# Lambda Instance Setup Script for vLLM Multi-LoRA Deployment
# Run this immediately after SSH into the instance

set -e  # Exit on any error

echo "=========================================="
echo "Lambda Instance Setup for vLLM Multi-LoRA"
echo "=========================================="

# ===========================================
# CONFIGURATION - Edit these values as needed
# ===========================================
HF_TOKEN="YOUR_HF_TOKEN_HERE"

# Base model configuration
BASE_MODEL_REPO="meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_NAME="llama-3.1-8b"  # Short name for easier reference

# FinAI models
FINLLAVA_REPO="TheFinAI/FinLLaVA"
FIN_O1_REPO="TheFinAI/Fin-o1-14B"

# 0. Install Lambda Guest Agent (for metrics monitoring)
echo ""
echo "[0/9] Installing Lambda Guest Agent..."
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

echo ""
echo "Verifying Guest Agent status..."
sudo systemctl --no-pager status lambda-guest-agent* || echo "Warning: Guest Agent may not have started correctly"

# 1. Install uv package manager for fast installations
echo ""
echo "[1/9] Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create virtual environment
echo ""
echo "[2/9] Creating virtual environment..."
uv venv ~/vllm_env
source ~/vllm_env/bin/activate

# 3. Install vLLM and Hugging Face Hub using uv
echo ""
echo "[3/9] Installing vLLM and Hugging Face Hub with uv..."
# Increase timeout for large CUDA packages (nvidia-cusparse-cu12, etc.)
export UV_HTTP_TIMEOUT=300
uv pip install --upgrade vllm huggingface_hub
uv pip install "numpy>1.22.4"

# 4. Login to Hugging Face
echo ""
echo "[4/9] Logging into Hugging Face..."
export HF_TOKEN=$HF_TOKEN
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# 5. Create directory structure
echo ""
echo "[5/9] Creating directory structure..."
mkdir -p ~/lora_adapters/finllama
mkdir -p ~/lora_adapters/custom
mkdir -p ~/models
mkdir -p ~/scripts

# 6. Download LoRA adapters (update repo names as needed)
echo ""
echo "[6/9] Downloading LoRA adapters..."
# Uncomment and update these with your actual LoRA adapter repos:
# huggingface-cli download <YOUR_FINLLAMA_LORA_REPO> --local-dir ~/lora_adapters/finllama
# huggingface-cli download <YOUR_CUSTOM_LORA_REPO> --local-dir ~/lora_adapters/custom

# 7. Download models
echo ""
echo "[7/9] Downloading models..."
echo "Downloading base model: $BASE_MODEL_REPO"
huggingface-cli download $BASE_MODEL_REPO --local-dir ~/models/$BASE_MODEL_NAME

echo "Downloading FinLLaVA..."
huggingface-cli download $FINLLAVA_REPO --local-dir ~/models/FinLLaVA

echo "Downloading Fin-o1-14B..."
huggingface-cli download $FIN_O1_REPO --local-dir ~/models/Fin-o1-14B

# 8. Create vLLM startup script
echo ""
echo "[8/9] Creating vLLM startup script..."
cat > ~/scripts/start_vllm.sh << EOF
#!/bin/bash
# vLLM Multi-LoRA Server Startup Script
# Configured for external access from anywhere (Colab, etc.)

# Activate virtual environment
source \$HOME/vllm_env/bin/activate

# Configuration variables
BASE_MODEL="\$HOME/models/$BASE_MODEL_NAME"
PORT=8000
MAX_LORA_RANK=64
MAX_MODEL_LEN=16384
GPU_MEMORY_UTIL=0.95
API_KEY="token-abc123"

# LoRA adapter paths
FINLLAMA_LORA_PATH="\$HOME/lora_adapters/finllama"
CUSTOM_LORA_PATH="\$HOME/lora_adapters/custom"

# Start vLLM with base model only (no LoRA)
python -m vllm.entrypoints.openai.api_server \\
    --model \${BASE_MODEL} \\
    --host 0.0.0.0 \\
    --port \${PORT} \\
    --api-key \${API_KEY} \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization \${GPU_MEMORY_UTIL} \\
    --max-model-len \${MAX_MODEL_LEN} \\
    --enable-auto-tool-choice \\
    --tool-call-parser llama3_json \\
    --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
    --disable-log-requests

# To enable LoRA adapters, use this instead:
# python -m vllm.entrypoints.openai.api_server \\
#     --model \${BASE_MODEL} \\
#     --host 0.0.0.0 \\
#     --port \${PORT} \\
#     --api-key \${API_KEY} \\
#     --enable-lora \\
#     --max-lora-rank \${MAX_LORA_RANK} \\
#     --lora-modules finllama=\${FINLLAMA_LORA_PATH} custom=\${CUSTOM_LORA_PATH} \\
#     --max-model-len \${MAX_MODEL_LEN} \\
#     --gpu-memory-utilization \${GPU_MEMORY_UTIL} \\
#     --enable-auto-tool-choice \\
#     --tool-call-parser llama3_json \\
#     --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
#     --disable-log-requests
EOF

# Make vLLM script executable
chmod +x ~/scripts/start_vllm.sh

# 9. Final verification
echo ""
echo "[9/9] Final verification..."
echo "Checking Lambda Guest Agent status..."
sudo systemctl --no-pager status lambda-guest-agent* || true

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Base model: $BASE_MODEL_REPO"
echo "Model path: ~/models/$BASE_MODEL_NAME"
echo ""
echo "Lambda Guest Agent:"
echo "  Status: sudo systemctl status lambda-guest-agent*"
echo "  Logs:   sudo journalctl -u lambda-guest-agent"
echo ""
echo "Directory structure created:"
echo "  ~/lora_adapters/finllama"
echo "  ~/lora_adapters/custom"
echo "  ~/models/$BASE_MODEL_NAME"
echo "  ~/scripts"
echo ""
echo "=========================================="
echo "Starting vLLM server..."
echo "=========================================="
echo ""

# Start vLLM server
~/scripts/start_vllm.sh
