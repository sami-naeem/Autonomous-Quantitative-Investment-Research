#!/bin/bash
# Lambda Instance Setup Script for vLLM with Fine-tuned Llama 3.1 8B
# Docker-based version — works on both x86_64 and aarch64 (GH200)
# Run this immediately after SSH into the instance

set -e  # Exit on any error

echo "=========================================="
echo "Lambda Instance Setup for vLLM + LoRA"
echo "        (Docker-based vLLM)"
echo "=========================================="

# ===========================================
# CONFIGURATION - Edit these values as needed
# ===========================================
HF_TOKEN="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"

# Base model configuration (updated to Llama 3.1 8B)
BASE_MODEL_REPO="meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_NAME="llama-3.1-8b"

# Your fine-tuned LoRA adapter (uploaded via scp from local machine)
LORA_ADAPTER_NAME="finllama-lora"

# vLLM Docker image — auto-detected by architecture, or override via env var
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    VLLM_IMAGE="${VLLM_IMAGE:-drikster80/vllm-gh200-openai}"
    # GH200 image uses "vllm serve" entrypoint — model is a positional arg
    MODEL_FLAG=""
    echo "Detected aarch64 (GH200/ARM) — using image: ${VLLM_IMAGE}"
else
    VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
    # Official image uses --model flag
    MODEL_FLAG="--model "
    echo "Detected ${ARCH} — using image: ${VLLM_IMAGE}"
fi

# vLLM serving configuration
API_KEY="token-abc123"
PORT=8000

# Expected local files to upload:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
# - README.md

# 0. Ensure Docker access for current user
echo ""
echo "[0/10] Ensuring Docker access..."
if ! docker info > /dev/null 2>&1; then
    echo "  Adding $USER to docker group..."
    sudo usermod -aG docker $USER
    echo "  Applying group change..."
    sg docker -c "docker info > /dev/null 2>&1" && echo "  Docker access confirmed." || {
        echo "  ERROR: Could not get Docker access. Please run:"
        echo "    sudo usermod -aG docker \$USER && newgrp docker"
        echo "  Then re-run this script."
        exit 1
    }
    # Re-exec the script under the new group so all subsequent docker calls work
    exec sg docker "$0"
fi
echo "  Docker access confirmed."

# 1. Install Lambda Guest Agent (for metrics monitoring)
echo ""
echo "[1/10] Installing Lambda Guest Agent..."
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

echo ""
echo "Verifying Guest Agent status..."
sudo systemctl --no-pager status lambda-guest-agent* || echo "Warning: Guest Agent may not have started correctly"

# 2. Install uv package manager for fast installations
echo ""
echo "[2/10] Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Create virtual environment (with system-site-packages for CUDA PyTorch access)
echo ""
echo "[3/10] Creating virtual environment..."
uv venv --system-site-packages ~/vllm_env
source ~/vllm_env/bin/activate

# 4. Install utility dependencies (NOT vllm — that runs in Docker)
echo ""
echo "[4/10] Installing utility dependencies with uv..."
uv pip install --reinstall huggingface_hub
uv pip install "numpy>1.22.4"
uv pip install peft  # Required for LoRA merging

# 5. Pull vLLM Docker image
echo ""
echo "[5/10] Pulling vLLM Docker image: ${VLLM_IMAGE}..."
docker pull ${VLLM_IMAGE}
echo "Verifying Docker image..."
docker inspect ${VLLM_IMAGE} > /dev/null 2>&1 && echo "Docker image ready: ${VLLM_IMAGE}" || { echo "ERROR: Failed to pull Docker image"; exit 1; }

# 6. Login to Hugging Face
echo ""
echo "[6/10] Logging into Hugging Face..."
export HF_TOKEN=$HF_TOKEN
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# 7. Create directory structure
echo ""
echo "[7/10] Creating directory structure..."
mkdir -p ~/models/${BASE_MODEL_NAME}
mkdir -p ~/lora_adapters/${LORA_ADAPTER_NAME}
mkdir -p ~/scripts

# 8. Download base model
echo ""
echo "[8/10] Downloading base model: $BASE_MODEL_REPO"
python -c "from huggingface_hub import snapshot_download; snapshot_download('$BASE_MODEL_REPO', local_dir='$HOME/models/${BASE_MODEL_NAME}')"

# 9. Setup LoRA adapters
echo ""
echo "[9/10] Setting up LoRA adapters directory..."
echo ""
echo "=========================================="
echo "UPLOAD YOUR LORA ADAPTERS"
echo "=========================================="
echo ""
echo "From your Mac, run:"
echo ""
echo 'scp -r "/Users/campbelltaylor/Downloads/Misc/University of Chicago/Capstone_2/Autonomous-Quantitative-Investment-Research/finetuning_model/lora_hp_search/lora_adapters/"* ubuntu@$(curl -s ifconfig.me):~/lora_adapters/finllama-lora/'
echo ""
echo "Expected files:"
echo "  - adapter_config.json"
echo "  - adapter_model.safetensors"
echo "  - tokenizer.json"
echo "  - tokenizer_config.json"
echo "  - special_tokens_map.json"
echo ""

# 10. Create vLLM startup scripts (Docker-based)
echo ""
echo "[10/10] Creating vLLM startup scripts (Docker-based)..."

# Script to run BASE model only (no LoRA)
cat > ~/scripts/start_vllm_base.sh << EOF
#!/bin/bash
# vLLM Server - Base Model Only (Docker)

# Stop any existing vllm container
docker stop vllm-server 2>/dev/null && docker rm vllm-server 2>/dev/null || true

echo "Starting vLLM base model server via Docker..."
docker run -d --name vllm-server \\
    --gpus all \\
    --ipc=host \\
    -v \$HOME/models:/models \\
    -p ${PORT}:${PORT} \\
    ${VLLM_IMAGE} \\
    ${MODEL_FLAG}/models/${BASE_MODEL_NAME} \\
    --host 0.0.0.0 \\
    --port ${PORT} \\
    --api-key "${API_KEY}" \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 8192 \\
    --disable-log-requests

echo "Container started. Waiting for vLLM to load model..."
echo "  Monitor logs: docker logs -f vllm-server"
echo "  Test:         curl http://localhost:${PORT}/v1/models -H 'Authorization: Bearer ${API_KEY}'"
EOF

# Script to run with LoRA adapter
cat > ~/scripts/start_vllm_lora.sh << EOF
#!/bin/bash
# vLLM Server - With LoRA Adapter (Docker)

# Stop any existing vllm container
docker stop vllm-server 2>/dev/null && docker rm vllm-server 2>/dev/null || true

echo "Starting vLLM server with LoRA adapter via Docker..."
docker run -d --name vllm-server \\
    --gpus all \\
    --ipc=host \\
    -v \$HOME/models:/models \\
    -v \$HOME/lora_adapters:/lora_adapters \\
    -p ${PORT}:${PORT} \\
    ${VLLM_IMAGE} \\
    ${MODEL_FLAG}/models/${BASE_MODEL_NAME} \\
    --host 0.0.0.0 \\
    --port ${PORT} \\
    --api-key "${API_KEY}" \\
    --enable-lora \\
    --lora-modules finllama=/lora_adapters/${LORA_ADAPTER_NAME} \\
    --max-lora-rank 64 \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 8192 \\
    --disable-log-requests

echo "Container started. Waiting for vLLM to load model + LoRA..."
echo "  Monitor logs: docker logs -f vllm-server"
echo "  Test:         curl http://localhost:${PORT}/v1/models -H 'Authorization: Bearer ${API_KEY}'"
echo "  Use model:    model='finllama' in API calls"
EOF

# Script to merge LoRA into base model (runs on host, not Docker)
cat > ~/scripts/merge_lora.py << 'EOF'
#!/usr/bin/env python3
"""
Merge LoRA adapter into base model to create a standalone fine-tuned model.
This eliminates the need for --enable-lora at runtime.
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Configuration
BASE_MODEL_PATH = os.path.expanduser("~/models/llama-3.1-8b")
LORA_ADAPTER_PATH = os.path.expanduser("~/lora_adapters/finllama-lora")
MERGED_MODEL_PATH = os.path.expanduser("~/models/finllama-merged")

print(f"Loading base model from: {BASE_MODEL_PATH}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading tokenizer from: {BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

print("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to: {MERGED_MODEL_PATH}")
os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print("Done! Merged model saved.")
print(f"\nTo run the merged model:")
print(f"  ~/scripts/start_vllm_merged.sh")
EOF

# Script to run merged model (after merge_lora.py)
cat > ~/scripts/start_vllm_merged.sh << EOF
#!/bin/bash
# vLLM Server - Merged Fine-tuned Model (Docker, no LoRA loading needed)

# Stop any existing vllm container
docker stop vllm-server 2>/dev/null && docker rm vllm-server 2>/dev/null || true

echo "Starting vLLM merged model server via Docker..."
docker run -d --name vllm-server \\
    --gpus all \\
    --ipc=host \\
    -v \$HOME/models:/models \\
    -p ${PORT}:${PORT} \\
    ${VLLM_IMAGE} \\
    ${MODEL_FLAG}/models/finllama-merged \\
    --host 0.0.0.0 \\
    --port ${PORT} \\
    --api-key "${API_KEY}" \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 8192 \\
    --disable-log-requests

echo "Container started. Waiting for vLLM to load merged model..."
echo "  Monitor logs: docker logs -f vllm-server"
echo "  Test:         curl http://localhost:${PORT}/v1/models -H 'Authorization: Bearer ${API_KEY}'"
EOF

# Script to stop vLLM server
cat > ~/scripts/stop_vllm.sh << 'EOF'
#!/bin/bash
# Stop vLLM Docker container

if docker ps -q -f name=vllm-server | grep -q .; then
    echo "Stopping vllm-server..."
    docker stop vllm-server && docker rm vllm-server
    echo "vLLM server stopped."
else
    echo "No running vllm-server container found."
    # Clean up stopped container if it exists
    docker rm vllm-server 2>/dev/null || true
fi
EOF

# Make all scripts executable
chmod +x ~/scripts/*.sh
chmod +x ~/scripts/*.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Base Model: $BASE_MODEL_REPO"
echo "  Location: ~/models/${BASE_MODEL_NAME}"
echo ""
echo "LoRA Adapter Location: ~/lora_adapters/${LORA_ADAPTER_NAME}"
echo ""
echo "Docker Image: ${VLLM_IMAGE}"
echo "Architecture: ${ARCH}"
echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Upload your LoRA adapters from your Mac:"
echo '   scp -r "/Users/campbelltaylor/Downloads/Misc/University of Chicago/Capstone_2/Autonomous-Quantitative-Investment-Research/finetuning_model/lora_hp_search/lora_adapters/"* ubuntu@<LAMBDA_IP>:~/lora_adapters/finllama-lora/'
echo ""
echo "2. Verify upload:"
echo "   ls -la ~/lora_adapters/finllama-lora/"
echo ""
echo "3. Choose how to deploy:"
echo ""
echo "   OPTION A - Run with dynamic LoRA loading:"
echo "   ~/scripts/start_vllm_lora.sh"
echo "   Then call API with: model='finllama'"
echo ""
echo "   OPTION B - Merge LoRA into base model first (recommended):"
echo "   source ~/vllm_env/bin/activate"
echo "   python ~/scripts/merge_lora.py"
echo "   ~/scripts/start_vllm_merged.sh"
echo ""
echo "   OPTION C - Base model only (no LoRA):"
echo "   ~/scripts/start_vllm_base.sh"
echo ""
echo "4. Monitor the server:"
echo "   docker logs -f vllm-server"
echo ""
echo "5. Test the API:"
echo "   curl http://localhost:${PORT}/v1/models -H 'Authorization: Bearer ${API_KEY}'"
echo ""
echo "6. Stop the server:"
echo "   ~/scripts/stop_vllm.sh"
echo ""
echo "=========================================="
