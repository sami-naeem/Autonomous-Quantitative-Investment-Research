#!/bin/bash
# Lambda Instance Setup Script for vLLM Multi-LoRA Deployment
# Run this immediately after SSH into the instance

set -e  # Exit on any error

echo "=========================================="
echo "Lambda Instance Setup for vLLM Multi-LoRA"
echo "=========================================="

# 0. Install Lambda Guest Agent (for metrics monitoring)
echo ""
echo "[0/7] Installing Lambda Guest Agent..."
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

echo ""
echo "Verifying Guest Agent status..."
sudo systemctl --no-pager status lambda-guest-agent* || echo "Warning: Guest Agent may not have started correctly"

# Set your Hugging Face token here
HF_TOKEN="REDACTED_FOR_SECURITY"

# 1. Install vLLM and Hugging Face Hub with CLI
echo ""
echo "[1/7] Installing vLLM and Hugging Face Hub..."
pip install --upgrade vllm huggingface_hub[cli]
pip install "numpy<2"

# 2. Login to Hugging Face
echo ""
echo "[2/7] Logging into Hugging Face..."
# Set HF_TOKEN as environment variable (used by huggingface_hub automatically)
export HF_TOKEN=$HF_TOKEN
# Programmatic login using Python
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# 3. Create directory structure
echo ""
echo "[3/7] Creating directory structure..."
mkdir -p ~/lora_adapters/custom
mkdir -p ~/models
mkdir -p ~/scripts

# 4. Download LoRA adapters (update repo names as needed)
echo ""
echo "[4/7] Downloading LoRA adapters..."
# Syntax: hf download <repo_id> --local-dir <path>
# Uncomment and update these with your actual repo names:
# hf download <YOUR_CUSTOM_REPO> --local-dir ~/lora_adapters/custom

# 5. Download base model (if needed)
echo ""
echo "[5/7] Downloading base model..."
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ~/models/Llama-3.2-3B-Instruct
hf download TheFinAI/FinLLaVA --local-dir ~/models/FinLLaVA
hf download TheFinAI/Fin-o1-14B --local-dir ~/models/Fin-o1-14B
hf download DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning --local-dir ~/models/Llama3

# 6. Create vLLM startup script
echo ""
echo "[6/7] Creating vLLM startup script..."
cat > ~/scripts/start_vllm.sh << 'EOF'
#!/bin/bash
# vLLM Multi-LoRA Server Startup Script
# Configured for external access from anywhere (Colab, etc.)

# Configuration variables
BASE_MODEL="$HOME/models/Llama3"
PORT=8000
MAX_LORA_RANK=64
MAX_MODEL_LEN=16384
GPU_MEMORY_UTIL=0.95
API_KEY="token-abc123"

# LoRA adapter paths
FINLLAMA_LORA_PATH="$HOME/lora_adapters/finllama"
CUSTOM_LORA_PATH="$HOME/lora_adapters/custom"

# Using DavidAU model while waiting for Meta Llama access approval
python -m vllm.entrypoints.openai.api_server \
    --model ${BASE_MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --api-key ${API_KEY} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --disable-log-requests

# Alternative: Once Meta approves access and LoRA adapters are ready:
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --host 0.0.0.0 \
#     --port ${PORT} \
#     --api-key ${API_KEY} \
#     --enable-lora \
#     --max-lora-rank ${MAX_LORA_RANK} \
#     --lora-modules finllama=${FINLLAMA_LORA_PATH} custom=${CUSTOM_LORA_PATH} \
#     --max-model-len ${MAX_MODEL_LEN} \
#     --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
#     --disable-log-requests
EOF

# Make vLLM script executable
chmod +x ~/scripts/start_vllm.sh

# 7. Final verification
echo ""
echo "[7/7] Final verification..."
echo "Checking Lambda Guest Agent status..."
sudo systemctl --no-pager status lambda-guest-agent* || true

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Lambda Guest Agent:"
echo "  Status: sudo systemctl status lambda-guest-agent*"
echo "  Logs:   sudo journalctl -u lambda-guest-agent"
echo ""
echo "Directory structure created:"
echo "  ~/lora_adapters/custom"
echo "  ~/models"
echo "  ~/scripts"
echo ""
echo "Next steps:"
echo "  1. DONE Update HF_TOKEN in this script with your actual token"
echo "  2. DONE Models will be downloaded automatically"
echo "  3. Run: ~/scripts/start_vllm.sh to start the vLLM server"
echo ""
echo "To run vLLM server:"
echo "  cd ~/scripts && ./start_vllm.sh"
echo ""
echo "Or run in background with nohup:"
echo "  nohup ~/scripts/start_vllm.sh > ~/vllm.log 2>&1 &"
echo ""
