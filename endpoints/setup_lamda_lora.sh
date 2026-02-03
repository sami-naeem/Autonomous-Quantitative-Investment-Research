#!/bin/bash
# Lambda Instance Setup Script for vLLM with Fine-tuned Llama 3.1 8B
# Run this immediately after SSH into the instance

set -e  # Exit on any error

echo "=========================================="
echo "Lambda Instance Setup for vLLM + LoRA"
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

# Expected local files to upload:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
# - README.md

# 0. Install Lambda Guest Agent (for metrics monitoring)
echo ""
echo "[0/8] Installing Lambda Guest Agent..."
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

echo ""
echo "Verifying Guest Agent status..."
sudo systemctl --no-pager status lambda-guest-agent* || echo "Warning: Guest Agent may not have started correctly"

# 1. Install uv package manager for fast installations
echo ""
echo "[1/8] Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create virtual environment
echo ""
echo "[2/8] Creating virtual environment..."
uv venv ~/vllm_env
source ~/vllm_env/bin/activate

# 3. Install vLLM and dependencies
echo ""
echo "[3/8] Installing vLLM and dependencies with uv..."
uv pip install --upgrade vllm huggingface_hub
uv pip install "numpy>1.22.4"
uv pip install peft  # Required for LoRA loading

# 4. Login to Hugging Face
echo ""
echo "[4/8] Logging into Hugging Face..."
export HF_TOKEN=$HF_TOKEN
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# 5. Create directory structure
echo ""
echo "[5/8] Creating directory structure..."
mkdir -p ~/models/${BASE_MODEL_NAME}
mkdir -p ~/lora_adapters/${LORA_ADAPTER_NAME}
mkdir -p ~/scripts

# 6. Download base model
echo ""
echo "[6/8] Downloading base model: $BASE_MODEL_REPO"
huggingface-cli download $BASE_MODEL_REPO --local-dir ~/models/${BASE_MODEL_NAME}

# 7. Setup LoRA adapters
echo ""
echo "[7/8] Setting up LoRA adapters directory..."
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

# 8. Create vLLM startup scripts
echo ""
echo "[8/8] Creating vLLM startup scripts..."

# Script to run BASE model only (no LoRA)
cat > ~/scripts/start_vllm_base.sh << 'EOF'
#!/bin/bash
# vLLM Server - Base Model Only

source $HOME/vllm_env/bin/activate

python -m vllm.entrypoints.openai.api_server \
    --model $HOME/models/llama-3.1-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "token-abc123" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --disable-log-requests
EOF

# Script to run with LoRA adapter
cat > ~/scripts/start_vllm_lora.sh << 'EOF'
#!/bin/bash
# vLLM Server - With LoRA Adapter

source $HOME/vllm_env/bin/activate

# Configuration
BASE_MODEL="$HOME/models/llama-3.1-8b"
LORA_PATH="$HOME/lora_adapters/finllama-lora"
PORT=8000
API_KEY="token-abc123"

python -m vllm.entrypoints.openai.api_server \
    --model ${BASE_MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --api-key ${API_KEY} \
    --enable-lora \
    --lora-modules finllama=${LORA_PATH} \
    --max-lora-rank 64 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --disable-log-requests
EOF

# Script to merge LoRA into base model (creates standalone model)
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
cat > ~/scripts/start_vllm_merged.sh << 'EOF'
#!/bin/bash
# vLLM Server - Merged Fine-tuned Model (no LoRA loading needed)

source $HOME/vllm_env/bin/activate

python -m vllm.entrypoints.openai.api_server \
    --model $HOME/models/finllama-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "token-abc123" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --disable-log-requests
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
echo "4. Test the API:"
echo "   curl http://localhost:8000/v1/models -H 'Authorization: Bearer token-abc123'"
echo ""
echo "=========================================="