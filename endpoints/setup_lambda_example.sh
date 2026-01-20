#!/bin/bash
# Lambda Instance Setup Script for vLLM Multi-LoRA Deployment
# Run this immediately after SSH into the instance

set -e  # Exit on any error

echo "=========================================="
echo "Lambda Instance Setup for vLLM Multi-LoRA"
echo "=========================================="

# Set your Hugging Face token here
HF_TOKEN="REDACTED_FOR_SECURITY"

# 1. Install vLLM and Hugging Face Hub with CLI
echo ""
echo "[1/7] Installing vLLM and Hugging Face Hub..."
pip install --upgrade vllm huggingface_hub[cli]

# 2. Login to Hugging Face
echo ""
echo "[2/6] Logging into Hugging Face..."
# Set HF_TOKEN as environment variable (used by huggingface_hub automatically)
export HF_TOKEN=$HF_TOKEN
# Programmatic login using Python
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# 3. Create directory structure
echo ""
echo "[3/6] Creating directory structure..."
mkdir -p ~/lora_adapters/custom
mkdir -p ~/models
mkdir -p ~/scripts

# 4. Download LoRA adapters (update repo names as needed)
echo ""
echo "[4/6] Downloading LoRA adapters..."
# Syntax: hf download <repo_id> --local-dir <path>
# Uncomment and update these with your actual repo names:
# hf download <YOUR_CUSTOM_REPO> --local-dir ~/lora_adapters/custom

# 5. Download base model (if needed)
echo ""
echo "[5/6] Downloading base model..."
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ~/models/Llama-3.2-3B-Instruct
hf download TheFinAI/FinLLaVA --local-dir ~/models/FinLLaVA
hf download TheFinAI/Fin-o1-14B --local-dir ~/models/Fin-o1-14B
hf download DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning --local-dir ~/models/Llama3

# 6. Create vLLM startup script
echo ""
echo "[6/6] Creating vLLM startup script..."
cat > ~/scripts/start_vllm.sh << 'EOF'
#!/bin/bash
# vLLM Multi-LoRA Server Startup Script

# Using DavidAU model while waiting for Meta Llama access approval
python -m vllm.entrypoints.openai.api_server \
    --model ~/models/Llama3 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key token-abc123 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9

# Alternative: Once Meta approves access, switch to:
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --enable-lora \
#     --lora-modules finllama=~/lora_adapters/finllama custom=~/lora_adapters/custom \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --api-key token-abc123
EOF

# Make vLLM script executable
chmod +x ~/scripts/start_vllm.sh

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
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
