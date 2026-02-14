#!/bin/bash
# =============================================================================
# Lambda Instance: Full Setup (vLLM + Data + Code)
# Single script — replaces running setup_lambda_docker.sh + setup_data_and_code.sh
#
# WARNING: If this file was edited or copied on a Windows PC, it will have
#   CRLF line endings (\r\n) which cause silent failures on Linux.
#   You MUST run:  sed -i 's/\r$//' <filename>  BEFORE executing the script.
#   The commands below already include this step.
#
# HOW TO USE, SEE SSH SCRIPT FOR MORE INFO:
#   Option A — Terminal (scp + ssh):
#     scp -i "$KEY" setup_lambda_all.sh ubuntu@$IP:~/setup.sh
#     ssh -i "$KEY" ubuntu@$IP "sed -i 's/\r$//' ~/setup.sh && bash ~/setup.sh"
#
#   Option B — Lambda Jupyter Notebook (drag & drop):
#     1. Open Lambda's Jupyter UI in your browser
#     2. Drag & drop this file into the home folder
#     3. Open a Terminal tab in Jupyter
#     4. Run:  sed -i 's/\r$//' ~/setup_lambda_all.sh && bash ~/setup_lambda_all.sh
#
# WHAT THIS DOES:
#   Phase 1 — Docker + vLLM + Models
#     - Docker group fix, Lambda Guest Agent, uv + venv
#     - Pulls vLLM Docker image (auto-detects x86/ARM)
#     - Downloads Llama 3.1 8B Instruct (base) from HuggingFace
#     - Downloads Fino1-8B (finllama) from HuggingFace
#     - Generates startup scripts in ~/scripts/
#
#   Phase 2 — Data + LoRA + Code
#     - Authenticates with GCS, pulls Sharadar data
#     - Pulls custom LoRA adapters from GCS
#     - Sparse-checkout clones the repo
#     - Symlinks data into the project
#
# 3 MODELS AVAILABLE AFTER SETUP:
#   base     = meta-llama/Llama-3.1-8B-Instruct  (~/models/llama-3.1-8b)
#   finllama = TheFinAI/Fino1-8B                   (~/models/finllama)
#   custom   = Your LoRA fine-tuned adapter        (~/lora_adapters/custom-lora)
# =============================================================================

set -e

echo "=========================================="
echo "  Lambda Full Setup: vLLM + Data + Code"
echo "=========================================="

# =============================================================================
# CONFIGURATION
# =============================================================================
#SEBASTIANS HF TOKEN
HF_TOKEN="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"

# Models
BASE_MODEL_REPO="meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_NAME="llama-3.1-8b"
FINLLAMA_MODEL_REPO="TheFinAI/Fino1-8B"
FINLLAMA_MODEL_NAME="finllama"

# Custom LoRA adapter (pulled from GCS, NOT the same as finllama)
CUSTOM_LORA_NAME="custom-lora"

# vLLM Docker image — auto-detected by architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    VLLM_IMAGE="${VLLM_IMAGE:-drikster80/vllm-gh200-openai}"
    MODEL_FLAG=""
    echo "Detected aarch64 (GH200/ARM) — image: ${VLLM_IMAGE}"
else
    VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
    MODEL_FLAG="--model "
    echo "Detected ${ARCH} — image: ${VLLM_IMAGE}"
fi

# vLLM serving
API_KEY="token-abc123"
PORT=8000

# GCS
GCS_BUCKET="investment-research"
GCS_DATA_PATH="lake/sh"
GCS_LORA_PATH="Fine_Tuning/lora_adapters"
LOCAL_DATA_DIR="$HOME/data/sharadar"

# Git repo
REPO_URL="https://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@github.com/sami-naeem/Autonomous-Quantitative-Investment-Research.git"
REPO_DIR="$HOME/capstone"

# =============================================================================
# PHASE 1: Docker + vLLM + Models
# =============================================================================
echo ""
echo "══════════════════════════════════════════"
echo "  PHASE 1: Docker + vLLM + Models"
echo "══════════════════════════════════════════"

# --- Step 1: Docker group ---
echo ""
echo "[1/13] Ensuring Docker access..."
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
    exec sg docker "$0"
fi
echo "  Docker access confirmed."

# --- Step 2: Lambda Guest Agent ---
echo ""
echo "[2/13] Installing Lambda Guest Agent..."
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash
sudo systemctl --no-pager status lambda-guest-agent* || echo "Warning: Guest Agent may not have started correctly"

# --- Step 3: uv + venv ---
echo ""
echo "[3/13] Installing uv and creating virtual environment..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --system-site-packages ~/vllm_env
source ~/vllm_env/bin/activate

# --- Step 4: Python dependencies ---
echo ""
echo "[4/13] Installing Python dependencies..."
uv pip install --reinstall huggingface_hub
uv pip install "numpy>1.22.4"
uv pip install peft

# --- Step 5: Pull Docker image ---
echo ""
echo "[5/13] Pulling vLLM Docker image: ${VLLM_IMAGE}..."
docker pull ${VLLM_IMAGE}
docker inspect ${VLLM_IMAGE} > /dev/null 2>&1 && echo "Docker image ready." || { echo "ERROR: Failed to pull Docker image"; exit 1; }

# --- Step 6: HuggingFace login + model downloads ---
echo ""
echo "[6/13] Logging into HuggingFace..."
export HF_TOKEN=$HF_TOKEN
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

mkdir -p ~/models/${BASE_MODEL_NAME}
mkdir -p ~/models/${FINLLAMA_MODEL_NAME}
mkdir -p ~/lora_adapters/${CUSTOM_LORA_NAME}
mkdir -p ~/scripts

# --- Step 7: Download base model ---
echo ""
echo "[7/13] Downloading base model: $BASE_MODEL_REPO..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$BASE_MODEL_REPO', local_dir='$HOME/models/${BASE_MODEL_NAME}')"

# --- Step 8: Download Fino1-8B (finllama) ---
echo ""
echo "[8/13] Downloading Fino1-8B model: $FINLLAMA_MODEL_REPO..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$FINLLAMA_MODEL_REPO', local_dir='$HOME/models/${FINLLAMA_MODEL_NAME}')"

echo "  Phase 1 complete."

# =============================================================================
# PHASE 2: Data + LoRA + Code
# =============================================================================
echo ""
echo "══════════════════════════════════════════"
echo "  PHASE 2: Data + Custom LoRA + Code"
echo "══════════════════════════════════════════"

# --- Step 9: GCS Authentication ---
echo ""
echo "[9/13] Setting up GCS authentication..."
if [ -f "$HOME/service-account-key.json" ]; then
    echo "  Found service account key, activating..."
    gcloud auth activate-service-account --key-file="$HOME/service-account-key.json"
    export GOOGLE_APPLICATION_CREDENTIALS="$HOME/service-account-key.json"
    echo "  Authenticated via service account."
else
    echo "  No service account key found at ~/service-account-key.json"
    echo "  Trying gcloud auth login..."
    if command -v gcloud &> /dev/null; then
        gcloud auth login --no-launch-browser
    else
        echo "  gcloud not found. Installing..."
        sudo apt-get update -qq && sudo apt-get install -y -qq google-cloud-cli
        gcloud auth login --no-launch-browser
    fi
fi

echo "  Verifying bucket access..."
if gsutil ls "gs://${GCS_BUCKET}/${GCS_DATA_PATH}/" > /dev/null 2>&1; then
    echo "  Can access gs://${GCS_BUCKET}/${GCS_DATA_PATH}/"
else
    echo "  ERROR: Cannot access bucket. Check auth and bucket name."
    exit 1
fi

# --- Step 10: Pull Sharadar data ---
echo ""
echo "[10/13] Pulling Sharadar data from GCS..."
mkdir -p "${LOCAL_DATA_DIR}"
gsutil -m cp -r "gs://${GCS_BUCKET}/${GCS_DATA_PATH}/*" "${LOCAL_DATA_DIR}/"

echo "  Downloaded files:"
ls -lh "${LOCAL_DATA_DIR}/"

ZIP_COUNT=$(find "${LOCAL_DATA_DIR}" -name "*.zip" 2>/dev/null | wc -l)
if [ "$ZIP_COUNT" -gt 0 ]; then
    echo "  Found ${ZIP_COUNT} zip file(s), extracting..."
    for zipfile in "${LOCAL_DATA_DIR}"/*.zip; do
        echo "    Extracting: $(basename ${zipfile})"
        unzip -o -q "${zipfile}" -d "${LOCAL_DATA_DIR}/"
    done
    echo "  All zips extracted."
fi

# --- Step 11: Pull custom LoRA adapters from GCS ---
echo ""
echo "[11/13] Pulling custom LoRA adapters from GCS..."
mkdir -p ~/lora_adapters/${CUSTOM_LORA_NAME}
gsutil -m cp -r "gs://${GCS_BUCKET}/${GCS_LORA_PATH}/*" ~/lora_adapters/${CUSTOM_LORA_NAME}/

echo "  Custom LoRA adapter files:"
ls -lh ~/lora_adapters/${CUSTOM_LORA_NAME}/

# --- Step 12: Clone repo (sparse checkout) ---
echo ""
echo "[12/13] Cloning repo (sparse checkout)..."
if [ -d "${REPO_DIR}/.git" ]; then
    echo "  Repo already exists at ${REPO_DIR}, pulling latest..."
    cd "${REPO_DIR}"
    git pull
else
    git clone --filter=blob:none --sparse "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
    git sparse-checkout set --no-cone \
        Agents/ \
        Sandbox/ \
        langgraph.json \
        requirements.txt \
        endpoints \
        .env.example
fi

echo "  Repo contents:"
ls -la "${REPO_DIR}/"

# Symlink data into project
ln -sfn "${LOCAL_DATA_DIR}" "${REPO_DIR}/data"
echo "  Symlinked: ${REPO_DIR}/data -> ${LOCAL_DATA_DIR}"

# --- Step 13: Generate vLLM startup scripts ---
echo ""
echo "[13/13] Creating vLLM startup scripts..."

# -----------------------------------------------
# start_vllm_base.sh — Base Llama 3.1 8B only
# -----------------------------------------------
cat > ~/scripts/start_vllm_base.sh << SCRIPT_EOF
#!/bin/bash
# vLLM: Base Llama 3.1 8B Instruct (no LoRA, no fine-tuning)
# API model name: /models/${BASE_MODEL_NAME}

# Ensure docker group is active in this shell
if ! docker info > /dev/null 2>&1; then
    echo "Docker not accessible. Activating docker group..."
    exec sg docker "\$0"
fi

docker stop vllm-server 2>/dev/null && docker rm vllm-server 2>/dev/null || true
echo "Starting vLLM — base model (${BASE_MODEL_NAME})..."
docker run -d --name vllm-server \\
    --gpus all --ipc=host \\
    -v \$HOME/models:/models \\
    -p ${PORT}:${PORT} \\
    ${VLLM_IMAGE} \\
    ${MODEL_FLAG}/models/${BASE_MODEL_NAME} \\
    --host 0.0.0.0 --port ${PORT} \\
    --api-key "${API_KEY}" \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 18192 \\
    --enable-auto-tool-choice \\
    --tool-call-parser llama3_json \\
    --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
    --disable-log-requests
echo "Started. Logs: docker logs -f vllm-server"
SCRIPT_EOF

# -----------------------------------------------
# start_vllm_finllama.sh — Fino1-8B (financial reasoning model)
# -----------------------------------------------
cat > ~/scripts/start_vllm_finllama.sh << SCRIPT_EOF
#!/bin/bash
# vLLM: Fino1-8B model (standalone, downloaded from TheFinAI/Fino1-8B)
# API model name: /models/${FINLLAMA_MODEL_NAME}

# Ensure docker group is active in this shell
if ! docker info > /dev/null 2>&1; then
    echo "Docker not accessible. Activating docker group..."
    exec sg docker "\$0"
fi

docker stop vllm-server 2>/dev/null && docker rm vllm-server 2>/dev/null || true
echo "Starting vLLM — finllama (Fino1-8B)..."
docker run -d --name vllm-server \\
    --gpus all --ipc=host \\
    -v \$HOME/models:/models \\
    -p ${PORT}:${PORT} \\
    ${VLLM_IMAGE} \\
    ${MODEL_FLAG}/models/${FINLLAMA_MODEL_NAME} \\
    --host 0.0.0.0 --port ${PORT} \\
    --api-key "${API_KEY}" \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 18192 \\
    --enable-auto-tool-choice \\
    --tool-call-parser llama3_json \\
    --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
    --disable-log-requests
echo "Started. Logs: docker logs -f vllm-server"
SCRIPT_EOF

# -----------------------------------------------
# start_vllm_custom.sh — Base + custom LoRA adapter
# -----------------------------------------------
cat > ~/scripts/start_vllm_custom.sh << SCRIPT_EOF
#!/bin/bash
# vLLM: Base Llama 3.1 8B + custom LoRA adapter
# API model name: "custom" for LoRA, or /models/${BASE_MODEL_NAME} for base

# Ensure docker group is active in this shell
if ! docker info > /dev/null 2>&1; then
    echo "Docker not accessible. Activating docker group..."
    exec sg docker "\$0"
fi

docker stop vllm-server 2>/dev/null && docker rm vllm-server 2>/dev/null || true
echo "Starting vLLM — base + custom LoRA..."
docker run -d --name vllm-server \\
    --gpus all --ipc=host \\
    -v \$HOME/models:/models \\
    -v \$HOME/lora_adapters:/lora_adapters \\
    -p ${PORT}:${PORT} \\
    ${VLLM_IMAGE} \\
    ${MODEL_FLAG}/models/${BASE_MODEL_NAME} \\
    --host 0.0.0.0 --port ${PORT} \\
    --api-key "${API_KEY}" \\
    --enable-lora \\
    --lora-modules custom=/lora_adapters/${CUSTOM_LORA_NAME} \\
    --max-lora-rank 64 \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --enable-auto-tool-choice \\
    --tool-call-parser llama3_json \\
    --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
    --max-model-len 18192 \\
    --disable-log-requests
echo "Started. Use model='custom' in API calls for LoRA."
echo "Logs: docker logs -f vllm-server"
SCRIPT_EOF

# -----------------------------------------------
# merge_lora.py — Merge custom LoRA into base
# -----------------------------------------------
cat > ~/scripts/merge_lora.py << 'SCRIPT_EOF'
#!/usr/bin/env python3
"""Merge custom LoRA adapter into base model for standalone deployment."""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL_PATH = os.path.expanduser("~/models/llama-3.1-8b")
LORA_ADAPTER_PATH = os.path.expanduser("~/lora_adapters/custom-lora")
MERGED_MODEL_PATH = os.path.expanduser("~/models/custom-merged")

print(f"Loading base model from: {BASE_MODEL_PATH}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

print(f"Loading custom LoRA adapter from: {LORA_ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_MODEL_PATH)
print(f"Done! Saved to: {MERGED_MODEL_PATH}")
print("Run: ~/scripts/start_vllm_custom_merged.sh")
SCRIPT_EOF

# -----------------------------------------------
# start_vllm_custom_merged.sh — Merged custom model
# -----------------------------------------------
cat > ~/scripts/start_vllm_custom_merged.sh << SCRIPT_EOF
#!/bin/bash
# vLLM: Custom model (LoRA merged into base — standalone, no --enable-lora needed)
# API model name: /models/custom-merged
docker stop vllm-server 2>/dev/null && docker rm vllm-server 2>/dev/null || true
echo "Starting vLLM — custom merged model..."
docker run -d --name vllm-server \\
    --gpus all --ipc=host \\
    -v \$HOME/models:/models \\
    -p ${PORT}:${PORT} \\
    ${VLLM_IMAGE} \\
    ${MODEL_FLAG}/models/custom-merged \\
    --host 0.0.0.0 --port ${PORT} \\
    --api-key "${API_KEY}" \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 18192 \\
    --enable-auto-tool-choice \\
    --tool-call-parser llama3_json \\
    --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
    --disable-log-requests
echo "Started. Logs: docker logs -f vllm-server"
SCRIPT_EOF

# -----------------------------------------------
# stop_vllm.sh
# -----------------------------------------------
cat > ~/scripts/stop_vllm.sh << 'SCRIPT_EOF'
#!/bin/bash
if docker ps -q -f name=vllm-server | grep -q .; then
    echo "Stopping vllm-server..."
    docker stop vllm-server && docker rm vllm-server
    echo "Stopped."
else
    echo "No running vllm-server container found."
    docker rm vllm-server 2>/dev/null || true
fi
SCRIPT_EOF

chmod +x ~/scripts/*.sh ~/scripts/*.py

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "  MODELS:"
echo "    base     ~/models/${BASE_MODEL_NAME}           (Llama 3.1 8B Instruct)"
echo "    finllama ~/models/${FINLLAMA_MODEL_NAME}             (Fino1-8B)"
echo "    custom   ~/lora_adapters/${CUSTOM_LORA_NAME}  (your LoRA fine-tune)"
echo ""
echo "  DATA:    ${LOCAL_DATA_DIR}"
echo "  CODE:    ${REPO_DIR}"
echo "  DOCKER:  ${VLLM_IMAGE} (${ARCH})"
echo ""
echo "=========================================="
echo "  START vLLM — pick one:"
echo "=========================================="
echo ""
echo "  1) Base model (untrained Llama 3.1 8B):"
echo "     ~/scripts/start_vllm_base.sh"
echo ""
echo "  2) Fino1-8B (finllama):"
echo "     ~/scripts/start_vllm_finllama.sh"
echo ""
echo "  3) Custom LoRA (base + your fine-tuned adapter):"
echo "     ~/scripts/start_vllm_custom.sh"
echo "     (use model='custom' in API calls)"
echo ""
echo "  4) Custom merged (LoRA baked into base — recommended):"
echo "     source ~/vllm_env/bin/activate"
echo "     python ~/scripts/merge_lora.py"
echo "     ~/scripts/start_vllm_custom_merged.sh"
echo ""
echo "=========================================="
echo "  USEFUL COMMANDS"
echo "=========================================="
echo ""
echo "  Monitor:  docker logs -f vllm-server"
echo "  Test:     curl http://localhost:${PORT}/v1/models -H 'Authorization: Bearer ${API_KEY}'"
echo "  Stop:     ~/scripts/stop_vllm.sh"
echo ""
echo "=========================================="
