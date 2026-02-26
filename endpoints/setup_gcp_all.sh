#!/bin/bash
# =============================================================================
# GCP Instance: Full Setup (vLLM + Data + Code)
# Direct port of setup_lambda_all.sh for Google Cloud Platform VM instances.
#
# WARNING: If edited on Windows, strip CRLF before running:
#   sed -i 's/\r$//' ~/setup.sh && bash ~/setup.sh
#
# PREREQUISITES (GCP Console / gcloud — do these BEFORE running):
#   1. Create a GPU VM:
#        Recommended: a2-highgpu-1g (A100 40GB, x86_64)
#        OR:          g2-standard-4  (L4 24GB — works for 8B at lower cost)
#      Use a Deep Learning VM image (e.g. pytorch-latest-gpu) to get Docker +
#      NVIDIA drivers + NVIDIA Container Toolkit pre-installed.
#      This script installs them automatically if missing (plain Ubuntu works too).
#
#   2. GCS Access — pick one:
#        Option A (preferred): Attach a service account with
#                              roles/storage.objectViewer on gs://investment-research
#                              to the VM at creation time. No key file needed.
#        Option B: Upload service-account-key.json to ~/ after SSH-ing in.
#
#   3. Open firewall port 8000 for the vLLM API:
#        gcloud compute firewall-rules create allow-vllm \
#          --allow tcp:8000 \
#          --source-ranges=<YOUR_IP>/32 \
#          --target-tags=vllm-server
#        Then add the network tag 'vllm-server' to your VM instance.
#
# HOW TO USE:
#   Option A — Terminal:
#     scp -i "$KEY" endpoints/setup_gcp_all.sh user@$EXTERNAL_IP:~/setup_gcp_all.sh
#     ssh -i "$KEY" user@$EXTERNAL_IP "sed -i 's/\r$//' ~/setup_gcp_all.sh && bash ~/setup_gcp_all.sh"
#
#   Option B — GCP Cloud Shell / Serial Console:
#     Upload file, then: sed -i 's/\r$//' ~/setup_gcp_all.sh && bash ~/setup_gcp_all.sh
#
# WHAT THIS DOES:
#   Phase 1 — Docker + vLLM + Models
#     - Docker CE + NVIDIA Container Toolkit (installs if not present)
#     - uv + venv
#     - Pulls vLLM Docker image (auto-detects x86/ARM)
#     - Patches partial-json-parser in vLLM image
#     - Downloads Llama 3.1 8B Instruct (base) from HuggingFace
#     - Downloads Fino1-8B (finllama) from HuggingFace
#     - Generates startup scripts in ~/scripts/
#
#   Phase 2 — Data + LoRA + Code
#     - Authenticates with GCS (ADC via instance SA, or key file)
#     - Pulls Sharadar data from GCS
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
echo "  GCP Full Setup: vLLM + Data + Code"
echo "=========================================="

# =============================================================================
# CONFIGURATION
# =============================================================================
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
    echo "Detected aarch64 (ARM) — image: ${VLLM_IMAGE}"
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

# --- Step 1: Docker CE ---
echo ""
echo "[1/12] Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
    echo "  Docker not found. Installing Docker CE..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    sudo systemctl enable --now docker
    echo "  Docker CE installed."
else
    echo "  Docker already installed: $(docker --version)"
fi

if ! docker info > /dev/null 2>&1; then
    echo "  Adding $USER to docker group..."
    sudo usermod -aG docker $USER
    sg docker -c "docker info > /dev/null 2>&1" && echo "  Docker access confirmed." || {
        echo "  ERROR: Could not get Docker access. Please run:"
        echo "    sudo usermod -aG docker \$USER && newgrp docker"
        echo "  Then re-run this script."
        exit 1
    }
    exec sg docker "$0"
fi
echo "  Docker access confirmed."

# --- Step 2: NVIDIA Container Toolkit ---
echo ""
echo "[2/12] Checking NVIDIA Container Toolkit..."
if ! dpkg -l nvidia-container-toolkit >/dev/null 2>&1; then
    echo "  Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "  NVIDIA Container Toolkit installed and Docker restarted."
else
    echo "  NVIDIA Container Toolkit already installed."
fi

# --- Step 3: uv + venv ---
echo ""
echo "[3/12] Installing uv and creating virtual environment..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Try exact Python 3.12.12 first, fall back to any 3.12
uv venv -p python3.12.12 ~/vllm_env 2>/dev/null || uv venv -p python3.12 ~/vllm_env
source ~/vllm_env/bin/activate

# --- Step 4: Python dependencies ---
echo ""
echo "[4/12] Installing Python dependencies..."
uv pip install --reinstall huggingface_hub
uv pip install "numpy>1.22.4"
uv pip install peft

# --- Step 5: Pull Docker image ---
echo ""
echo "[5/12] Pulling vLLM Docker image: ${VLLM_IMAGE}..."
docker pull ${VLLM_IMAGE}
docker inspect ${VLLM_IMAGE} > /dev/null 2>&1 \
    && echo "  Docker image ready." \
    || { echo "  ERROR: Failed to pull Docker image"; exit 1; }

# --- Step 5b: Patch partial-json-parser inside vLLM image ---
# Fixes: JSONDecodeError: Unterminated string in llama_tool_parser.py
# Cause: partial-json-parser <= 0.2.x crashes on `{"` (2-char streaming buffer).
# Fix:   Build a thin wrapper image with a real PyPI release (0.2.1.1.post7) installed.
#        VLLM_IMAGE is updated so all generated startup scripts use the patched image.
echo ""
echo "[5b/12] Patching partial-json-parser in vLLM image..."
PATCHED_IMAGE="vllm-patched:latest"
docker build -t "${PATCHED_IMAGE}" - << DOCKERFILE_EOF
FROM ${VLLM_IMAGE}
RUN pip install --upgrade "partial-json-parser==0.2.1.1.post7" --quiet --no-cache-dir
DOCKERFILE_EOF
VLLM_IMAGE="${PATCHED_IMAGE}"
echo "  Patched image built: ${VLLM_IMAGE}"
echo "  All startup scripts will use this image."

# --- Step 6: HuggingFace login + model downloads ---
echo ""
echo "[6/12] Logging into HuggingFace..."
export HF_TOKEN=$HF_TOKEN
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

mkdir -p ~/models/${BASE_MODEL_NAME}
mkdir -p ~/models/${FINLLAMA_MODEL_NAME}
mkdir -p ~/lora_adapters/${CUSTOM_LORA_NAME}
mkdir -p ~/scripts

# --- Step 7: Download base model ---
echo ""
echo "[7/12] Downloading base model: $BASE_MODEL_REPO..."
export HF_HUB_ENABLE_HF_TRANSFER=0
export BASE_MODEL_REPO=$BASE_MODEL_REPO
export BASE_MODEL_NAME=$BASE_MODEL_NAME

python - <<'PY'
from huggingface_hub import snapshot_download
import os

repo = os.environ["BASE_MODEL_REPO"]
name = os.environ["BASE_MODEL_NAME"]
local_dir = os.path.expanduser(f'~/models/{name}')

snapshot_download(
    repo_id=repo,
    local_dir=local_dir,
    resume_download=True,
    max_workers=1,
    local_dir_use_symlinks=False,
)
print("Done:", local_dir)
PY

# --- Step 8: Download Fino1-8B (finllama) ---
echo ""
echo "[8/12] Downloading Fino1-8B model: $FINLLAMA_MODEL_REPO..."
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
echo "[9/12] Setting up GCS authentication..."
# On GCP VMs: if the instance has a service account with GCS read permissions
# attached, Application Default Credentials (ADC) work automatically via the
# metadata server — no key file needed.
# If a key file is present it takes priority (useful for cross-project access).

if ! command -v gcloud >/dev/null 2>&1 || ! command -v gsutil >/dev/null 2>&1; then
    echo "  Installing Google Cloud CLI..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq google-cloud-cli
fi

GCS_KEY_FILE=""
if [ -f "$HOME/service-account-key.json" ]; then
    GCS_KEY_FILE="$HOME/service-account-key.json"
    echo "  Found: ~/service-account-key.json"
elif [ -f "$HOME/credentials.json" ]; then
    GCS_KEY_FILE="$HOME/credentials.json"
    echo "  Found: ~/credentials.json"
fi

if [ -n "$GCS_KEY_FILE" ]; then
    echo "  Activating service account from key file..."
    gcloud auth activate-service-account --key-file="$GCS_KEY_FILE"
    export GOOGLE_APPLICATION_CREDENTIALS="$GCS_KEY_FILE"
    if ! grep -qF "GOOGLE_APPLICATION_CREDENTIALS" "$HOME/.bashrc" 2>/dev/null; then
        echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$GCS_KEY_FILE\"" >> "$HOME/.bashrc"
    fi
    echo "  Authenticated via $(basename $GCS_KEY_FILE)"
else
    echo "  No key file found — using GCP instance service account (ADC)."
    echo "  If this fails, ensure the VM has a service account with"
    echo "  roles/storage.objectViewer on gs://${GCS_BUCKET}, then re-run."
fi

echo "  Verifying bucket access..."
if gsutil ls "gs://${GCS_BUCKET}/${GCS_DATA_PATH}/" > /dev/null 2>&1; then
    echo "  Can access gs://${GCS_BUCKET}/${GCS_DATA_PATH}/"
else
    echo "  ERROR: Cannot access bucket. Check IAM permissions or upload a key file."
    exit 1
fi

# --- Step 10: Pull Sharadar data ---
echo ""
echo "[10/12] Pulling Sharadar data from GCS..."
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
echo "[11/12] Pulling custom LoRA adapters from GCS..."
mkdir -p ~/lora_adapters/${CUSTOM_LORA_NAME}
gsutil -m cp -r "gs://${GCS_BUCKET}/${GCS_LORA_PATH}/*" ~/lora_adapters/${CUSTOM_LORA_NAME}/

echo "  Custom LoRA adapter files:"
ls -lh ~/lora_adapters/${CUSTOM_LORA_NAME}/

# --- Step 12: Clone repo (sparse checkout) ---
echo ""
echo "[12/12] Cloning repo (sparse checkout)..."
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
        run_code_replication.py \
        endpoints \
        .env.example
fi

echo "  Repo contents:"
ls -la "${REPO_DIR}/"

# Symlink data into project
ln -sfn "${LOCAL_DATA_DIR}" "${REPO_DIR}/data"
echo "  Symlinked: ${REPO_DIR}/data -> ${LOCAL_DATA_DIR}"

# --- Generate vLLM startup scripts ---
echo ""
echo "[12b] Creating vLLM startup scripts..."

# -----------------------------------------------
# start_vllm_base.sh — Base Llama 3.1 8B only
# -----------------------------------------------
cat > ~/scripts/start_vllm_base.sh << SCRIPT_EOF
#!/bin/bash
# vLLM: Base Llama 3.1 8B Instruct (no LoRA, no fine-tuning)
# API model name: /models/${BASE_MODEL_NAME}

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
    --max-model-len 65536 \\
    --enable-auto-tool-choice \\
    --tool-call-parser llama3_json \\
    --chat-template examples/tool_chat_template_llama3.1_json.jinja \\
    --disable-log-requests
echo "Started. Logs: docker logs -f vllm-server"
SCRIPT_EOF

# -----------------------------------------------
# start_vllm_finllama.sh — Fino1-8B
# -----------------------------------------------
cat > ~/scripts/start_vllm_finllama.sh << SCRIPT_EOF
#!/bin/bash
# vLLM: Fino1-8B model (standalone, downloaded from TheFinAI/Fino1-8B)
# API model name: /models/${FINLLAMA_MODEL_NAME}

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
    --served-model-name finllama \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 65536 \\
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
    --max-model-len 65536 \\
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
    --max-model-len 65536 \\
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
echo "  GCP external IP test:"
echo "  curl http://\$(curl -s metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H 'Metadata-Flavor: Google'):${PORT}/v1/models -H 'Authorization: Bearer ${API_KEY}'"
echo ""
echo "=========================================="
