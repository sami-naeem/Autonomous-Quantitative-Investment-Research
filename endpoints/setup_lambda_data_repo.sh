#!/bin/bash
# =============================================================================
# Lambda Instance: Sandbox Setup (No vLLM)
# Use this on fresh Lambda instances to provision the code-execution sandbox:
#   - Sharadar data from GCS (required for backtest code to run)
#   - GitHub repo checkout (sparse: Agents/, Sandbox/, endpoints/)
#   - Python venv with all pipeline + FastAPI sandbox server dependencies
#   - ~/scripts/start_sandbox_server.sh  — FastAPI /execute endpoint
#   - ~/scripts/start_replication.sh     — batch fallback runner
#
# The sandbox server receives HTTP POST /execute requests from the LangGraph
# pipeline (which runs on a separate machine without Sharadar data), executes
# the generated backtest code in SandboxExecutor, and writes artifacts to GCS.
#
# WARNING: If edited on Windows, strip CRLF before running:
#   sed -i 's/\r$//' ~/setup.sh && bash ~/setup.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
GCS_BUCKET="investment-research"
GCS_DATA_PATH="lake/sh"
LOCAL_DATA_DIR="$HOME/data/sharadar"

# Use a PAT-enabled URL or switch to SSH URL if your key is configured.
REPO_URL="https://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@github.com/sami-naeem/Autonomous-Quantitative-Investment-Research.git"
REPO_DIR="$HOME/capstone"
REPO_BRANCH="main"
VENV_DIR="$REPO_DIR/.venv"

# =============================================================================
# HELPERS
# =============================================================================
log() {
    echo "[setup] $*"
}

ensure_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

# =============================================================================
# START
# =============================================================================
log "Starting Lambda sandbox setup (no vLLM)..."

log "Checking basic tools..."
if ! ensure_cmd git || ! ensure_cmd unzip; then
    log "Installing missing tools: git, unzip"
    sudo apt-get update -qq
    sudo apt-get install -y -qq git unzip
fi

if ! ensure_cmd gcloud || ! ensure_cmd gsutil; then
    log "Installing Google Cloud CLI (gcloud + gsutil)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq google-cloud-cli
fi

if ! ensure_cmd curl; then
    log "Installing missing tool: curl"
    sudo apt-get update -qq
    sudo apt-get install -y -qq curl
fi

# =============================================================================
# STEP 1: GCS AUTH
# =============================================================================
log "Setting up GCS authentication..."

GCS_KEY_FILE=""
if [ -f "$HOME/service-account-key.json" ]; then
    GCS_KEY_FILE="$HOME/service-account-key.json"
    log "Found service-account-key.json"
elif [ -f "$HOME/credentials.json" ]; then
    GCS_KEY_FILE="$HOME/credentials.json"
    log "Found credentials.json"
fi

if [ -n "$GCS_KEY_FILE" ]; then
    log "Activating service account..."
    gcloud auth activate-service-account --key-file="$GCS_KEY_FILE"
    export GOOGLE_APPLICATION_CREDENTIALS="$GCS_KEY_FILE"
    log "Authenticated via $(basename "$GCS_KEY_FILE")"

    # Persist so non-login shells (cron, systemd, background processes) keep auth
    if ! grep -qF "GOOGLE_APPLICATION_CREDENTIALS" "$HOME/.bashrc" 2>/dev/null; then
        echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$GCS_KEY_FILE\"" >> "$HOME/.bashrc"
        log "Persisted GOOGLE_APPLICATION_CREDENTIALS to ~/.bashrc"
    fi
else
    log "No local service account key found; using browserless gcloud login"
    gcloud auth login --no-launch-browser
fi

log "Verifying GCS access..."
if gsutil ls "gs://${GCS_BUCKET}/${GCS_DATA_PATH}/" >/dev/null 2>&1; then
    log "GCS access verified: gs://${GCS_BUCKET}/${GCS_DATA_PATH}/"
else
    log "ERROR: Cannot access gs://${GCS_BUCKET}/${GCS_DATA_PATH}/"
    exit 1
fi

# =============================================================================
# STEP 2: PULL SHARADAR DATA
# =============================================================================
log "Pulling Sharadar data from GCS..."
mkdir -p "$LOCAL_DATA_DIR"
gsutil -m cp -r "gs://${GCS_BUCKET}/${GCS_DATA_PATH}/*" "$LOCAL_DATA_DIR/"

ZIP_COUNT=$(find "$LOCAL_DATA_DIR" -maxdepth 1 -name "*.zip" | wc -l || true)
if [ "$ZIP_COUNT" -gt 0 ]; then
    log "Found ${ZIP_COUNT} zip file(s); extracting..."
    for zipfile in "$LOCAL_DATA_DIR"/*.zip; do
        unzip -o -q "$zipfile" -d "$LOCAL_DATA_DIR/"
    done
    log "Zip extraction complete."
fi

log "Data directory ready: $LOCAL_DATA_DIR"
ls -lh "$LOCAL_DATA_DIR" || true

# =============================================================================
# STEP 3: CLONE OR UPDATE REPO (SPARSE)
# =============================================================================
log "Setting up repository..."
if [ -d "$REPO_DIR/.git" ]; then
    log "Repo exists; pulling latest from ${REPO_BRANCH}"
    cd "$REPO_DIR"
    git fetch origin "$REPO_BRANCH"
    git checkout "$REPO_BRANCH"
    git pull --ff-only origin "$REPO_BRANCH"
else
    log "Cloning repo with sparse checkout..."
    git clone --filter=blob:none --sparse "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout "$REPO_BRANCH"
    git sparse-checkout set --no-cone \
        Agents/ \
        Sandbox/ \
        endpoints/ \
        langgraph.json \
        requirements.txt \
        .env.example \
        run_batch_tasks.py
fi

# Keep sparse set current even for existing clones.
cd "$REPO_DIR"
git sparse-checkout set --no-cone \
    Agents/ \
    Sandbox/ \
    endpoints/ \
    langgraph.json \
    requirements.txt \
    .env.example \
    run_batch_tasks.py

# =============================================================================
# STEP 4: LINK DATA INTO REPO
# =============================================================================
ln -sfn "$LOCAL_DATA_DIR" "$REPO_DIR/data"
log "Symlinked: $REPO_DIR/data -> $LOCAL_DATA_DIR"

# =============================================================================
# STEP 5: UV ENV + UPFRONT DEPENDENCY INSTALL
# =============================================================================
log "Installing uv (if needed) and creating project venv..."
if ! ensure_cmd uv; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
uv venv --python 3.11 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

log "Installing repo requirements with uv..."
uv pip install --upgrade pip
uv pip install -r requirements.txt

# Pipeline runtime deps not always in requirements.txt
log "Installing pipeline + sandbox extras..."
uv pip install \
    rank-bm25 \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    scikit-learn \
    opencv-python \
    pillow \
    pyyaml \
    beautifulsoup4 \
    python-dateutil \
    attrs \
    python-dotenv \
    google-cloud-storage

# =============================================================================
# STEP 6: .ENV SETUP
# =============================================================================
cd "$REPO_DIR"
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    log "Copied .env.example → .env"
    log "ACTION REQUIRED: Edit $REPO_DIR/.env and fill in:"
    log "  OPENAI_API_KEY, LAMBDA_ENDPOINT, GCS_OUTPUT_BUCKET_NAME, SHARADAR_DATA_DIR"
elif [ -f ".env" ]; then
    log ".env already exists — skipping copy."
else
    log "WARNING: no .env.example found; create $REPO_DIR/.env manually."
fi

# Persist SHARADAR_DATA_DIR for run_replication.py
if ! grep -qF "SHARADAR_DATA_DIR" "$HOME/.bashrc" 2>/dev/null; then
    echo "export SHARADAR_DATA_DIR=\"$LOCAL_DATA_DIR\"" >> "$HOME/.bashrc"
    log "Persisted SHARADAR_DATA_DIR=$LOCAL_DATA_DIR to ~/.bashrc"
fi

# =============================================================================
# STEP 7: GENERATE START SCRIPTS
# =============================================================================
mkdir -p "$HOME/scripts"

# --- 7a: batch fallback runner ---
cat > "$HOME/scripts/start_replication.sh" << 'STARTSCRIPT'
#!/bin/bash
# Batch replication fallback: reads A1+A2 GCS outputs, generates + runs code.
# Usage: ./start_replication.sh [model]   default: finllama
set -euo pipefail
REPO_DIR="$HOME/capstone"
VENV_DIR="$REPO_DIR/.venv"
MODEL="${1:-finllama}"

source "$VENV_DIR/bin/activate"
source "$HOME/.bashrc" 2>/dev/null || true
cd "$REPO_DIR"

echo "[start_replication] Using model: $MODEL"
python Agents/run_replication.py --all --model "$MODEL"
STARTSCRIPT
chmod +x "$HOME/scripts/start_replication.sh"
log "Generated: ~/scripts/start_replication.sh"

# --- 7b: FastAPI sandbox server ---
# Receives POST /execute from the LangGraph pipeline, runs SandboxExecutor,
# writes artifacts to GCS. LangGraph machine must set:
#   SANDBOX_ENDPOINT_URL=http://<this-ip>:<port>
cat > "$HOME/scripts/start_sandbox_server.sh" << 'SANDBOXSCRIPT'
#!/bin/bash
# Start the FastAPI sandbox execution server.
# Usage: ./start_sandbox_server.sh [port]   default port: 9000
set -euo pipefail
REPO_DIR="$HOME/capstone"
VENV_DIR="$REPO_DIR/.venv"
PORT="${1:-9000}"

source "$VENV_DIR/bin/activate"
source "$HOME/.bashrc" 2>/dev/null || true
cd "$REPO_DIR"

echo "[sandbox_server] Starting on 0.0.0.0:${PORT}"
echo "[sandbox_server] Sharadar data: ${SHARADAR_DATA_DIR:-$HOME/data/sharadar}"
uvicorn Sandbox.sandbox_server:app --host 0.0.0.0 --port "$PORT"
SANDBOXSCRIPT
chmod +x "$HOME/scripts/start_sandbox_server.sh"
log "Generated: ~/scripts/start_sandbox_server.sh"

# =============================================================================
# DONE
# =============================================================================
log "Setup complete."
echo ""
echo "Data:    $LOCAL_DATA_DIR"
echo "Repo:    $REPO_DIR"
echo "Env:     $REPO_DIR/.env  (edit to fill in API keys)"
echo ""
echo "Start sandbox server:  ~/scripts/start_sandbox_server.sh [port]"
echo "Batch fallback runner: ~/scripts/start_replication.sh [finllama|openai|base]"
echo ""
echo "Next steps:"
echo "  1. Edit $REPO_DIR/.env — set OPENAI_API_KEY, GCS_OUTPUT_BUCKET_NAME,"
echo "                           SHARADAR_DATA_DIR (already in ~/.bashrc)"
echo "  2. source ~/.bashrc"
echo "  3. ~/scripts/start_sandbox_server.sh"
echo "  4. On the LangGraph machine: SANDBOX_ENDPOINT_URL=http://<this-ip>:9000"
