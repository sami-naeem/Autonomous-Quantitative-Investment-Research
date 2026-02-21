#!/bin/bash
# =============================================================================
# Lambda Instance: Data + Repo Setup (No vLLM)
# Use this on fresh Lambda instances when you only need:
#   - Sharadar data from GCS
#   - GitHub repo checkout (sparse)
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
log "Starting Lambda data+repo setup (no vLLM)..."

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

# Install sandbox-common imports up front to reduce runtime pip in SandboxExecutor.
# These map to IMPORT_TO_PYPI in Sandbox/sandbox_executor_retry.py.
log "Installing sandbox-common extras up front..."
uv pip install \
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
# DONE
# =============================================================================
log "Setup complete."
echo ""
echo "Data: $LOCAL_DATA_DIR"
echo "Repo: $REPO_DIR"
echo ""
echo "Next steps:"
echo "  cd $REPO_DIR"
echo "  source $VENV_DIR/bin/activate"
