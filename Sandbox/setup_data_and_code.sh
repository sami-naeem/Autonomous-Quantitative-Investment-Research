#!/bin/bash
# =============================================================================
# Lambda Instance: Data & Code Setup
# Run after the base setup (vLLM, models, etc.) is complete.
#
# This script:
#   1. Authenticates with GCS
#   2. Pulls Sharadar data from gs://investment-research/lake/sh/
#   3. Clones only the needed folders from your repo (sparse checkout)
# =============================================================================

set -e

echo "=========================================="
echo "Data & Code Setup for Lambda Instance"
echo "=========================================="

# =============================================================================
# CONFIGURATION — Edit these
# =============================================================================

# GCS
GCS_BUCKET="investment-research"
GCS_DATA_PATH="lake/sh"
LOCAL_DATA_DIR="$HOME/data/sharadar"

# Git repo (pick ONE auth method below)
REPO_URL="https://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@github.com/sami-naeem/Autonomous-Quantitative-Investment-Research.git"
# REPO_URL="git@github.com:YOUR_ORG/YOUR_REPO.git"     # SSH (needs key)
REPO_DIR="$HOME/capstone"

# If using a GitHub personal access token for private repo:
# REPO_URL="https://<YOUR_TOKEN>@github.com/YOUR_ORG/YOUR_REPO.git"

# =============================================================================
# STEP 1: GCS Authentication
# =============================================================================
echo ""
echo "[1/4] Setting up GCS authentication..."

# Option A: Service account key (recommended for automation)
# Upload your key file to the instance first:
#   scp service-account-key.json ubuntu@<IP>:~/
if [ -f "$HOME/service-account-key.json" ]; then
    echo "  Found service account key, activating..."
    gcloud auth activate-service-account --key-file="$HOME/service-account-key.json"
    export GOOGLE_APPLICATION_CREDENTIALS="$HOME/service-account-key.json"
    echo "  ✅ Authenticated via service account"

# Option B: Interactive login (use if no service account key)
else
    echo "  No service account key found at ~/service-account-key.json"
    echo "  Trying gcloud auth login..."
    if command -v gcloud &> /dev/null; then
        gcloud auth login --no-launch-browser
        echo "  ✅ Authenticated via gcloud login"
    else
        echo "  ⚠️  gcloud not found. Installing..."
        sudo apt-get update -qq && sudo apt-get install -y -qq google-cloud-cli
        gcloud auth login --no-launch-browser
    fi
fi

# Verify access
echo "  Verifying bucket access..."
if gsutil ls "gs://${GCS_BUCKET}/${GCS_DATA_PATH}/" > /dev/null 2>&1; then
    echo "  ✅ Can access gs://${GCS_BUCKET}/${GCS_DATA_PATH}/"
else
    echo "  ❌ Cannot access bucket. Check auth and bucket name."
    echo "  Try: gsutil ls gs://${GCS_BUCKET}/"
    exit 1
fi

# =============================================================================
# STEP 2: List what's in the bucket
# =============================================================================
echo ""
echo "[2/4] Contents of gs://${GCS_BUCKET}/${GCS_DATA_PATH}/:"
echo "  ─────────────────────────────────────────"
gsutil ls -l "gs://${GCS_BUCKET}/${GCS_DATA_PATH}/" | head -30
echo "  ─────────────────────────────────────────"

# =============================================================================
# STEP 3: Pull data from GCS
# =============================================================================
echo ""
echo "[3/4] Pulling Sharadar data from GCS..."
mkdir -p "${LOCAL_DATA_DIR}"

# Download everything (CSVs + ZIPs) with parallel transfer
# -m = parallel, -r = recursive
gsutil -m cp -r "gs://${GCS_BUCKET}/${GCS_DATA_PATH}/*" "${LOCAL_DATA_DIR}/"

echo ""
echo "  Downloaded files:"
ls -lh "${LOCAL_DATA_DIR}/"

# Unzip any .zip files
echo ""
ZIP_COUNT=$(find "${LOCAL_DATA_DIR}" -name "*.zip" 2>/dev/null | wc -l)
if [ "$ZIP_COUNT" -gt 0 ]; then
    echo "  Found ${ZIP_COUNT} zip file(s), extracting..."
    for zipfile in "${LOCAL_DATA_DIR}"/*.zip; do
        echo "    Extracting: $(basename ${zipfile})"
        unzip -o -q "${zipfile}" -d "${LOCAL_DATA_DIR}/"
    done
    echo "  ✅ All zips extracted"
else
    echo "  No zip files to extract"
fi

echo ""
echo "  Final data directory:"
ls -lh "${LOCAL_DATA_DIR}/"
echo ""
echo "  ✅ Data ready at: ${LOCAL_DATA_DIR}"

# =============================================================================
# STEP 4: Clone repo (sparse checkout — only needed folders)
# =============================================================================
echo ""
echo "[4/4] Cloning repo (sparse checkout)..."

if [ -d "${REPO_DIR}/.git" ]; then
    echo "  Repo already exists at ${REPO_DIR}, pulling latest..."
    cd "${REPO_DIR}"
    git pull
else
    # Sparse checkout: only clone the folders we need
    git clone --filter=blob:none --sparse "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"

    # Add only the folders you need
    git sparse-checkout set \
        Agents/ \
        sandbox_executor.py \
        langgraph.json \
        requirements.txt \
        .env.example

    # If your folder structure is different, adjust above.
    # Common patterns:
    #   git sparse-checkout add src/Agents/
    #   git sparse-checkout add config/
fi

echo ""
echo "  Repo contents:"
ls -la "${REPO_DIR}/"
echo ""
echo "  ✅ Code ready at: ${REPO_DIR}"

# =============================================================================
# STEP 5: Symlink data into project (so code can find it)
# =============================================================================
echo ""
echo "[5/5] Linking data into project..."

# Create a symlink so your code can reference data/ within the project
ln -sfn "${LOCAL_DATA_DIR}" "${REPO_DIR}/data"
echo "  ✅ Symlinked: ${REPO_DIR}/data → ${LOCAL_DATA_DIR}"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Data:    ${LOCAL_DATA_DIR}"
echo "Code:    ${REPO_DIR}"
echo "Symlink: ${REPO_DIR}/data → ${LOCAL_DATA_DIR}"
echo ""
echo "CSV files available:"
find "${LOCAL_DATA_DIR}" -name "*.csv" -exec basename {} \; | sort
echo ""
echo "Next steps:"
echo "  1. cd ${REPO_DIR}"
echo "  2. cp .env.example .env && nano .env  # fill in API keys"
echo "  3. pip install -r requirements.txt --break-system-packages"
echo "  4. langgraph dev"
echo ""
echo "=========================================="
