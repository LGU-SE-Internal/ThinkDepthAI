#!/usr/bin/env bash
set -euo pipefail

# Setup script for RCA evaluation environment on a fresh machine.
# This script:
#   1. Clones the ThinkDepthAI repository
#   2. Installs dependencies with uv
#   3. Downloads the dataset from HuggingFace
#   4. Initializes eval.db from the dataset
#   5. Downloads telemetry data (optional)
#   6. Verifies the setup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_NAME="${DATASET_NAME:-lincyaw/rca}"
DB_PATH="${DB_PATH:-eval.db}"
TELEMETRY_DIR="${TELEMETRY_DIR:-data}"

echo "=== RCA Evaluation Environment Setup ==="
echo "Project dir: $PROJECT_DIR"
echo "Dataset: $DATASET_NAME"
echo "DB path: $DB_PATH"
echo ""

# --- 1. Check prerequisites ---
echo "[1/6] Checking prerequisites..."

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "  uv: OK"

# --- 2. Install dependencies ---
echo ""
echo "[2/6] Installing dependencies..."
cd "$PROJECT_DIR"
uv sync
echo "  Dependencies installed."

# --- 3. Download dataset metadata from HuggingFace ---
echo ""
echo "[3/6] Downloading dataset metadata from HuggingFace..."

HF_JSONL="${TELEMETRY_DIR}/data.jsonl"
if [ -f "$HF_JSONL" ]; then
    echo "  Found existing $HF_JSONL, skipping download."
else
    mkdir -p "$TELEMETRY_DIR"
    uv run python3 -c "
from datasets import load_dataset
import json

ds = load_dataset('${DATASET_NAME}', split='train')
print(f'Downloaded {len(ds)} samples')

with open('${HF_JSONL}', 'w') as f:
    for item in ds:
        f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')

print(f'Written to ${HF_JSONL}')
"
fi

# --- 4. Initialize eval.db ---
echo ""
echo "[4/6] Initializing eval.db..."
uv run python3 "$SCRIPT_DIR/init_eval_db.py" \
    --jsonl "$HF_JSONL" \
    --db "$DB_PATH"
echo "  eval.db initialized."

# --- 5. Download telemetry data (optional) ---
echo ""
echo "[5/6] Checking telemetry data..."

if [ -d "$TELEMETRY_DIR" ] && [ "$(ls -A "$TELEMETRY_DIR" | grep -v 'data.jsonl' | wc -l)" -gt 0 ]; then
    echo "  Telemetry data already exists in $TELEMETRY_DIR, skipping download."
else
    echo "  Telemetry data not found locally."
    echo "  To download from HuggingFace, run:"
    echo "    uv run python3 -c \"from datasets import load_dataset; ds = load_dataset('${DATASET_NAME}')\""
    echo "  Or manually download and extract to: $TELEMETRY_DIR/"
    echo "  Then update config/eval/openrca2_lite.yaml source_path to point to $TELEMETRY_DIR"
fi

# --- 6. Verify setup ---
echo ""
echo "[6/6] Verifying setup..."

uv run python3 -c "
from sqlmodel import create_engine, text
engine = create_engine('sqlite:///${DB_PATH}')
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM data'))
    count = result.fetchone()[0]
    result = conn.execute(text('SELECT tags FROM data LIMIT 1'))
    tags = result.fetchone()[0]
    print(f'  Database: {count} samples')
    print(f'  Sample tags: {tags}')
"

# --- Summary ---
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Configure API keys in .env:"
echo "     UTU_LLM_TYPE=chat.completions"
echo "     UTU_LLM_MODEL=<your-model>"
echo "     UTU_LLM_BASE_URL=<your-api-url>"
echo "     UTU_LLM_API_KEY=<your-api-key>"
echo ""
echo "  2. Update config/eval/openrca2_lite.yaml:"
echo "     db_url: \"sqlite:///${DB_PATH}\""
echo "     source_path: \"/path/to/telemetry/data\""
echo ""
echo "  3. Run evaluation:"
echo "     uv run rca llm-eval run config/eval/openrca2_lite.yaml -a thinkdepthai --source-path /path/to/telemetry/data"
echo ""
