#!/bin/bash
#SBATCH --job-name=bentxt-clair
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ ! -f .env ]; then
    echo "Missing .env."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

LLAMA_BIN="${LLAMA_BIN:-$HOME/.local/bin/llama}"
if [ -z "${EVALUATION_OUTPUT_ROOT:-}" ] || [ -z "${CLAIR_MODEL_PATH:-}" ]; then
    echo "EVALUATION_OUTPUT_ROOT and CLAIR_MODEL_PATH are required."
    exit 1
fi
if [ ! -x "$LLAMA_BIN" ]; then
    echo "llama executable is unavailable: $LLAMA_BIN"
    exit 1
fi
if [ ! -f "$CLAIR_MODEL_PATH" ]; then
    echo "CLAIR GGUF is unavailable: $CLAIR_MODEL_PATH"
    exit 1
fi

output_dir="${EVALUATION_OUTPUT_ROOT%/}/clair_${SLURM_JOB_ID}"
mkdir -p "$output_dir"
port=$((20000 + SLURM_JOB_ID % 20000))
base_url="http://127.0.0.1:${port}/v1"
server_log="$output_dir/llama_server.log"
llama_version="$($LLAMA_BIN version 2>&1 | head -n 1)"

echo "CLAIR output: $output_dir"
echo "Judge model: $CLAIR_MODEL_PATH"
echo "llama.cpp: $llama_version"
echo "Server URL: $base_url"
echo "Scorer arguments: $*"

"$LLAMA_BIN" serve \
    --model "$CLAIR_MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$port" \
    --n-gpu-layers 999 \
    --ctx-size 16384 \
    --parallel 8 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --jinja \
    >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "llama server exited during startup."
        tail -n 100 "$server_log"
        exit 1
    fi
    sleep 2
done
if ! curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "llama server did not become healthy."
    tail -n 100 "$server_log"
    exit 1
fi

uv run python scripts/score_bentxt_clair.py \
    --output-dir "$output_dir" \
    --base-url "$base_url" \
    --model-path "$CLAIR_MODEL_PATH" \
    --llama-version "$llama_version" \
    "$@"
